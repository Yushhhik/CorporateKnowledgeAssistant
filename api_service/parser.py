"""Парсер документации (.docx и .pdf) -> чанки для Qdrant.

Что вытаскиваем:
  - текст, разбитый на чанки ~450 символов с перекрытием, с привязкой к разделу;
  - таблицы (каждая строка превращается в "Колонка: значение; ...");
  - картинки (скриншоты, схемы): сохраняем в images/ и описываем текстом
    через vision-модель OpenAI. Пока ключа нет, описание = заглушка из контекста.

Формат чанка:
  {"type": "text" | "image", "source": "файл", "section": "раздел или стр. N",
   "page": номер страницы (для PDF) или None (для docx), "text": "...", "image_path": путь или None}
  У картинок есть ещё "described": True, если описание получено от vision-модели, и False, если там заглушка.

Запуск для проверки:
    python parser.py docs

Что умеет PDF-парсер сверх простого извлечения текста:
  - убирает колонтитулы и повторяющийся мусор экспорта автоматически (строки, повторяющиеся на многих страницах);
  - находит заголовки по размеру и жирности шрифта и подписывает каждый чанк разделом
    («5 Инструкция», «2.1.1 Редактор фильтров»), чтобы в ответе бота был точный источник;
  - таблицы превращает в строки «Колонка: значение», а не в кашу из склеенных ячеек;
  - берёт даже узкие скриншоты (кнопки, панели инструментов);
  - если нет ключа OpenAI, читает текст со скриншотов через tesseract (необязательно: нужен pytesseract + tesseract).
"""
import base64
import hashlib
import json
import os
import re
import shutil
import sys
from io import BytesIO
from pathlib import Path

import pdfplumber
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from PIL import Image

# Версия логики парсера. Меняй при правках parser.py: ingest.py увидит новую версию
# и сам пересчитает все файлы (без ручного --full).
PARSER_VERSION = "4"

CHUNK_SIZE = 450
CHUNK_OVERLAP = 80
MIN_CHUNK_LEN = 40
MIN_IMAGE_MAX_SIDE = 100      # картинки, у которых даже длинная сторона меньше (px), это иконки
MIN_IMAGE_MIN_SIDE = 12       # и совсем тонкие полоски тоже пропускаем

IMAGES_DIR = Path(os.getenv("IMAGES_DIR", "images"))
CACHE_FILE = Path("image_descriptions.json")   # кеш описаний, чтобы не платить за одно и то же дважды
VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")

NOISE_RE = re.compile(r"Ошибка! Используйте вкладку.*?отображаться\.\s*–\s*\d+", re.S)
PAGE_HEADER_RE = re.compile(r"(?m)^ЕИС\s*–.*$")


# ----------------------------------------------------------------- текст

def clean_text(text: str) -> str:
    """Убираем мусор экспорта, переносы слов, лишние пробелы и переводы строк."""
    text = NOISE_RE.sub(" ", text)
    text = PAGE_HEADER_RE.sub(" ", text)
    text = re.sub(r"[\uf000-\uf8ff\u200b]", "", text)   # значки-иконки шрифтов (Wingdings и т.п.), не текст
    text = text.replace("\xa0", " ")
    text = re.sub(r"-\n(?=\w)", "", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Режем текст на куски ~size символов, стараясь резать по концу предложения или по пробелу."""
    chunks = []
    start, n = 0, len(text)

    while start < n:
        end = min(start + size, n)

        if end < n:
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
            if cut < size * 0.5:
                cut = window.rfind(" ")
            if cut >= size * 0.5:
                end = start + cut + 1

        chunk = text[start:end].strip()
        # короткий хвост отбрасываем (он уже покрыт перекрытием), но короткий единственный раздел оставляем
        if len(chunk) >= MIN_CHUNK_LEN or (not chunks and len(chunk) >= 12 and end >= n):
            chunks.append(chunk)

        if end >= n:
            break

        new_start = max(end - overlap, start + 1)
        space = text.find(" ", new_start, end)
        start = space + 1 if space != -1 else new_start

    return chunks


def _is_toc(piece: str) -> bool:
    """Оглавление ("Раздел........5") в базу знаний не нужно."""
    return bool(re.search(r"\.{5,}", piece))


def _text_chunk(source: str, section: str, page, text: str) -> dict:
    return {"type": "text", "source": source, "section": section,
            "page": page, "text": text, "image_path": None}


# ------------------------------------------------------------- картинки

def _load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _describe_with_openai(img: Image.Image, section: str, context: str, doc: str = "") -> str | None:
    try:
        from openai import OpenAI

        buf = BytesIO()
        img.save(buf, "PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        prompt = (
            f"Это изображение из документации корпоративной системы. Документ: {doc}. Раздел: {section}. "
            f"Текст рядом: {context}\n\n"
            "Опиши изображение по-русски для поискового индекса. Если это скриншот интерфейса: "
            "перечисли кнопки, пункты меню, поля и подписи (дословно), как выглядят иконки и где "
            "всё расположено. Если это схема бизнес-процесса: по порядку перечисли шаги, роли, "
            "условия и переходы между ними. Без вступлений, не длиннее 1200 символов."
        )
        resp = OpenAI().chat.completions.create(
            model=VISION_MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
            max_tokens=700,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [vision] не удалось описать картинку: {e}")
        return None


_OCR_STATE: dict = {}


def _ocr_langs() -> str | None:
    """Языки tesseract или None, если OCR недоступен (нет pytesseract или самого tesseract). Проверяем один раз."""
    if "langs" not in _OCR_STATE:
        langs = None
        try:
            if shutil.which("tesseract"):
                import pytesseract

                have = set(pytesseract.get_languages())
                use = [l for l in ("rus", "eng") if l in have]
                langs = "+".join(use) if use else None
        except Exception:
            langs = None
        _OCR_STATE["langs"] = langs
    return _OCR_STATE["langs"]


def _ocr_image(img: Image.Image) -> str:
    """Текст со скриншота через tesseract. Только уверенно распознанные слова, чтобы не тащить мусор."""
    langs = _ocr_langs()
    if not langs:
        return ""
    try:
        import pytesseract

        if img.mode in ("RGBA", "LA", "P"):
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, "white")
            bg.paste(rgba, mask=rgba.split()[3])
            img = bg
        gray = img.convert("L")
        w, h = gray.size
        if h < 200:                                   # узкие полоски с кнопками надо увеличить
            k = min(4, -(-200 // h))
            gray = gray.resize((w * k, h * k), Image.LANCZOS)
        data = pytesseract.image_to_data(gray, lang=langs, config="--psm 6",
                                         output_type=pytesseract.Output.DICT)
        lines: dict = {}
        for i, word in enumerate(data["text"]):
            word = word.strip()
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1
            if len(word) < 2 or conf < 60 or not re.search(r"\w", word):
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(word)
        text = " | ".join(" ".join(ws) for ws in lines.values())
        return text if len(text) >= 6 else ""
    except Exception:
        return ""


def make_image_chunk(img_bytes: bytes, source: str, stem: str, section: str, context: str,
                     seen: set, page=None, skip_if_few_colors: bool = False,
                     native_size: tuple | None = None) -> dict | None:
    """Картинка -> чанк с описанием. Описание: vision-модель OpenAI, иначе OCR, иначе заглушка.

    described=True только если описание получено от vision-модели (ingest.py повторит
    попытку для остальных, когда ключ заработает). method: vision | ocr | placeholder.
    """
    key = hashlib.md5(img_bytes).hexdigest()
    if key in seen:
        return None
    seen.add(key)

    try:
        img = Image.open(BytesIO(img_bytes))
        img.load()
    except Exception:
        return None  # например emf/wmf, которые Pillow не открывает

    nw, nh = native_size or img.size       # размер картинки в файле (для PDF render увеличивает crop)
    if max(nw, nh) < MIN_IMAGE_MAX_SIDE or min(nw, nh) < MIN_IMAGE_MIN_SIDE:
        return None                        # иконки (лупа, значки) не нужны
    if skip_if_few_colors:
        small = img.convert("RGB").resize((64, 64))
        colors = small.getcolors(maxcolors=4096)
        if colors is not None and len(colors) <= 6:
            return None  # пустые заглушки вроде красного крестика

    if img.mode not in ("RGB", "RGBA", "L", "P"):
        img = img.convert("RGB")
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = IMAGES_DIR / f"{stem}_{key[:8]}.png"
    img.save(out_path, "PNG")

    cache = _load_cache()
    cache_key = f"{VISION_MODEL}:{key}"       # у другой модели будет своё описание
    desc = cache.get(cache_key)
    method = "vision" if desc else None
    if not desc and os.getenv("OPENAI_API_KEY"):
        desc = _describe_with_openai(img, section, context, stem)
        if desc:
            cache[cache_key] = desc
            CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
            method = "vision"
    described = bool(desc)                    # True только если описание реально получено от модели
    if not desc:
        ocr = _ocr_image(img)
        if ocr:
            desc, method = f"Скриншот. Текст на изображении: {ocr}", "ocr"
            if context:
                desc += f". Рядом в документе: {context[-200:]}"
    if not desc:
        method = "placeholder"
        desc = f"Скриншот или схема из раздела «{section}». Рядом в документе: {context}"

    return {"type": "image", "source": source, "section": section, "page": page,
            "text": desc, "image_path": str(out_path), "described": described, "method": method}


# ----------------------------------------------------------------- DOCX

def _heading_level(style_name: str) -> int | None:
    m = re.match(r"(?:Heading|Заголовок)\s*(\d+)", style_name or "", re.I)
    return int(m.group(1)) if m else None


def _rows_to_lines(rows: list[list[str]]) -> list[str]:
    """Строки таблицы -> строки вида 'Колонка: значение; Колонка2: значение2.'"""
    rows = [[" ".join((c or "").split()) for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return []

    if max(len(r) for r in rows) == 1:           # врезки "Важно" и т.п.
        return [r[0] for r in rows if r[0]]

    header = rows[0]
    if len(rows) == 1:
        return [" | ".join(c for c in header if c)]

    lines = []
    for r in rows[1:]:
        if len(r) == len(header):
            line = "; ".join(f"{h}: {c}" for h, c in zip(header, r) if c)
        else:
            line = " | ".join(c for c in r if c)
        if line:
            lines.append(line + ".")
    return lines


def _table_lines(table: Table) -> list[str]:
    rows = []
    for row in table.rows:
        cells, prev = [], None
        for cell in row.cells:
            if cell._tc is prev:       # объединённые ячейки повторяются, выкидываем дубли
                continue
            prev = cell._tc
            cells.append(cell.text)
        rows.append(cells)
    return _rows_to_lines(rows)


def parse_docx(path: Path) -> list[dict]:
    doc = Document(str(path))
    chunks, buf, stack = [], [], {}
    section = path.stem
    seen: set = set()

    def flush():
        text = clean_text(" ".join(buf))
        buf.clear()
        for piece in split_text(text):
            if not _is_toc(piece):
                chunks.append(_text_chunk(path.name, section, None, piece))

    def handle_images(element):
        for rid in element.xpath(".//a:blip/@r:embed"):
            part = doc.part.related_parts.get(rid)
            if part is None:
                continue
            context = clean_text(" ".join(buf))[-300:]
            ch = make_image_chunk(part.blob, path.name, path.stem, section, context, seen)
            if ch:
                chunks.append(ch)

    for child in doc.element.body.iterchildren():
        tag = child.tag.split("}")[-1]

        if tag == "p":
            p = Paragraph(child, doc)
            try:
                style = p.style.name if p.style is not None else ""
            except Exception:
                style = ""
            if style.lower().startswith(("toc", "оглавление", "table of contents")):
                continue

            text = p.text.strip()
            level = _heading_level(style)
            if level and text:
                flush()
                stack = {k: v for k, v in stack.items() if k < level}
                stack[level] = text
                section = " > ".join(stack[k] for k in sorted(stack))
                continue

            if text:
                buf.append(text)
            handle_images(child)

        elif tag == "tbl":
            buf.extend(_table_lines(Table(child, doc)))
            handle_images(child)

    flush()
    return chunks


# ------------------------------------------------------------------ PDF

_TOC_TITLES = {"table of contents", "содержание", "оглавление", "1 содержание"}
_NUM_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+(?!\d+\.)\S")   # "5 Инструкция", "2.1.1 Фильтры", но не "1. Шаг"


def _norm_line(text: str) -> str:
    return re.sub(r"\d+", "#", text.strip().lower())


def _line_style(line: dict) -> tuple[float, float]:
    chars = [c for c in line["chars"] if c["text"].strip()]
    if not chars:
        return 0.0, 0.0
    size = sum(c["size"] for c in chars) / len(chars)
    bold = sum(1 for c in chars if re.search(r"bold|black|heavy", c.get("fontname", ""), re.I)) / len(chars)
    return size, bold


def _inside(obj: dict, boxes: list) -> bool:
    cx, cy = (obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2
    return any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes)


def _looks_like_table(rows: list[list]) -> bool:
    """Настоящая таблица: минимум 2 строки и 2 колонки, большинство ячеек заполнено, ячейки короткие.
    Рамки-врезки ("Важно") и сетки из пустых ячеек таблицами не считаем: их текст лучше брать как есть."""
    if len(rows) < 2 or max(len(r) for r in rows) < 2:
        return False
    cells = [(c or "").strip() for r in rows for c in r]
    filled = sum(1 for c in cells if c)
    return filled / len(cells) >= 0.5 and max(len(c) for c in cells) <= 200


def _extract_page(page, n: int) -> dict:
    """Одна страница -> элементы в порядке чтения: строки, таблицы, картинки."""
    items, images = [], []

    boxes, tables = [], []
    try:
        for t in page.find_tables():
            rows = t.extract()
            if rows and _looks_like_table(rows):
                boxes.append(t.bbox)
                tables.append((t.bbox[1], rows))
    except Exception:
        boxes, tables = [], []

    view = page.filter(lambda o: o.get("object_type") != "char" or not _inside(o, boxes)) if boxes else page
    for ln in view.extract_text_lines():
        text = re.sub(r"[\uf000-\uf8ff\u200b]", "", ln["text"]).strip()
        if not text or re.search(r"\.{5,}\s*\d*\s*$", text):   # пустое или строка оглавления "Раздел.......5"
            continue
        size, bold = _line_style(ln)
        items.append({"kind": "line", "top": ln["top"], "text": text, "size": size, "bold": bold})
    for top, rows in tables:
        items.append({"kind": "table", "top": top, "rows": rows})
    items.sort(key=lambda it: it["top"])

    for img in page.images:
        w, h = img["x1"] - img["x0"], img["bottom"] - img["top"]
        sw, sh = img.get("srcsize") or (w, h)
        if max(sw, sh) < MIN_IMAGE_MAX_SIDE or min(sw, sh) < MIN_IMAGE_MIN_SIDE or w < 5 or h < 5:
            continue
        try:
            bbox = (max(img["x0"], 0), max(img["top"], 0),
                    min(img["x1"], page.width), min(img["bottom"], page.height))
            dpi = int(min(300, max(150, sw / w * 72)))     # рендерим не хуже исходного разрешения
            pil = page.crop(bbox).to_image(resolution=dpi).original
            buf = BytesIO()
            pil.save(buf, "PNG")
        except Exception:
            continue
        images.append({"kind": "image", "top": img["top"], "png": buf.getvalue(), "native": (sw, sh)})

    return {"n": n, "items": items, "images": images, "height": page.height}


def _strip_running_headers(pages: list[dict]) -> None:
    """Колонтитулы и повторяющийся мусор: строки с верха/низа страницы, которые встречаются на многих страницах."""
    if len(pages) < 4:
        return
    counts: dict = {}
    for p in pages:
        lines = [it for it in p["items"] if it["kind"] == "line"]
        edge = lines[:3] + lines[-3:]
        for key in {_norm_line(it["text"]) for it in edge}:
            counts[key] = counts.get(key, 0) + 1
    limit = max(3, int(0.3 * len(pages)))
    noise = {k for k, v in counts.items() if v >= limit}
    for p in pages:
        lines = [it for it in p["items"] if it["kind"] == "line"]
        edge_ids = {id(it) for it in lines[:3] + lines[-3:]}
        p["items"] = [it for it in p["items"]
                      if not (it["kind"] == "line" and id(it) in edge_ids and _norm_line(it["text"]) in noise)]


def _find_headings(pages: list[dict]) -> dict[int, int]:
    """Помечает строки-заголовки (it['heading']=True) и возвращает {округлённый размер: уровень}."""
    weights: dict = {}
    for p in pages:
        for it in p["items"]:
            if it["kind"] == "line":
                k = round(it["size"])
                weights[k] = weights.get(k, 0) + len(it["text"])
    body = max(weights, key=weights.get) if weights else 10

    sizes = set()
    for p in pages:
        prev = None
        for it in p["items"]:
            it["heading"] = False
            if it["kind"] != "line":
                prev = None
                continue
            text, size, bold = it["text"], it["size"], it["bold"]
            short = len(text) <= 110 and not text.endswith((",", ";", ":"))
            numbered = bool(_NUM_HEADING_RE.match(text))
            is_toc_title = text.strip().lower() in _TOC_TITLES
            # заголовок: жирный и заметно крупнее обычного текста (или с номером раздела), либо очень крупный.
            # Жирные слова внутри абзацев и подписи врезок ("Важно") под это не попадают.
            has_words = sum(ch.isalpha() for ch in text) >= 3     # значок или номер без слов заголовком не считаем
            if short and has_words and not is_toc_title and (
                    (bold >= 0.6 and (size >= body * 1.3 or (numbered and size >= body * 1.05)))
                    or size >= body * 1.5):
                it["heading"] = True
                sizes.add(round(size))
                if prev is not None and abs(prev["size"] - size) < 0.5 and it["top"] - prev["top"] < size * 2.2:
                    prev["text"] += " " + text          # заголовок, перенесённый на вторую строку
                    it["heading"] = None               # помечаем на удаление
                else:
                    prev = it
            else:
                prev = None
        p["items"] = [it for it in p["items"] if it.get("heading") is not None]
    return {sz: lvl for lvl, sz in enumerate(sorted(sizes, reverse=True), start=1)}


def _context_before(items: list[dict], top: float) -> str:
    lines = [it["text"] for it in items if it["kind"] == "line" and it["top"] < top]
    return clean_text(" ".join(lines))[-300:]


def parse_pdf(path: Path) -> list[dict]:
    pages = []
    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        for n, page in enumerate(pdf.pages, start=1):
            pages.append(_extract_page(page, n))
            if total > 60 and n % 50 == 0:
                print(f"  {path.name}: прочитано {n}/{total} страниц")

    _strip_running_headers(pages)
    levels = _find_headings(pages)

    chunks, seen = [], set()
    stack: dict[int, str] = {}

    for p in pages:
        n = p["n"]
        buf: list[str] = []
        section = " > ".join(stack[k] for k in sorted(stack)) or f"стр. {n}"

        def flush():
            text = clean_text("\n".join(buf))
            buf.clear()
            for piece in split_text(text):
                if not _is_toc(piece):
                    chunks.append(_text_chunk(path.name, section, n, piece))

        events = p["items"] + p["images"]
        events.sort(key=lambda e: e["top"])
        for ev in events:
            if ev["kind"] == "line" and ev.get("heading"):
                flush()
                lvl = levels.get(round(ev["size"]), 1)
                stack = {k: v for k, v in stack.items() if k < lvl}
                stack[lvl] = ev["text"]
                section = " > ".join(stack[k] for k in sorted(stack))
            elif ev["kind"] == "line":
                buf.append(ev["text"])
            elif ev["kind"] == "table":
                buf.extend(_rows_to_lines(ev["rows"]))
            else:   # image
                ch = make_image_chunk(ev["png"], path.name, path.stem, section,
                                      _context_before(p["items"], ev["top"]), seen,
                                      page=n, skip_if_few_colors=True, native_size=ev["native"])
                if ch:
                    chunks.append(ch)
        flush()
    return chunks


# ----------------------------------------------------------------- общее

def parse_file(path: Path) -> list[dict]:
    if path.suffix.lower() == ".docx":
        return parse_docx(path)
    if path.suffix.lower() == ".pdf":
        return parse_pdf(path)
    return []


def list_document_files(folder: str | Path) -> list[Path]:
    """Все .docx и .pdf из папки. Если есть и docx, и pdf с одним именем, берём docx."""
    folder = Path(folder)
    by_stem: dict[str, Path] = {}
    for p in sorted(folder.iterdir()):
        if p.name.startswith("~$") or p.suffix.lower() not in (".docx", ".pdf"):
            continue
        if p.stem not in by_stem or p.suffix.lower() == ".docx":
            by_stem[p.stem] = p
    return list(by_stem.values())


def parse_folder(folder: str | Path) -> list[dict]:
    """Все документы из папки -> единый список чанков."""
    all_chunks = []
    for p in list_document_files(folder):
        chunks = parse_file(p)
        n_img = sum(1 for c in chunks if c["type"] == "image")
        print(f"{p.name}: {len(chunks) - n_img} текстовых чанков, {n_img} картинок")
        all_chunks.extend(chunks)
    return all_chunks


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "docs"
    chunks = parse_folder(folder)
    texts = [c for c in chunks if c["type"] == "text"]
    print(f"\nВсего: {len(texts)} текстовых чанков, {len(chunks) - len(texts)} картинок")
    if texts:
        lens = [len(c["text"]) for c in texts]
        print(f"Длина текстового чанка: min={min(lens)}, max={max(lens)}, avg={sum(lens) // len(lens)}")
        print("\nПример текста:", texts[0])
    imgs = [c for c in chunks if c["type"] == "image"]
    if imgs:
        print("\nПример картинки:", imgs[0])

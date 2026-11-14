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
"""
import base64
import hashlib
import json
import os
import re
import sys
from io import BytesIO
from pathlib import Path

import pdfplumber
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from PIL import Image

CHUNK_SIZE = 450
CHUNK_OVERLAP = 80
MIN_CHUNK_LEN = 40
MIN_IMAGE_SIDE = 100          # картинки меньше (иконки, значки) пропускаем

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
        if len(chunk) >= MIN_CHUNK_LEN:
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


def _describe_with_openai(img: Image.Image, section: str, context: str) -> str | None:
    try:
        from openai import OpenAI

        buf = BytesIO()
        img.save(buf, "PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        prompt = (
            f"Это изображение из документации корпоративной системы БАРС-Офис, раздел: {section}. "
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


def make_image_chunk(img_bytes: bytes, source: str, stem: str, section: str, context: str,
                     seen: set, page=None, skip_if_few_colors: bool = False) -> dict | None:
    key = hashlib.md5(img_bytes).hexdigest()
    if key in seen:
        return None
    seen.add(key)

    try:
        img = Image.open(BytesIO(img_bytes))
        img.load()
    except Exception:
        return None  # например emf/wmf, которые Pillow не открывает

    if min(img.size) < MIN_IMAGE_SIDE:
        return None
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
    if not desc and os.getenv("OPENAI_API_KEY"):
        desc = _describe_with_openai(img, section, context)
        if desc:
            cache[cache_key] = desc
            CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    described = bool(desc)                    # True только если описание реально получено от модели
    if not desc:
        # заглушка: ingest.py увидит described=False и повторит попытку, когда ключ заработает
        desc = f"Скриншот или схема из раздела «{section}». Рядом в документе: {context}"

    return {"type": "image", "source": source, "section": section, "page": page,
            "text": desc, "image_path": str(out_path), "described": described}


# ----------------------------------------------------------------- DOCX

def _heading_level(style_name: str) -> int | None:
    m = re.match(r"(?:Heading|Заголовок)\s*(\d+)", style_name or "", re.I)
    return int(m.group(1)) if m else None


def _table_lines(table: Table) -> list[str]:
    """Таблица -> строки вида 'Колонка: значение; Колонка2: значение2'."""
    rows = []
    for row in table.rows:
        cells, prev = [], None
        for cell in row.cells:
            if cell._tc is prev:       # объединённые ячейки повторяются, выкидываем дубли
                continue
            prev = cell._tc
            cells.append(" ".join(cell.text.split()))
        if any(cells):
            rows.append(cells)
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
            pairs = [f"{h}: {c}" for h, c in zip(header, r) if c]
            line = "; ".join(pairs)
        else:
            line = " | ".join(c for c in r if c)
        if line:
            lines.append(line + ".")
    return lines


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

def parse_pdf(path: Path) -> list[dict]:
    chunks, seen = [], set()
    with pdfplumber.open(path) as pdf:
        for n, page in enumerate(pdf.pages, start=1):
            text = clean_text(page.extract_text() or "")
            section = f"стр. {n}"

            for piece in split_text(text):
                if not _is_toc(piece):
                    chunks.append(_text_chunk(path.name, section, n, piece))

            for img in page.images:
                w, h = img["x1"] - img["x0"], img["bottom"] - img["top"]
                if min(w, h) < 60:
                    continue
                try:
                    bbox = (max(img["x0"], 0), max(img["top"], 0),
                            min(img["x1"], page.width), min(img["bottom"], page.height))
                    pil = page.crop(bbox).to_image(resolution=150).original
                    buf = BytesIO()
                    pil.save(buf, "PNG")
                except Exception:
                    continue
                ch = make_image_chunk(buf.getvalue(), path.name, path.stem, section, text[:300],
                                      seen, page=n, skip_if_few_colors=True)
                if ch:
                    chunks.append(ch)
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

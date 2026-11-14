"""Инкрементальная загрузка: документы из папки -> чанки -> эмбеддинги -> Qdrant.

Запуск:
    python ingest.py docs          # обработает только новые и изменённые файлы
    python ingest.py docs --full   # полная переиндексация (удалит коллекцию и загрузит всё заново)

Как это работает:
  - для каждого файла считается MD5, он сравнивается с docs_cache.json;
  - файл не изменился и его чанки лежат в Qdrant -> пропускаем (эмбеддинги не считаются);
  - файл новый или изменился -> удаляем его старые чанки (по полю source) и загружаем новые;
  - файл удалили из папки -> его чанки убираются из Qdrant;
  - если коллекцию в Qdrant удалили или она неполная, файлы обработаются заново сами.

Заново обработать файл также придётся, если:
  - поменялись CHUNK_SIZE / CHUNK_OVERLAP, модель эмбеддингов или vision-модель (отслеживается автоматически);
  - есть ключ OPENAI_API_KEY, а часть картинок файла осталась с заглушками вместо описаний
    (ключ появился позже или API не ответил): такие файлы повторяются при каждом запуске, пока все картинки не описаны;
  - ты правила код parser.py (его изменения скрипт не видит): тогда запусти с --full.

Ключ для описания картинок:
    export OPENAI_API_KEY=...        (Windows PowerShell: $env:OPENAI_API_KEY="...")
"""
import hashlib
import json
import os
import sys
from pathlib import Path

from parser import CHUNK_OVERLAP, CHUNK_SIZE, VISION_MODEL, list_document_files, parse_file
from qdrant_store import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    count_document,
    create_collection,
    delete_document,
    get_client,
    upload_chunks,
)

CACHE_FILE = Path(os.getenv("DOCS_CACHE", "docs_cache.json"))
# Если поменять любой из этих параметров, чанки и эмбеддинги изменятся, и файлы надо пересчитать.
FINGERPRINT = f"{EMBEDDING_MODEL}|chunk={CHUNK_SIZE}|overlap={CHUNK_OVERLAP}|vision={VISION_MODEL}"


def get_file_hash(path: Path) -> str:
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            md5.update(block)
    return md5.hexdigest()


def load_cache() -> dict:
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            print(f"{CACHE_FILE} повреждён, начинаю с чистого кэша")
    return {}


def save_cache(cache: dict) -> None:
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def why_reprocess(entry: dict | None, file_hash: str, in_qdrant: int, vision: bool) -> str | None:
    """Возвращает причину, по которой файл надо обработать, или None, если его можно пропустить."""
    if entry is None:
        return "новый файл"
    if entry.get("hash") != file_hash:
        return "файл изменён"
    if entry.get("fingerprint") != FINGERPRINT:
        return "изменились размер чанка или модель эмбеддингов"
    if in_qdrant != entry.get("chunks"):
        return f"в Qdrant {in_qdrant} чанков вместо {entry.get('chunks')} (коллекцию пересоздавали?)"
    missing = entry.get("images", 0) - entry.get("images_described", 0)
    if vision and missing > 0:
        return f"{missing} картинок без описания, пробую описать через OpenAI"
    return None


def main(folder: str = "docs", full: bool = False) -> None:
    if not Path(folder).is_dir():
        # опечатка в пути не должна приводить к очистке базы
        print(f"Папки '{folder}' не существует. Ничего не изменено.")
        return

    files = list_document_files(folder)
    if not files:
        print(f"В папке {folder} нет .docx или .pdf: из базы будут убраны чанки всех ранее загруженных файлов")

    client = get_client()
    cache = load_cache()
    vision = bool(os.getenv("OPENAI_API_KEY"))
    if not vision:
        print("OPENAI_API_KEY не задан: картинки получат заглушки вместо описаний")

    if full or not client.collection_exists(COLLECTION_NAME):
        if full:
            print("Режим --full: коллекция будет пересоздана")
        create_collection(client, recreate=True)
        cache = {}
        save_cache(cache)

    # Файлы, которых больше нет в папке: убираем их чанки
    current = {p.name for p in files}
    removed = [name for name in cache if name not in current]
    for name in removed:
        delete_document(client, name)
        cache.pop(name)
        save_cache(cache)
        print(f"Удалён из базы (файла нет в папке): {name}")

    processed, skipped = [], []
    for path in files:
        name = path.name
        file_hash = get_file_hash(path)
        reason = why_reprocess(cache.get(name), file_hash, count_document(client, name), vision)

        if reason is None:
            skipped.append(name)
            print(f"Пропускаю (без изменений): {name}")
            continue

        print(f"\nОбрабатываю: {name} ({reason})")
        cache.pop(name, None)       # пока файл не загружен целиком, он не считается обработанным
        save_cache(cache)
        delete_document(client, name)

        chunks = parse_file(path)
        if not chunks:
            print(f"  Чанков нет, файл пропущен. Нужен документ с текстовым слоем (не скан): {name}")
            continue

        images = [c for c in chunks if c["type"] == "image"]
        n_img = len(images)
        n_described = sum(1 for c in images if c.get("described"))
        print(f"  {len(chunks) - n_img} текстовых чанков, {n_img} картинок (с описанием от модели: {n_described})")
        if vision and n_described < n_img:
            print(f"  Внимание: {n_img - n_described} картинок остались с заглушками. "
                  f"Проверь ключ, баланс и доступ к OpenAI, при следующем запуске файл будет обработан снова")
        upload_chunks(client, chunks)

        cache[name] = {
            "hash": file_hash,
            "fingerprint": FINGERPRINT,
            "chunks": len(chunks),
            "images": n_img,
            "images_described": n_described,
        }
        save_cache(cache)           # сохраняем после каждого файла: сбой на следующем ничего не сломает
        processed.append(name)

    total = client.count(COLLECTION_NAME, exact=True).count
    print(f"\nГотово. Обработано: {len(processed)}, пропущено: {len(skipped)}, удалено: {len(removed)}")
    print(f"Всего чанков в коллекции '{COLLECTION_NAME}': {total}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else "docs", full="--full" in sys.argv)

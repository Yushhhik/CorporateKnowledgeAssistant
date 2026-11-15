import hashlib
import json
import os
import sys
from pathlib import Path

from parser import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    PARSER_VERSION,
    VISION_MODEL,
    list_document_files,
    parse_file,
)
from qdrant_store import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    count_document,
    create_collection,
    delete_document,
    get_client,
    stats,
    upload_chunks,
)

CACHE_FILE = Path(os.getenv("DOCS_CACHE", "docs_cache.json"))

SYSTEMS = {
    "БП Командировка.pdf": "bars",
    "Реестр мебели.pdf": "bars",
    "Функциональные_возможности_работы_с_БАРС_Офисом.pdf": "bars",
    "1 доп док.pdf": "garantexpress",
    "2 доп док.pdf": "docsvision",
}

FINGERPRINT = (
    f"{EMBEDDING_MODEL}|chunk={CHUNK_SIZE}|"
    f"overlap={CHUNK_OVERLAP}|parser={PARSER_VERSION}|vision={VISION_MODEL}|"
    f"systems={json.dumps(SYSTEMS, sort_keys=True, ensure_ascii=False)}"
)


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
    CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def why_reprocess(
    entry: dict | None,
    file_hash: str,
    in_qdrant: int,
    vision: bool
) -> str | None:
    if entry is None:
        return "новый файл"

    if entry.get("hash") != file_hash:
        return "файл изменён"

    if entry.get("fingerprint") != FINGERPRINT:
        return "изменились параметры обработки или система документа"

    if in_qdrant != entry.get("chunks"):
        return "количество чанков в Qdrant не совпадает с кэшем"

    missing = entry.get("images", 0) - entry.get("images_described", 0)

    if vision and missing > 0 and not entry.get("vision_tried"):
        return f"{missing} картинок без описания, пробую обработать повторно"

    return None


def main(folder: str = "docs", full: bool = False) -> None:
    if not Path(folder).is_dir():
        print(f"Папки '{folder}' не существует. Ничего не изменено.")
        return

    files = list_document_files(folder)

    if not files:
        print(
            f"В папке {folder} нет .docx или .pdf: "
            "из базы будут убраны чанки ранее загруженных файлов"
        )

    client = get_client()
    cache = load_cache()

    vision = bool(os.getenv("OPENAI_API_KEY"))

    if not vision:
        print("OPENAI_API_KEY не задан: картинки получат заглушки")

    if full or not client.collection_exists(COLLECTION_NAME):
        if full:
            print("Режим --full: коллекция будет пересоздана")

        create_collection(client, recreate=True)
        cache = {}
        save_cache(cache)

    current = {p.name for p in files}
    in_qdrant = set(stats(client)) if client.collection_exists(COLLECTION_NAME) else set()
    removed = sorted((set(cache) | in_qdrant) - current)

    for name in removed:
        delete_document(client, name)
        cache.pop(name, None)
        save_cache(cache)
        print(f"Удалён из базы: {name}")

    processed = []
    skipped = []

    for path in files:
        name = path.name
        file_hash = get_file_hash(path)

        reason = why_reprocess(
            cache.get(name),
            file_hash,
            count_document(client, name),
            vision
        )

        if reason is None:
            skipped.append(name)
            print(f"Пропускаю (без изменений): {name}")
            continue

        print(f"\nОбрабатываю: {name} ({reason})")

        cache.pop(name, None)
        save_cache(cache)

        delete_document(client, name)

        chunks = parse_file(path)

        if not chunks:
            print(f"Чанков нет, файл пропущен: {name}")
            continue

        system = SYSTEMS.get(name, "unknown")

        for chunk in chunks:
            chunk["system"] = system

        images = [c for c in chunks if c["type"] == "image"]
        n_img = len(images)
        n_described = sum(1 for c in images if c.get("described"))

        print(f"  Система: {system}")
        print(
            f"  {len(chunks) - n_img} текстовых чанков, "
            f"{n_img} картинок, описано: {n_described}"
        )

        if vision and n_described < n_img:
            print(
                f"  Внимание: {n_img - n_described} картинок "
                "остались без описания"
            )

        upload_chunks(client, chunks)

        cache[name] = {
            "hash": file_hash,
            "fingerprint": FINGERPRINT,
            "chunks": len(chunks),
            "images": n_img,
            "images_described": n_described,
            "vision_tried": vision,
        }

        save_cache(cache)
        processed.append(name)

    total = client.count(COLLECTION_NAME, exact=True).count

    print(
        f"\nГотово. Обработано: {len(processed)}, "
        f"пропущено: {len(skipped)}, удалено: {len(removed)}"
    )
    print(f"Всего чанков в коллекции '{COLLECTION_NAME}': {total}")

    print("\nСостав базы (документ: текст / картинки):")
    in_db = stats(client)

    for name in sorted(in_db):
        row = in_db[name]
        print(f"  {name}: {row.get('text', 0)} / {row.get('image', 0)}")

    for name in SYSTEMS:
        if name not in in_db:
            print(f"  ВНИМАНИЕ: {name} нет в базе (файла нет в папке {folder}?)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else "docs", full="--full" in sys.argv)
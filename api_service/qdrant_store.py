"""Работа с Qdrant: эмбеддинги, создание коллекции, загрузка чанков.

Эмбеддинги считаем локально (ключ OpenAI для этого не нужен).
Модель multilingual-e5-small нормально понимает русский, размер вектора 384.
"""
import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, MatchValue, PayloadSchemaType, PointStruct, VectorParams,
)
from sentence_transformers import SentenceTransformer

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "docs")

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
VECTOR_SIZE = 384

_model = None


def get_model() -> SentenceTransformer:
    """Модель грузим один раз и переиспользуем."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Эмбеддинги для кусков документации. У e5 для них нужен префикс 'passage: '."""
    prefixed = [f"passage: {t}" for t in texts]
    vectors = get_model().encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Эмбеддинг вопроса пользователя. Префикс 'query: '."""
    vector = get_model().encode(f"query: {text}", normalize_embeddings=True)
    return vector.tolist()


def get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def _create_source_index(client: QdrantClient) -> None:
    """Индекс по полю source: быстрое удаление и подсчёт чанков конкретного файла."""
    try:
        client.create_payload_index(COLLECTION_NAME, "source", PayloadSchemaType.KEYWORD)
    except Exception:
        pass  # индекс уже есть или клиент его не требует


def create_collection(client: QdrantClient, recreate: bool = True) -> None:
    """Создаём коллекцию. recreate=True удалит старую вместе со всеми данными (полная переиндексация)."""
    if client.collection_exists(COLLECTION_NAME):
        if not recreate:
            return
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    _create_source_index(client)
    print(f"Коллекция '{COLLECTION_NAME}' создана")


def _source_filter(source: str) -> Filter:
    return Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])


def delete_document(client: QdrantClient, source: str) -> None:
    """Удаляет все чанки одного файла (по полю source), остальные не трогает."""
    client.delete(collection_name=COLLECTION_NAME, points_selector=_source_filter(source))


def count_document(client: QdrantClient, source: str) -> int:
    """Сколько чанков файла сейчас лежит в Qdrant."""
    return client.count(COLLECTION_NAME, count_filter=_source_filter(source), exact=True).count


def chunk_point_id(source: str, index: int) -> str:
    """Стабильный уникальный id точки: один и тот же файл и номер чанка всегда дают один и тот же id.
    Нужен, чтобы чанки разных файлов не перезаписывали друг друга при инкрементальной загрузке."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}#{index}"))


def _embed_text(chunk: dict) -> str:
    """Для поиска к тексту добавляем название раздела: так заголовок тоже помогает находить кусок."""
    return f"{chunk['section']}. {chunk['text']}"


def upload_chunks(client: QdrantClient, chunks: list[dict], batch_size: int = 32) -> int:
    """Загружаем чанки пачками. В payload кладём всё, что нужно для ответа и ссылки на источник."""
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectors = embed_passages([_embed_text(c) for c in batch])

        points = [
            PointStruct(
                id=chunk_point_id(batch[j]["source"], i + j),
                vector=vectors[j],
                payload={
                    "type": batch[j]["type"],
                    "source": batch[j]["source"],
                    "section": batch[j]["section"],
                    "page": batch[j]["page"],
                    "text": batch[j]["text"],
                    "image_path": batch[j]["image_path"],
                },
            )
            for j in range(len(batch))
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        total += len(points)
        print(f"Загружено {total}/{len(chunks)}")
    return total

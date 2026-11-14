"""Работа с Qdrant: эмбеддинги, создание коллекции, загрузка чанков."""

import os
import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "docs")

EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
VECTOR_SIZE = 384

_model = None


def get_model() -> SentenceTransformer:
    global _model

    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)

    return _model


def embed_passages(texts: list[str]) -> list[list[float]]:
    prefixed = [f"passage: {t}" for t in texts]

    vectors = get_model().encode(
        prefixed,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return vectors.tolist()


@lru_cache(maxsize=256)
def _embed_query_cached(text: str) -> tuple:
    vector = get_model().encode(
        f"query: {text}",
        normalize_embeddings=True
    )

    return tuple(vector.tolist())


def embed_query(text: str) -> list[float]:
    return list(_embed_query_cached(text.strip()))


def warm_up() -> None:
    embed_query("прогрев")


def get_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def _create_source_index(client: QdrantClient) -> None:
    try:
        client.create_payload_index(
            COLLECTION_NAME,
            "source",
            PayloadSchemaType.KEYWORD
        )
    except Exception:
        pass


def _create_system_index(client: QdrantClient) -> None:
    try:
        client.create_payload_index(
            COLLECTION_NAME,
            "system",
            PayloadSchemaType.KEYWORD
        )
    except Exception:
        pass


def create_collection(client: QdrantClient, recreate: bool = True) -> None:
    if client.collection_exists(COLLECTION_NAME):
        if not recreate:
            return

        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        ),
    )

    _create_source_index(client)
    _create_system_index(client)

    print(f"Коллекция '{COLLECTION_NAME}' создана")


def _source_filter(source: str) -> Filter:
    return Filter(
        must=[
            FieldCondition(
                key="source",
                match=MatchValue(value=source)
            )
        ]
    )


def delete_document(client: QdrantClient, source: str) -> None:
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=_source_filter(source)
    )


def count_document(client: QdrantClient, source: str) -> int:
    return client.count(
        COLLECTION_NAME,
        count_filter=_source_filter(source),
        exact=True
    ).count


def chunk_point_id(source: str, index: int) -> str:
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"{source}#{index}"
        )
    )


def _embed_text(chunk: dict) -> str:
    section = chunk.get("section") or ""
    return f"{section}. {chunk['text']}".strip()


def upload_chunks(
    client: QdrantClient,
    chunks: list[dict],
    batch_size: int = 32
) -> int:

    total = 0

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        vectors = embed_passages([
            _embed_text(c) for c in batch
        ])

        points = [
            PointStruct(
                id=chunk_point_id(
                    batch[j]["source"],
                    i + j
                ),
                vector=vectors[j],
                payload={
                    "type": batch[j]["type"],
                    "source": batch[j]["source"],
                    "section": batch[j]["section"],
                    "page": batch[j]["page"],
                    "text": batch[j]["text"],
                    "image_path": batch[j]["image_path"],
                    "system": batch[j].get("system", "unknown"),
                    "chunk_index": i + j,
                },
            )
            for j in range(len(batch))
        ]

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points
        )

        total += len(points)
        print(f"Загружено {total}/{len(chunks)}")

    return total


def get_neighbors(client: QdrantClient, source: str, index: int) -> list[dict]:
    ids = [chunk_point_id(source, k) for k in (index - 1, index + 1) if k >= 0]

    points = client.retrieve(
        COLLECTION_NAME,
        ids=ids,
        with_payload=True,
        with_vectors=False
    )

    return sorted(
        (p.payload for p in points),
        key=lambda p: p.get("chunk_index", 0)
    )


def get_page_images(client: QdrantClient, source: str, page: int) -> list[dict]:
    flt = Filter(must=[
        FieldCondition(key="source", match=MatchValue(value=source)),
        FieldCondition(key="type", match=MatchValue(value="image")),
        FieldCondition(key="page", match=MatchValue(value=page)),
    ])

    result, offset = [], None

    while True:
        points, offset = client.scroll(
            COLLECTION_NAME,
            scroll_filter=flt,
            limit=50,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )
        result.extend(p.payload for p in points)

        if offset is None:
            return result


def stats(client: QdrantClient) -> dict:
    result: dict = {}
    offset = None

    while True:
        points, offset = client.scroll(
            COLLECTION_NAME,
            limit=256,
            offset=offset,
            with_payload=["source", "type"],
            with_vectors=False
        )

        for p in points:
            row = result.setdefault(p.payload["source"], {"text": 0, "image": 0})
            row[p.payload.get("type", "text")] = row.get(p.payload.get("type", "text"), 0) + 1

        if offset is None:
            return result
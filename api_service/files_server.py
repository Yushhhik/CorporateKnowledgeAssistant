"""Отдача документов по подписанным ссылкам.

Запускается и отдельно (`uvicorn files_server:app --port 8001`, наружу публикуется только он),
и как часть основного API. Отдельный сервис нужен, чтобы в интернет не выставлять ни /webhook,
ни /documents, ни /users: по публичной ссылке доступны только файлы с верной подписью.
"""

import os
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse

from links import verify

DOCS_DIR = Path(os.getenv("DOCS_DIR", "/app/docs"))
MEDIA = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

router = APIRouter()


@router.get("/files/{name}")
async def get_file(name: str, sig: str = ""):
    name = Path(name).name

    if not verify(name, sig):
        raise HTTPException(status_code=403, detail="Недействительная ссылка")

    path = DOCS_DIR / name

    if path.suffix.lower() not in MEDIA or not path.is_file():
        raise HTTPException(status_code=404, detail="Документ не найден")

    return FileResponse(path, media_type=MEDIA[path.suffix.lower()], filename=name, content_disposition_type="inline")


app = FastAPI(title="files", docs_url=None, redoc_url=None, openapi_url=None)
app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok"}

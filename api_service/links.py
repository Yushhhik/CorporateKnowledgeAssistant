"""Публичные ссылки на документы-источники и группировка страниц.

Ссылка подписана (HMAC): открыть можно только файл из ответа бота, перебором имён в папке
docs чужие документы не скачать. Подпись считается от имени файла и SECRET из окружения.

Базовый адрес берётся из PUBLIC_FILES_URL (или PUBLIC_API_URL). Если он не задан, ссылки ведут на
http://localhost:8001 (порт сервиса files): они кликабельны на компьютере, где запущен Docker.
Для открытия с других устройств укажите в PUBLIC_FILES_URL внешний адрес или IP компьютера в сети.
Адреса внутри Docker-сети (app, files, host.docker.internal) снаружи не открываются, вместо них
подставляется localhost.
"""

import hashlib
import hmac
import os
from pathlib import Path
from urllib.parse import quote, urlparse

DEFAULT_LOCAL_URL = os.getenv("FILES_LOCAL_URL", "http://localhost:8001").rstrip("/")
# Имена, которые существуют только внутри Docker-сети или не годятся как адрес для клика
INTERNAL_HOSTS = {"0.0.0.0", "::", "::1", "host.docker.internal", "app", "files"}


def _secret() -> bytes:
    raw = os.getenv("FILES_SECRET") or os.getenv("DATABASE_URL") or "dev-only-secret"
    return hashlib.sha256(raw.encode("utf-8")).digest()


def sign(filename: str) -> str:
    return hmac.new(_secret(), Path(filename).name.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


def verify(filename: str, signature: str) -> bool:
    return hmac.compare_digest(sign(filename), signature or "")


def public_base() -> str:
    """Базовый адрес ссылок: из окружения, а если он пуст или внутренний, то localhost."""
    url = (os.getenv("PUBLIC_FILES_URL") or os.getenv("PUBLIC_API_URL") or "").strip().rstrip("/")

    if not url:
        return DEFAULT_LOCAL_URL

    host = (urlparse(url).hostname or "").lower()

    if not host or host in INTERNAL_HOSTS:
        return DEFAULT_LOCAL_URL

    return url


def file_url(source: str, page: int | None = None) -> str | None:
    base = public_base()
    name = Path(source).name
    url = f"{base}/files/{quote(name)}?sig={sign(name)}"

    return f"{url}#page={page}" if page else url


def _title(source: str) -> str:
    return Path(source).stem.replace("_", " ").strip()


def _page_key(page):
    try:
        return (0, int(page))
    except (TypeError, ValueError):
        return (1, str(page))


def format_sources(hits: list[dict]) -> list[str]:
    """Источники ответа, сгруппированные по файлу.

    Несколько страниц одного документа дают одну строку:
        [Функциональные возможности, стр. 3, 6, 13](ссылка)
    """
    groups: dict[str, dict] = {}

    for h in hits:
        g = groups.setdefault(h["source"], {"pages": [], "sections": []})

        if h.get("page") is not None and h.get("page") != "":
            if h["page"] not in g["pages"]:
                g["pages"].append(h["page"])
        elif h.get("section") and h["section"] not in g["sections"]:
            g["sections"].append(h["section"])

    lines = []

    for source, g in groups.items():
        pages = sorted(g["pages"], key=_page_key)
        label = _title(source)

        if pages:
            label += ", стр. " + ", ".join(str(p) for p in pages)
        elif g["sections"]:
            label += ", " + ", ".join(f"раздел «{s}»" for s in g["sections"][:3])

        url = file_url(source, pages[0] if pages else None)
        lines.append(f"[{label}]({url})" if url else f"Источник: {label}")

    return lines

"""Тесты загрузки документов (/documents), ссылок на источники и группировки страниц.

Работают без Qdrant, Ollama и базы: тяжёлые модули подменяются заглушками из test_pipeline,
индексацию заменяет мок. Запуск:  python -m pytest test_documents.py -v
"""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("multipart")

import test_pipeline  # noqa: F401, E402  (подставляет заглушки для redis/sqlalchemy/qdrant, если их нет)
import files_server  # noqa: E402
import links  # noqa: E402
import main  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

PDF = b"%PDF-1.4\n% test\n"
client = TestClient(main.app)


@pytest.fixture
def docs(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DOCS_DIR", tmp_path)
    monkeypatch.setattr(files_server, "DOCS_DIR", tmp_path)
    ran = []
    monkeypatch.setattr(main, "run_ingest", lambda folder, full=False: ran.append((folder, full)))
    return tmp_path, ran


def upload(name, content=PDF, mime="application/pdf"):
    return client.post("/documents", files={"file": (name, content, mime)})


# ---------- загрузка ----------

def test_upload_pdf_saves_file_and_starts_ingest(docs):
    path, ran = docs
    r = upload("Новый документ.pdf")

    assert r.status_code == 200
    assert r.json()["status"] == "success"
    assert (path / "Новый документ.pdf").read_bytes() == PDF
    assert ran == [(str(path), False)]               # индексация запущена в фоне, без пересоздания базы


def test_upload_docx_is_accepted(docs):
    path, _ = docs
    r = upload("Инструкция.docx", b"PK\x03\x04", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")

    assert r.status_code == 200
    assert (path / "Инструкция.docx").exists()


def test_upload_rejects_other_extensions(docs):
    path, ran = docs
    r = upload("virus.exe", b"MZ", "application/octet-stream")

    assert r.status_code == 400
    assert not list(path.iterdir()) and not ran


def test_upload_rejects_empty_file(docs):
    r = upload("empty.pdf", b"")

    assert r.status_code == 400
    assert not list(docs[0].iterdir())


def test_upload_rejects_too_big_file(docs, monkeypatch):
    monkeypatch.setattr(main, "DOC_MAX_SIZE", 10)
    r = upload("big.pdf", b"x" * 100)

    assert r.status_code == 413
    assert not list(docs[0].iterdir())               # недописанный файл не остаётся


def test_upload_path_traversal_is_neutralized(docs):
    path, _ = docs
    r = upload("../../etc/evil.pdf")

    assert r.status_code == 200
    assert r.json()["filename"] == "evil.pdf"
    assert (path / "evil.pdf").exists()
    assert not (path.parent / "evil.pdf").exists()


def test_documents_list(docs):
    path, _ = docs
    (path / "b.pdf").write_bytes(PDF)
    (path / "a.docx").write_bytes(b"x")
    (path / "notes.txt").write_text("x")

    assert client.get("/documents").json() == {"documents": ["a.docx", "b.pdf"]}


# ---------- подписанные ссылки ----------

def test_file_download_with_valid_signature(docs):
    path, _ = docs
    (path / "Doc.pdf").write_bytes(PDF)
    r = client.get(f"/files/Doc.pdf?sig={links.sign('Doc.pdf')}")

    assert r.status_code == 200
    assert r.content == PDF
    assert r.headers["content-type"].startswith("application/pdf")


def test_file_download_rejects_bad_signature(docs):
    (docs[0] / "Doc.pdf").write_bytes(PDF)

    assert client.get("/files/Doc.pdf?sig=wrong").status_code == 403
    assert client.get("/files/Doc.pdf").status_code == 403


def test_signature_of_one_file_does_not_open_another(docs):
    (docs[0] / "Secret.pdf").write_bytes(PDF)

    assert client.get(f"/files/Secret.pdf?sig={links.sign('Doc.pdf')}").status_code == 403


def test_missing_file_is_404(docs):
    assert client.get(f"/files/Nope.pdf?sig={links.sign('Nope.pdf')}").status_code == 404


def test_only_pdf_and_docx_are_served(docs):
    (docs[0] / "secret.txt").write_text("пароли")

    assert client.get(f"/files/secret.txt?sig={links.sign('secret.txt')}").status_code == 404


# ---------- ссылки в ответе бота ----------

HITS = [
    {"source": "Функциональные_возможности.pdf", "page": 6, "section": "a"},
    {"source": "Функциональные_возможности.pdf", "page": 3, "section": "b"},
    {"source": "Функциональные_возможности.pdf", "page": 13, "section": "c"},
    {"source": "Функциональные_возможности.pdf", "page": 6, "section": "d"},     # дубль страницы
    {"source": "Реестр мебели.pdf", "page": 9, "section": "e"},
]


def test_sources_grouped_by_file_with_sorted_unique_pages(monkeypatch):
    monkeypatch.setenv("PUBLIC_FILES_URL", "https://docs.example.com")
    lines = links.format_sources(HITS)

    assert len(lines) == 2
    assert lines[0].startswith("[Функциональные возможности, стр. 3, 6, 13](https://docs.example.com/files/")
    assert lines[1].startswith("[Реестр мебели, стр. 9](https://docs.example.com/files/")


def test_pages_are_sorted_numerically_not_as_strings(monkeypatch):
    monkeypatch.setenv("PUBLIC_FILES_URL", "https://docs.example.com")
    hits = [{"source": "A.pdf", "page": p} for p in (13, 3, 6)]

    assert "стр. 3, 6, 13" in links.format_sources(hits)[0]


def test_link_has_valid_signature_and_page_anchor(monkeypatch):
    monkeypatch.setenv("PUBLIC_FILES_URL", "https://docs.example.com")
    url = links.format_sources([{"source": "Реестр мебели.pdf", "page": 9}])[0].split("](")[1].rstrip(")")

    assert "/files/%D0%A0%D0%B5%D0%B5%D1%81%D1%82%D1%80%20%D0%BC%D0%B5%D0%B1%D0%B5%D0%BB%D0%B8.pdf?sig=" in url
    assert url.endswith("#page=9")
    assert links.verify("Реестр мебели.pdf", url.split("sig=")[1].split("#")[0])


@pytest.mark.parametrize("url", ["", "http://app:8000", "http://files:8001", "http://host.docker.internal:8001", "http://0.0.0.0:8001"])
def test_internal_or_empty_address_falls_back_to_localhost_link(monkeypatch, url):
    monkeypatch.delenv("PUBLIC_API_URL", raising=False)
    monkeypatch.setenv("PUBLIC_FILES_URL", url)
    lines = links.format_sources(HITS)

    assert lines[0].startswith("[Функциональные возможности, стр. 3, 6, 13](http://localhost:8001/files/")
    assert all("](" in line for line in lines)


@pytest.mark.parametrize("url", ["http://127.0.0.1:8001", "http://localhost:8001", "http://192.168.1.5:8001", "http://10.0.0.7:8001"])
def test_local_and_lan_addresses_are_kept_as_clickable_links(monkeypatch, url):
    monkeypatch.delenv("PUBLIC_API_URL", raising=False)
    monkeypatch.setenv("PUBLIC_FILES_URL", url)

    assert f"]({url}/files/" in links.format_sources(HITS)[0]


def test_docx_sources_without_pages_use_sections(monkeypatch):
    monkeypatch.delenv("PUBLIC_FILES_URL", raising=False)
    monkeypatch.delenv("PUBLIC_API_URL", raising=False)
    hits = [{"source": "Инструкция.docx", "page": None, "section": "Вход"},
            {"source": "Инструкция.docx", "page": None, "section": "Пароль"}]

    lines = links.format_sources(hits)

    assert len(lines) == 1
    assert lines[0].startswith("[Инструкция, раздел «Вход», раздел «Пароль»](http://localhost:8001/files/")
    assert "#page=" not in lines[0]

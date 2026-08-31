import asyncio
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import inspect
import pytest
from fastapi import BackgroundTasks, HTTPException, UploadFile
from starlette.datastructures import Headers
from app.api.dependencies import require_teacher
from app.api.routes import teacher as routes


def test_all_teacher_library_routes_require_teacher_role():
    for endpoint in (routes.generate_lesson_plan, routes.list_library, routes.upload_document, routes.save_resource, routes.get_resource, routes.update_resource, routes.delete_document, routes.delete_resource, routes.history):
        assert any(getattr(parameter.default, "dependency", None) is require_teacher for parameter in inspect.signature(endpoint).parameters.values())


@pytest.mark.parametrize(("filename", "content_type"), [("bad.exe", "application/octet-stream"), ("bad.js", "text/javascript"), ("bad.py", "text/plain")])
def test_unsupported_uploads_are_rejected_before_storage(tmp_path, monkeypatch, filename, content_type):
    monkeypatch.setattr(routes, "UPLOAD_ROOT", tmp_path)
    upload = UploadFile(filename=filename, file=BytesIO(b"unsafe"), headers=Headers({"content-type": content_type}))
    with pytest.raises(HTTPException) as error:
        asyncio.run(routes.upload_document(BackgroundTasks(), upload, SimpleNamespace(id=7), SimpleNamespace()))
    assert error.value.status_code == 422 and list(tmp_path.rglob("*")) == []


def test_path_traversal_filename_is_sanitized_and_private(tmp_path, monkeypatch):
    monkeypatch.setattr(routes, "UPLOAD_ROOT", tmp_path)
    class Db:
        def add(self, _item): pass
        def flush(self): pass
        def commit(self): pass
        def refresh(self, item): item.id = 1; item.created_at = None
        def rollback(self): pass
    upload = UploadFile(filename="../../other.pdf", file=BytesIO(b"%PDF"), headers=Headers({"content-type": "application/pdf"}))
    result = asyncio.run(routes.upload_document(BackgroundTasks(), upload, SimpleNamespace(id=7), Db()))
    assert result["original_filename"] == "other.pdf"
    assert list((tmp_path / "7").glob("*.pdf"))
    assert not (tmp_path / "other.pdf").exists()


def test_safe_responses_never_include_owner_or_storage_key():
    document = SimpleNamespace(id=1, title="private", original_filename="private.pdf", mime_type="application/pdf", file_size=1, status="ready", processing_stage="completed", processing_error=None, created_at=None, owner_id=4, storage_key="4/secret.pdf")
    resource = SimpleNamespace(id=2, resource_type="exercises", title="La famille", cefr_level="A1", theme=None, created_at=None, updated_at=None, owner_id=4, content={})
    assert "owner_id" not in routes.document_data(document) and "storage_key" not in routes.document_data(document)
    assert "owner_id" not in routes.resource_data(resource) and "content" not in routes.resource_data(resource)

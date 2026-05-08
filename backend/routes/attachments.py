"""Attachment routes (PR 8) — drop a file onto chat input → ingest into
RAG (persistent) or attach as ephemeral context for the next message.

Three endpoints share the /api/chat/* prefix the renderer is already on:

  POST   /api/chat/{conversation_id}/attach    multipart upload
  GET    /api/chat/{conversation_id}/attachments
  DELETE /api/chat/attachments/{id}

The orchestrator reads ``attachments`` rows for the conversation on the
next send and prepends ephemeral extracts to the user message inside a
quarantine envelope before calling the worker. Persistent rows survive
the send because they're already living in the RAG store.

Supported file types are limited to what services/rag_index.py can already
read with ``Path.read_text`` — text-based formats only. Binary formats
like .pdf and .docx need parsers we don't ship (PyPDF2 / pypdf /
python-docx); the route rejects them so we don't drop a binary blob into
the index. Adding parsers would mean new pip deps, which is out of scope
for this PR.
"""

from __future__ import annotations

import logging
import mimetypes
import uuid
from datetime import datetime, timezone
from pathlib import Path

import db as _db
from core import paths
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from ._helpers import get_api

log = logging.getLogger("iMakeAiTeams.attachments")

router = APIRouter()


# Mirror RAGIndex.DEFAULT_EXTENSIONS — every entry can be read straight off
# disk via Path.read_text(). Keep this in sync with rag_index.py if the
# parser set ever changes.
_TEXT_EXTENSIONS: frozenset[str] = frozenset({
    ".txt", ".py", ".json", ".md", ".csv", ".yaml", ".yml",
    ".html", ".css", ".js", ".ts", ".jsx", ".tsx", ".toml",
    ".ini", ".cfg", ".xml", ".sql", ".sh", ".bat", ".ps1",
    ".r", ".rs", ".go", ".java", ".c", ".cpp", ".h", ".rb",
    ".log", ".rst", ".tsv",
})

# Phase 11: image input. Anthropic's vision API and Ollama's vision
# models both accept these four formats. The mapping resolves the file
# extension to a canonical media_type so downstream consumers don't have
# to inspect bytes.
_IMAGE_EXTENSION_MIME: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}
_IMAGE_EXTENSIONS: frozenset[str] = frozenset(_IMAGE_EXTENSION_MIME.keys())
_IMAGE_MIME_TYPES: frozenset[str] = frozenset(_IMAGE_EXTENSION_MIME.values())

# Files we explicitly reject with a parser-aware error message — the user
# is most likely dropping these expecting them to work.
_KNOWN_UNSUPPORTED: frozenset[str] = frozenset({
    ".pdf", ".docx", ".doc", ".rtf", ".odt",
    ".xlsx", ".xls", ".pptx", ".ppt",
    ".zip", ".tar", ".gz", ".7z",
    ".bmp",
    ".mp3", ".mp4", ".wav", ".mov",
})

# Hard cap on upload size — bigger files are almost always binary blobs
# masquerading as text after read_text(errors="replace") strips them.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB

# Image-specific cap. Anthropic accepts images up to ~5MB per block, but
# many Ollama vision models choke on anything past 20MB. We hard-stop
# uploads above this so the user gets a clear error, not a silent failure
# downstream.
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB


def _ext(name: str) -> str:
    return Path(name).suffix.lower()


def _is_image_ext(ext: str) -> bool:
    return ext in _IMAGE_EXTENSIONS


def _is_supported(filename: str) -> tuple[bool, str]:
    """Return (ok, reason). reason is empty when ok."""
    ext = _ext(filename)
    if not ext:
        return False, "File has no extension. Drop a text-based file (.txt, .md, .json, …)."
    if ext in _IMAGE_EXTENSIONS:
        return True, ""
    if ext in _KNOWN_UNSUPPORTED:
        return False, (
            f"{ext} files are not supported in this build "
            "(no parser is installed). Convert to plain text or markdown first."
        )
    if ext not in _TEXT_EXTENSIONS:
        return False, (
            f"{ext} is not a supported file type. "
            f"Supported: {', '.join(sorted(_TEXT_EXTENSIONS))}."
        )
    return True, ""


def _extract_text(disk_path: Path) -> str:
    """Read a UTF-8 text file. Mirrors RAGIndex.add_file's read path so we
    extract the same content the persistent path would have indexed.
    """
    try:
        return disk_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not read attachment: {exc}",
        ) from exc


@router.post("/chat/{conversation_id}/attach")
async def upload_attachment(
    conversation_id: str,
    request: Request,
    file: UploadFile = File(...),
    persist: str = Form("false"),
) -> dict:
    """Receive a multipart upload, store it under userData/attachments,
    extract its text (or, for images, capture filename metadata),
    optionally index into RAG, and record the row.

    Images (PNG/JPEG/GIF/WebP) are routed into the chat as vision blocks
    and are always ephemeral — RAG is text-only in this codebase, so
    persist=true is silently downgraded to false for image rows.
    """
    filename = (file.filename or "upload").strip() or "upload"
    ok, reason = _is_supported(filename)
    if not ok:
        raise HTTPException(status_code=400, detail=reason)

    ext = _ext(filename)
    is_image = _is_image_ext(ext)

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")
    size_cap = MAX_IMAGE_BYTES if is_image else MAX_UPLOAD_BYTES
    if len(raw) > size_cap:
        if is_image:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Image too large ({len(raw) // (1024 * 1024)} MB). "
                    f"Maximum {MAX_IMAGE_BYTES // (1024 * 1024)} MB."
                ),
            )
        raise HTTPException(
            status_code=400,
            detail=(
                f"File is too large ({len(raw) // 1024} KB). "
                f"Limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
            ),
        )

    persist_flag = str(persist).strip().lower() in {"true", "1", "yes", "on"}
    # Images are always ephemeral — RAG is text-only.
    if is_image:
        persist_flag = False

    attachment_id = str(uuid.uuid4())
    disk_path = paths.attachments_dir() / f"{attachment_id}{ext}"
    try:
        disk_path.write_bytes(raw)
    except OSError as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not save attachment: {exc}",
        ) from exc

    if is_image:
        # Pillow isn't a dep of this codebase; just capture the filename so
        # the chip strip and orchestrator can render something useful.
        extracted = f"[image: {filename}]"
    else:
        extracted = _extract_text(disk_path)

    if is_image:
        # Force a canonical media_type derived from the extension instead
        # of trusting the upload's Content-Type header (browsers send all
        # sorts of things for paste-from-clipboard images).
        mime_type = _IMAGE_EXTENSION_MIME.get(ext, "")
    else:
        mime_type = (
            file.content_type or mimetypes.guess_type(filename)[0] or ""
        )

    rag_doc_id: str | None = None
    if persist_flag:
        rag = getattr(get_api(request), "_rag", None)
        if rag is None:
            try:
                disk_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise HTTPException(
                status_code=503,
                detail="RAG index is unavailable; cannot persist attachment.",
            )
        try:
            rag.add_text(extracted, source=filename)
            rag_doc_id = attachment_id
        except Exception as exc:
            log.warning("RAG ingest failed for %s: %s", filename, exc)
            try:
                disk_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise HTTPException(
                status_code=500, detail=f"RAG ingest failed: {exc}",
            ) from exc

    now = datetime.now(timezone.utc).isoformat()
    _db.execute(
        "INSERT INTO attachments (id, conversation_id, filename, mime_type, "
        "size_bytes, persist, rag_doc_id, content_extract, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            attachment_id, conversation_id, filename, mime_type, len(raw),
            1 if persist_flag else 0, rag_doc_id, extracted, now,
        ),
    )
    _db.commit()

    return {
        "id": attachment_id,
        "filename": filename,
        "size_bytes": len(raw),
        "persist": persist_flag,
        "extract_chars": len(extracted),
    }


@router.get("/chat/attachments/{attachment_id}/blob")
async def get_attachment_blob(attachment_id: str) -> FileResponse:
    """Serve the raw bytes of an attachment.

    PR 11 uses this to render image thumbnails in the chip strip after a
    conversation reload, where the original ``File`` object is gone but
    the row is still on disk. ``mime_type`` from the row drives the
    Content-Type header so browsers render the response inline.
    """
    row = _db.fetchone(
        "SELECT id, filename, mime_type FROM attachments WHERE id = ?",
        (attachment_id,),
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Attachment not found.")
    name = row["filename"] or ""
    ext = ""
    dot = name.rfind(".")
    if dot >= 0:
        ext = name[dot:].lower()
    disk_path = paths.attachments_dir() / f"{attachment_id}{ext}"
    if not disk_path.exists():
        raise HTTPException(status_code=404, detail="Attachment file missing.")
    media_type = row["mime_type"] or "application/octet-stream"
    return FileResponse(
        path=disk_path, media_type=media_type, filename=name,
    )


@router.get("/chat/{conversation_id}/attachments")
async def list_attachments(conversation_id: str, request: Request) -> list[dict]:
    """Return the attachments still in flight for a conversation. The
    frontend calls this on conversation switch so the chip strip rehydrates.
    """
    rows = _db.fetchall(
        "SELECT id, conversation_id, filename, mime_type, size_bytes, "
        "persist, rag_doc_id, created_at "
        "FROM attachments WHERE conversation_id = ? "
        "ORDER BY created_at ASC",
        (conversation_id,),
    )
    return [
        {
            "id": r["id"],
            "conversation_id": r["conversation_id"],
            "filename": r["filename"],
            "mime_type": r["mime_type"] or "",
            "size_bytes": r["size_bytes"],
            "persist": bool(r["persist"]),
            "rag_doc_id": r["rag_doc_id"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


@router.delete("/chat/attachments/{attachment_id}")
async def delete_attachment(attachment_id: str, request: Request) -> dict:
    """Drop the row, the on-disk file, and (when persisted) the RAG document."""
    row = _db.fetchone(
        "SELECT id, filename, persist, rag_doc_id FROM attachments WHERE id = ?",
        (attachment_id,),
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Attachment not found.")

    if row["persist"]:
        rag = getattr(get_api(request), "_rag", None)
        if rag is not None:
            try:
                import services.semantic_search as ss
                ss_rows = _db.fetchall(
                    "SELECT id FROM documents WHERE source = ?",
                    (row["filename"],),
                )
                for doc_row in ss_rows:
                    doc_id = doc_row["id"]
                    try:
                        map_row = _db.fetchone(
                            "SELECT vec_rowid FROM vec_documents_map WHERE doc_id = ?",
                            (doc_id,),
                        )
                        if map_row:
                            _db.execute(
                                "DELETE FROM vec_documents WHERE rowid = ?",
                                (map_row["vec_rowid"],),
                            )
                            _db.execute(
                                "DELETE FROM vec_documents_map WHERE doc_id = ?",
                                (doc_id,),
                            )
                    except Exception as exc:
                        log.debug(
                            "vec cleanup failed for %s: %s", doc_id, exc,
                        )
                    _db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
                    _db.execute("DELETE FROM bm25_corpus WHERE doc_id = ?", (doc_id,))
                _db.commit()
                if hasattr(ss, "_bm25_load_from_db"):
                    try:
                        ss._bm25_load_from_db()
                    except Exception as exc:
                        log.debug("bm25 reload failed: %s", exc)
            except Exception as exc:
                log.warning(
                    "RAG cleanup failed for attachment %s: %s",
                    attachment_id, exc,
                )

    _db.execute("DELETE FROM attachments WHERE id = ?", (attachment_id,))
    _db.commit()

    ext = _ext(row["filename"])
    disk_path = paths.attachments_dir() / f"{attachment_id}{ext}"
    try:
        disk_path.unlink(missing_ok=True)
    except OSError as exc:
        log.debug("attachment file unlink failed: %s", exc)

    return {"ok": True}

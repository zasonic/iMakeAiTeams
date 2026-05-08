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

# Files we explicitly reject with a parser-aware error message — the user
# is most likely dropping these expecting them to work.
_KNOWN_UNSUPPORTED: frozenset[str] = frozenset({
    ".pdf", ".docx", ".doc", ".rtf", ".odt",
    ".xlsx", ".xls", ".pptx", ".ppt",
    ".zip", ".tar", ".gz", ".7z",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
    ".mp3", ".mp4", ".wav", ".mov",
})

# Hard cap on upload size — bigger files are almost always binary blobs
# masquerading as text after read_text(errors="replace") strips them.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB


def _ext(name: str) -> str:
    return Path(name).suffix.lower()


def _is_supported(filename: str) -> tuple[bool, str]:
    """Return (ok, reason). reason is empty when ok."""
    ext = _ext(filename)
    if not ext:
        return False, "File has no extension. Drop a text-based file (.txt, .md, .json, …)."
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
    extract its text, optionally index into RAG, and record the row.
    """
    filename = (file.filename or "upload").strip() or "upload"
    ok, reason = _is_supported(filename)
    if not ok:
        raise HTTPException(status_code=400, detail=reason)

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"File is too large ({len(raw) // 1024} KB). "
                f"Limit is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
            ),
        )

    persist_flag = str(persist).strip().lower() in {"true", "1", "yes", "on"}

    attachment_id = str(uuid.uuid4())
    ext = _ext(filename)
    disk_path = paths.attachments_dir() / f"{attachment_id}{ext}"
    try:
        disk_path.write_bytes(raw)
    except OSError as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not save attachment: {exc}",
        ) from exc

    extracted = _extract_text(disk_path)

    mime_type = file.content_type or mimetypes.guess_type(filename)[0] or ""

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

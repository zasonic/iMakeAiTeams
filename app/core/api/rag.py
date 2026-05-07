"""
core/api/rag.py — RAG index bridge methods.
"""

from __future__ import annotations

from pathlib import Path

from core import paths
from core.service_guard import requires as _requires
from core.worker import run_in_thread

from services import input_sanitizer, semantic_search

from ._base import BaseAPI


class RagAPI(BaseAPI):

    @_requires("embedder", default=None)
    def build_rag_index(self, folder_path: str) -> None:
        """Build/rebuild the RAG index from a folder."""
        def _work():
            try:
                self._emit("rag_progress", {"status": "Scanning files…", "pct": 5})

                def _on_progress(status, pct):
                    self._emit("rag_progress", {"status": status, "pct": pct})

                self._rag.build_from_folder(Path(folder_path), on_progress=_on_progress)
                cache_path = paths.rag_cache_dir() / "index.npz"
                self._rag.save(cache_path)
                count = self._rag.chunk_count()
                self._emit("rag_done", {"chunks": count, "folder": folder_path})
                self._os_notify("RAG Index Built",
                                f"Indexed {count} chunks from {folder_path}")
            except Exception as e:
                self._log.error(f"RAG build error: {e}")
                err_msg = str(e).lower()
                if "permission" in err_msg or "access" in err_msg:
                    friendly = "Can't read that folder — check that the app has permission to access it."
                elif "not found" in err_msg or "no such" in err_msg:
                    friendly = "Folder not found — make sure the path still exists."
                elif "memory" in err_msg or "oom" in err_msg:
                    friendly = "Not enough memory to index that folder. Try a smaller one, or index individual files."
                else:
                    friendly = "Indexing failed. Check the folder path and try again."
                self._emit("rag_error", {"error": friendly})
        run_in_thread(_work)

    @_requires("embedder", default={"error": "RAG unavailable"})
    def rag_add_file(self, file_path: str) -> dict:
        """Add a single file to the existing RAG index."""
        try:
            _content = Path(file_path).read_text(errors="replace")[:50000]
            _scan = input_sanitizer.scan_document(_content, filename=file_path)
            if _scan.get("blocked"):
                return {"error": f"Document blocked by security scan — possible injection content detected.",
                        "scan_id": _scan.get("scan_id")}
        except Exception as _fe:
            self._log.debug(f"Document scan skipped: {_fe}")
        try:
            p = Path(file_path)
            n = self._rag.add_file(p)
            if n:
                cache_path = paths.rag_cache_dir() / "index.npz"
                self._rag.save(cache_path)
            return {"chunks_added": n, "total_chunks": self._rag.chunk_count()}
        except Exception as e:
            return {"error": str(e)}

    @_requires("embedder", default={"error": "RAG unavailable"})
    def rag_add_text(self, text: str, source: str = "manual") -> dict:
        """Add raw text to the RAG index."""
        try:
            n = self._rag.add_text(text, source=source)
            if n:
                cache_path = paths.rag_cache_dir() / "index.npz"
                self._rag.save(cache_path)
            return {"chunks_added": n, "total_chunks": self._rag.chunk_count()}
        except Exception as e:
            return {"error": str(e)}

    @_requires("rag_index", default={"error": "RAG unavailable"})
    def rag_clear(self) -> dict:
        """Clear the entire RAG index."""
        self._rag.clear()
        cache_path = paths.rag_cache_dir() / "index.npz"
        if cache_path.exists():
            cache_path.unlink()
        chunks_path = paths.rag_cache_dir() / "index_chunks.json"
        if chunks_path.exists():
            chunks_path.unlink()
        return {"ok": True}

    def rag_status(self) -> dict:
        status = self._status.get("rag_load", {}) if hasattr(self, "_status") else {}
        return {
            "chunk_count": self._rag.chunk_count() if self._rag is not None else 0,
            "index_exists": (paths.rag_cache_dir() / "index.npz").exists(),
            "available": bool(status.get("ok", self._rag is not None)),
            "error": status.get("error"),
        }

    @_requires("embedder", default=[])
    def rag_search(self, query: str, top_k: int = 5) -> list:
        results = self._rag.search(query, top_k=top_k)
        return [r[0] if isinstance(r, (list, tuple)) else r for r in results]

    def rag_search_hybrid(self, query: str, top_k: int = 5,
                          method: str = "hybrid", doc_type: str = "") -> list:
        """Hybrid BM25 + vector + RRF document search."""
        return semantic_search.search_documents_hybrid(
            query_text=query, top_k=top_k, doc_type=doc_type or None, method=method
        )

    def bm25_corpus_size(self) -> dict:
        """Return BM25 corpus size and availability."""
        return {
            "bm25_available": getattr(semantic_search, "_bm25_available", False),
            "corpus_size":    len(getattr(semantic_search, "_bm25_doc_ids", [])),
            "chroma_docs":    semantic_search.document_count(),
        }

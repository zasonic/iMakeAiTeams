"""
services/rag_index.py — RAG index (thin wrapper over semantic_search / ChromaDB).

Stage 2 consolidation:
  ChromaDB (semantic_search.py) is now the single document store.
  This module is kept as a compatibility shim so that all call-sites
  (api.py, memory.py, health_monitor.py) continue to work unchanged.

  On first startup, if a legacy .npz index exists alongside a
  _chunks.json file, the chunks are re-ingested into ChromaDB and the
  old files are deleted to avoid double-indexing.

  search() now returns (text, score) tuples instead of plain strings
  so that memory.py's similarity gating can filter low-relevance chunks.
  Callers that only need the text can do:  [t for t, _ in results]

Dependencies:
  sentence-transformers >= 2.7.0
  chromadb >= 0.4.0
"""

import json
import logging
import os
from pathlib import Path

log = logging.getLogger("MyAIEnv.rag_index")

# Similarity threshold below which chunks are still returned but scored
# (the MemoryManager decides whether to include them).
_SCORE_THRESHOLD = 0.0  # RAGIndex itself does not filter — MemoryManager does


class RAGIndex:
    """
    Public API (unchanged from the original):
      build_from_folder(folder)
      add_file(file_path)
      add_text(text, source)
      search(query, top_k) -> list[(text, score)] or list[str] (backward-compat)
      save(path)   — no-op (ChromaDB handles persistence)
      load(path)   — migrates legacy .npz on first call
      clear()
      chunk_count()

    All storage is delegated to semantic_search.py / ChromaDB.
    The SentenceTransformer model is shared via the module-level
    _shared_st_model set in api.py.
    """

    DEFAULT_EXTENSIONS = [
        ".txt", ".py", ".json", ".md", ".csv", ".yaml", ".yml",
        ".html", ".css", ".js", ".ts", ".jsx", ".tsx", ".toml",
        ".ini", ".cfg", ".xml", ".sql", ".sh", ".bat", ".ps1",
        ".r", ".rs", ".go", ".java", ".c", ".cpp", ".h", ".rb",
    ]

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", model=None):
        """
        model: optional pre-loaded SentenceTransformer instance (shared from api.py).
        model_name: ignored when model is provided; kept for API compatibility.
        """
        self._model = model  # may be None if semantic_search is not initialised yet
        self._semantic = None  # set lazily when semantic_search is available

    def _get_semantic(self):
        """Lazy import so that the module can be imported without chromadb installed."""
        if self._semantic is not None:
            return self._semantic
        try:
            import services.semantic_search as ss
            if ss.is_available():
                self._semantic = ss
        except Exception:
            pass
        return self._semantic

    # ── Index construction ────────────────────────────────────────────────────

    @staticmethod
    def _split_chunks(text: str, chunk_size: int = 800, overlap: int = 200) -> list[str]:
        """Split text into overlapping character-level chunks."""
        chunks: list[str] = []
        start = 0
        length = len(text)
        while start < length:
            end = min(start + chunk_size, length)
            chunks.append(text[start:end])
            if end == length:
                break
            start = end - overlap
        return chunks

    def build_from_folder(
        self,
        folder: Path,
        extensions: list[str] | None = None,
        on_progress=None,
    ) -> None:
        """
        Walk folder recursively, read matching files, chunk and ingest into ChromaDB.
        on_progress: optional callable(status_str, pct_int) for UI feedback.
        """
        if extensions is None:
            extensions = self.DEFAULT_EXTENSIONS

        ss = self._get_semantic()
        if ss is None:
            log.warning("RAGIndex.build_from_folder: semantic_search not available; skipping.")
            return

        # First pass: count files for progress reporting
        all_files = []
        for root, _dirs, files in os.walk(folder):
            for filename in sorted(files):
                filepath = Path(root) / filename
                if filepath.suffix.lower() in extensions:
                    all_files.append(filepath)

        total = len(all_files)
        for i, filepath in enumerate(all_files):
            if on_progress and i % 5 == 0:
                pct = int((i / max(total, 1)) * 100)
                on_progress(f"Indexing {filepath.name} ({i+1}/{total})", pct)
            try:
                content = filepath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not content.strip():
                continue
            try:
                rel = str(filepath.relative_to(folder))
            except ValueError:
                rel = str(filepath)
            self.add_text(content, source=rel)

    MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MB — reading beyond this risks OOM

    def add_file(self, file_path: Path) -> int:
        """Add a single file to the index. Returns number of chunks added."""
        file_path = Path(file_path)
        try:
            size = file_path.stat().st_size
            if size > self.MAX_FILE_BYTES:
                log.warning(
                    "add_file: %s is %.1f MB — skipping (limit %d MB). "
                    "Split the file into smaller parts to index it.",
                    file_path.name, size / 1_048_576, self.MAX_FILE_BYTES // 1_048_576,
                )
                return 0
        except OSError:
            return 0
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return 0
        if not content.strip():
            return 0
        return self.add_text(content, source=str(file_path))

    def add_text(self, text: str, source: str = "manual") -> int:
        """Add arbitrary text to the ChromaDB index. Returns number of chunks added."""
        if not text.strip():
            return 0
        ss = self._get_semantic()
        if ss is None:
            log.debug("RAGIndex.add_text: semantic_search not available; skipping.")
            return 0
        chunks = self._split_chunks(text)
        header = f"[{source}]\n"
        count = 0
        for chunk in chunks:
            try:
                ss.ingest_document(
                    content=header + chunk,
                    source=source,
                    doc_type="file",
                )
                count += 1
            except Exception as exc:
                log.warning("RAGIndex.add_text: ingest failed for %s: %s", source, exc)
        return count

    def clear(self) -> None:
        """Delete all documents from the ChromaDB collection."""
        ss = self._get_semantic()
        if ss is None:
            log.debug("RAGIndex.clear(): semantic_search not available; nothing to clear.")
            return
        try:
            if ss._documents_col is not None:
                # ChromaDB doesn't have a bulk-delete-all, so we retrieve all IDs
                # and delete them in one call.
                all_ids = ss._documents_col.get()["ids"]
                if all_ids:
                    ss._documents_col.delete(ids=all_ids)
                    log.info("RAGIndex.clear(): deleted %d document chunks from ChromaDB.", len(all_ids))
            # Also clear the corresponding rows from the SQLite documents table
            import db as _db_mod
            _db_mod.execute("DELETE FROM documents")
            _db_mod.commit()
        except Exception as exc:
            log.error("RAGIndex.clear() failed: %s", exc)

    def chunk_count(self) -> int:
        """Return number of indexed documents."""
        ss = self._get_semantic()
        if ss is None:
            return 0
        try:
            return ss.document_count()
        except Exception:
            return 0

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list:
        """
        Return the top_k most semantically similar chunks.

        Return type:
          list[(text: str, score: float)]   — when semantic_search is available
          list[]                             — when unavailable

        memory.py handles both plain strings and (text, score) tuples; returning
        scored tuples enables the similarity threshold filter in MemoryManager.
        """
        ss = self._get_semantic()
        if ss is None:
            return []
        if not query.strip():
            return []
        try:
            results = ss.search_documents(query, top_k=top_k)
            return [(r["content"], r["score"]) for r in results]
        except Exception as exc:
            log.debug(f"RAGIndex.search failed: {exc}")
            return []

    # ── Persistence (legacy compatibility) ───────────────────────────────────

    def save(self, path: Path) -> None:
        """
        No-op: ChromaDB handles its own persistence.
        Kept for API compatibility with call-sites that call save().
        """
        log.debug("RAGIndex.save() called — ChromaDB handles persistence; no file written.")

    def load(self, path: Path) -> None:
        """
        Migrate a legacy .npz + _chunks.json index into ChromaDB on first call,
        then delete the old files so they are not re-ingested on subsequent starts.
        """
        path = Path(path)
        chunks_path = path.parent / (path.stem + "_chunks.json")

        if not path.exists() and not chunks_path.exists():
            return  # Nothing to migrate

        log.info(
            "RAGIndex: legacy .npz index found at %s — migrating to ChromaDB.", path
        )
        try:
            chunks: list[str] = []
            if chunks_path.exists():
                chunks = json.loads(chunks_path.read_text(encoding="utf-8"))

            ss = self._get_semantic()
            if ss is not None and chunks:
                for chunk in chunks:
                    try:
                        ss.ingest_document(
                            content=chunk,
                            source="legacy_npz_migration",
                            doc_type="file",
                        )
                    except Exception as exc:
                        log.warning("RAGIndex migration: chunk ingest failed: %s", exc)
                log.info(
                    "RAGIndex: migrated %d legacy chunks into ChromaDB.", len(chunks)
                )

            # Remove legacy files so we don't re-ingest on next startup
            for legacy_file in (path, chunks_path):
                try:
                    if legacy_file.exists():
                        legacy_file.unlink()
                        log.info("RAGIndex: deleted legacy file %s", legacy_file)
                except OSError as exc:
                    log.warning("RAGIndex: could not delete %s: %s", legacy_file, exc)

        except Exception as exc:
            log.error("RAGIndex.load (migration) failed: %s", exc)

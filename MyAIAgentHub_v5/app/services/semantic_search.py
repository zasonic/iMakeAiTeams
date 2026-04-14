"""
services/semantic_search.py — Local Embedding and Semantic Search.

Original functionality (UNCHANGED):
  - Background embedding indexer (ChromaDB + sentence-transformers)
  - search_documents() and search_memories() API
  - Dirty-tracking integration with SQLite

Priority 2 additions (BM25 Hybrid Search):
  - _bm25_rebuild_index()    — rebuild BM25Okapi from bm25_corpus SQLite table
  - _bm25_add_document()     — add/update one document in bm25_corpus
  - _bm25_search()           — BM25 keyword search returning ranked (doc_id, score) list
  - reciprocal_rank_fusion() — RRF(d) = Σ 1/(k + rank_r(d)), k=60
  - search_documents_hybrid() — BM25 + ChromaDB + RRF, or vector-only fallback
  - tokenize()               — shared tokeniser (identical at index and query time)
  - ingest_document() updated to also call _bm25_add_document()

Usage:
  # Hybrid search (recommended)
  results = search_documents_hybrid("refund policy", top_k=5, method="hybrid")

  # Vector-only (original behaviour)
  results = search_documents("refund policy", top_k=5)

Requirements addition: rank-bm25>=0.2.2
"""

import logging
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import db

log = logging.getLogger("semantic_search")

_chroma_client = None
_documents_col = None
_memory_col    = None
_embed_fn      = None
_init_lock     = threading.Lock()
_initialized   = False

# ── Priority 2: BM25 module-level state ──────────────────────────────────────

_bm25_lock       = threading.Lock()
_bm25_index      = None   # BM25Okapi instance or None
_bm25_doc_ids:   list[str]       = []
_bm25_corpus:    list[list[str]] = []   # parallel list of token arrays
_bm25_contents:  dict[str, str]  = {}   # doc_id → raw content
_bm25_available  = False

try:
    from rank_bm25 import BM25Okapi  # type: ignore
    _bm25_available = True
except ImportError:
    BM25Okapi = None  # type: ignore
    log.warning(
        "rank-bm25 not installed — hybrid search will fall back to vector-only. "
        "Run: pip install rank-bm25"
    )

# English stopwords (lightweight, no NLTK dependency)
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "it", "its", "this", "that", "not",
    "so", "if", "then", "than", "more", "also", "just", "what", "which",
    "who", "when", "where", "how", "i", "you", "he", "she", "we", "they",
})

RRF_K = 60  # standard dampening constant (Cormack et al. 2009)


# ── Initialisation (unchanged) ────────────────────────────────────────────────

def init_vector_store(app_root: Path, shared_model=None) -> bool:
    """
    Initialise ChromaDB persistent client and embedding function.
    Also loads BM25 corpus from SQLite into memory.
    """
    global _chroma_client, _documents_col, _memory_col, _embed_fn, _initialized

    with _init_lock:
        if _initialized:
            return True
        try:
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
        except ImportError:
            log.warning(
                "chromadb or sentence-transformers not installed. "
                "Semantic search unavailable."
            )
            return False

        vector_dir = app_root / "myai_vector_store"
        vector_dir.mkdir(exist_ok=True)

        try:
            if shared_model is not None:
                class _SharedEmbedFn:
                    name = "all-MiniLM-L6-v2"
                    def __call__(self, input: list[str]) -> list[list[float]]:
                        import numpy as np
                        vecs = shared_model.encode(
                            input,
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                        )
                        return vecs.tolist()
                _embed_fn = _SharedEmbedFn()
            else:
                _embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

            _chroma_client = chromadb.PersistentClient(path=str(vector_dir))
            _documents_col = _chroma_client.get_or_create_collection(
                name="documents_index",
                embedding_function=_embed_fn,
                metadata={"hnsw:space": "cosine"},
            )
            _memory_col = _chroma_client.get_or_create_collection(
                name="memory_index",
                embedding_function=_embed_fn,
                metadata={"hnsw:space": "cosine"},
            )
            _initialized = True
            log.info("Semantic search vector store initialised.")

            # ── Priority 2: Load BM25 corpus from SQLite ──────────────────────
            _bm25_load_from_db()

            return True
        except Exception as exc:
            log.error(f"Vector store init failed: {exc}")
            return False


def is_available() -> bool:
    return _initialized


def document_count() -> int:
    if not _initialized or _documents_col is None:
        return 0
    try:
        return _documents_col.count()
    except Exception:
        return 0


# ── Priority 2: BM25 helpers ─────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Consistent tokeniser: lowercase → word tokens → strip stopwords."""
    text   = text.lower()
    tokens = re.findall(r"\b[a-z0-9_][a-z0-9_']*\b", text)
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _bm25_load_from_db() -> None:
    """Load BM25 corpus from SQLite bm25_corpus table and rebuild the in-memory index."""
    import json as _json
    global _bm25_doc_ids, _bm25_corpus, _bm25_contents, _bm25_index

    if not _bm25_available:
        return

    try:
        rows = db.fetchall("SELECT doc_id, tokens, content FROM bm25_corpus ORDER BY rowid")
    except Exception as exc:
        log.debug("bm25_corpus table not yet created: %s", exc)
        return

    with _bm25_lock:
        _bm25_doc_ids  = [r["doc_id"] for r in rows]
        _bm25_corpus   = [_json.loads(r["tokens"]) for r in rows]
        _bm25_contents = {r["doc_id"]: r["content"] for r in rows}
        _bm25_rebuild_index_locked()

    log.info("BM25: loaded %d documents from corpus.", len(_bm25_doc_ids))


def _bm25_rebuild_index_locked() -> None:
    """Rebuild BM25Okapi. Must be called with _bm25_lock held."""
    global _bm25_index
    if not _bm25_available or not _bm25_corpus:
        _bm25_index = None
        return
    _bm25_index = BM25Okapi(_bm25_corpus)


def _bm25_add_document(doc_id: str, content: str, metadata: dict | None = None) -> None:
    """
    Add or replace a document in the BM25 corpus (SQLite + in-memory).
    Thread-safe. Rebuilds the in-memory index after mutation.
    """
    import json as _json
    if not _bm25_available:
        return

    tokens    = tokenize(content)
    meta_json = _json.dumps(metadata or {})
    now       = datetime.now(timezone.utc).isoformat()

    try:
        db.execute(
            """
            INSERT INTO bm25_corpus (doc_id, tokens, content, metadata, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                tokens=excluded.tokens, content=excluded.content,
                metadata=excluded.metadata, updated_at=excluded.updated_at
            """,
            (doc_id, _json.dumps(tokens), content, meta_json, now),
        )
        db.commit()
    except Exception as exc:
        log.warning("BM25 db write failed for %s: %s", doc_id, exc)
        return

    with _bm25_lock:
        if doc_id in _bm25_contents:
            idx = _bm25_doc_ids.index(doc_id)
            _bm25_corpus[idx] = tokens
        else:
            _bm25_doc_ids.append(doc_id)
            _bm25_corpus.append(tokens)
        _bm25_contents[doc_id] = content
        _bm25_rebuild_index_locked()


def _bm25_search(query: str, n: int = 50) -> list[tuple[str, float]]:
    """
    BM25 search. Returns (doc_id, score) tuples sorted descending.
    Returns empty list if BM25 unavailable or corpus is empty.
    """
    if not _bm25_available or _bm25_index is None or not _bm25_doc_ids:
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    with _bm25_lock:
        scores  = _bm25_index.get_scores(query_tokens)
        doc_ids = list(_bm25_doc_ids)

    paired = [
        (doc_ids[i], float(scores[i]))
        for i in range(len(doc_ids))
        if scores[i] > 0.0
    ]
    paired.sort(key=lambda x: x[1], reverse=True)
    return paired[:n]


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """
    RRF across any number of ranked doc ID lists.
    RRF(d) = Σ 1/(k + rank_r(d)), rank is 1-indexed.
    Returns [(doc_id, rrf_score)] sorted descending.
    """
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def search_documents_hybrid(
    query_text: str,
    top_k: int = 10,
    doc_type: str | None = None,
    method: str = "hybrid",
) -> list[dict]:
    """
    Hybrid document search: BM25 + ChromaDB vector search + Reciprocal Rank Fusion.

    Parameters
    ----------
    query_text : Search query
    top_k      : Results to return
    doc_type   : Optional ChromaDB filter
    method     : "hybrid" (default) | "vector" (ChromaDB only) | "bm25" (BM25 only)

    Returns
    -------
    List of result dicts with keys:
      doc_id, content, score, file_source, doc_type, result_source, bm25_rank, vector_rank, rrf_score
    """
    if not query_text.strip():
        return []

    # Degrade to vector-only if BM25 not available
    if method == "hybrid" and not _bm25_available:
        method = "vector"

    bm25_results:   list[tuple[str, float]] = []
    vector_results: list[dict]              = []

    # ── BM25 candidates ──────────────────────────────────────────────────────
    if method in ("hybrid", "bm25"):
        bm25_results = _bm25_search(query_text, n=top_k * 5)

    # ── Vector candidates ─────────────────────────────────────────────────────
    if method in ("hybrid", "vector"):
        vector_results = search_documents(query_text, top_k=top_k * 5, doc_type=doc_type)

    # ── Vector-only: return as-is ─────────────────────────────────────────────
    if method == "vector":
        for r in vector_results:
            r["result_source"] = "vector"
            r["bm25_rank"]     = None
            r["vector_rank"]   = r.get("vector_rank", None)
            r["rrf_score"]     = None
        return vector_results[:top_k]

    # ── BM25-only ─────────────────────────────────────────────────────────────
    if method == "bm25":
        out = []
        for rank, (doc_id, score) in enumerate(bm25_results[:top_k], 1):
            content = _bm25_contents.get(doc_id, "")
            out.append({
                "doc_id":       doc_id,
                "content":      content,
                "score":        score,
                "file_source":  "",
                "doc_type":     "text",
                "result_source": "bm25",
                "bm25_rank":    rank,
                "vector_rank":  None,
                "rrf_score":    1.0 / (RRF_K + rank),
            })
        return out

    # ── Hybrid: RRF fusion ────────────────────────────────────────────────────
    bm25_ids   = [doc_id for doc_id, _ in bm25_results]
    vector_ids = [r["doc_id"] for r in vector_results]

    bm25_score_map   = {doc_id: score for doc_id, score in bm25_results}
    bm25_rank_map    = {doc_id: i + 1 for i, doc_id in enumerate(bm25_ids)}
    vector_rank_map  = {r["doc_id"]: i + 1 for i, r in enumerate(vector_results)}
    vector_data_map  = {r["doc_id"]: r for r in vector_results}

    fused = reciprocal_rank_fusion([bm25_ids, vector_ids], k=RRF_K)

    out = []
    seen: set[str] = set()

    for doc_id, rrf_score in fused[:top_k]:
        if doc_id in seen:
            continue
        seen.add(doc_id)

        in_bm25   = doc_id in bm25_rank_map
        in_vector = doc_id in vector_rank_map

        if in_vector:
            vdata    = vector_data_map[doc_id]
            content  = vdata["content"]
            fsource  = vdata.get("file_source", "")
            dtype    = vdata.get("doc_type", "text")
            vscore   = vdata.get("score", 0.0)
        else:
            content  = _bm25_contents.get(doc_id, "")
            fsource  = ""
            dtype    = "text"
            vscore   = None

        source_label = "both" if (in_bm25 and in_vector) else ("bm25" if in_bm25 else "vector")

        out.append({
            "doc_id":       doc_id,
            "content":      content,
            "score":        rrf_score,
            "file_source":  fsource,
            "doc_type":     dtype,
            "result_source": source_label,
            "bm25_rank":    bm25_rank_map.get(doc_id),
            "vector_rank":  vector_rank_map.get(doc_id),
            "rrf_score":    rrf_score,
            "bm25_score":   bm25_score_map.get(doc_id),
            "vector_score": vscore,
        })

    return out


# ── Background embedding indexer (unchanged) ──────────────────────────────────

def _index_dirty_documents(batch_size: int = 50) -> int:
    if not _initialized or _documents_col is None:
        return 0
    rows = db.fetchall(
        "SELECT id, content, source, doc_type, updated_at "
        "FROM documents WHERE embedding_status = 'dirty' LIMIT ?",
        (batch_size,),
    )
    if not rows:
        return 0

    ids, texts, metas = [], [], []
    for r in rows:
        if not r["content"] or not r["content"].strip():
            continue
        ids.append(r["id"])
        texts.append(r["content"])
        metas.append({
            "source":   r["source"] or "",
            "doc_type": r["doc_type"] or "text",
            "updated_at": r["updated_at"] or "",
        })

    if ids:
        try:
            _documents_col.upsert(ids=ids, documents=texts, metadatas=metas)
            db.executemany(
                "UPDATE documents SET embedding_status = 'clean' WHERE id = ?",
                [(i,) for i in ids],
            )
            # ── Priority 2: also update BM25 corpus for newly indexed docs ───
            for doc_id, text, meta in zip(ids, texts, metas):
                _bm25_add_document(doc_id, text, meta)
        except Exception as exc:
            log.error(f"Document upsert failed: {exc}")
            return 0

    return len(ids)


def _index_dirty_memories(batch_size: int = 50) -> int:
    if not _initialized or _memory_col is None:
        return 0
    rows = db.fetchall(
        "SELECT id, content, session_id, tags, created_at FROM memory_entries "
        "WHERE embedding_status = 'dirty' LIMIT ?",
        (batch_size,),
    )
    if not rows:
        return 0

    ids, texts, metas = [], [], []
    for r in rows:
        if not r["content"] or not r["content"].strip():
            continue
        ids.append(r["id"])
        texts.append(r["content"])
        metas.append({
            "session_id": r["session_id"] or "",
            "tags": r["tags"] or "[]",
            "created_at": r["created_at"] or "",
        })

    if ids:
        try:
            _memory_col.upsert(ids=ids, documents=texts, metadatas=metas)
            db.executemany(
                "UPDATE memory_entries SET embedding_status = 'clean' WHERE id = ?",
                [(i,) for i in ids],
            )
        except Exception as exc:
            log.error(f"Memory upsert failed: {exc}")
            return 0

    return len(ids)


def run_indexer_cycle() -> int:
    d = _index_dirty_documents()
    m = _index_dirty_memories()
    return d + m


def start_background_indexer(interval_seconds: int = 60) -> threading.Thread:
    def _loop():
        log.info("Background embedding indexer started.")
        while True:
            try:
                n = run_indexer_cycle()
                if n:
                    log.debug(f"Indexed {n} records.")
            except Exception as exc:
                log.error(f"Indexer cycle error: {exc}")
            time.sleep(interval_seconds)

    t = threading.Thread(target=_loop, name="embedding_indexer", daemon=True)
    t.start()
    return t


# ── Document ingestion (updated to also sync BM25) ────────────────────────────

def ingest_document(content: str, source: str, doc_type: str = "text",
                    metadata: dict | None = None) -> None:
    """
    Write a document chunk to the documents table for embedding.
    Priority 2: also syncs BM25 corpus immediately (not waiting for dirty-indexer cycle).
    """
    import json as _json
    now = datetime.now(timezone.utc).isoformat()
    row = db.fetchone("SELECT id FROM documents WHERE content = ? AND source = ?", (content, source))
    if row:
        db.execute(
            "UPDATE documents SET content = ?, doc_type = ?, metadata = ?, "
            "updated_at = ?, embedding_status = 'dirty' WHERE id = ?",
            (content, doc_type, _json.dumps(metadata or {}), now, row["id"]),
        )
        doc_id = row["id"]
    else:
        doc_id = str(uuid.uuid4())
        db.execute(
            "INSERT INTO documents (id, content, source, doc_type, metadata, "
            "embedding_status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 'dirty', ?, ?)",
            (doc_id, content, source, doc_type, _json.dumps(metadata or {}), now, now),
        )
    db.commit()

    # ── Priority 2: sync BM25 immediately ────────────────────────────────────
    _bm25_add_document(doc_id, content, {"source": source, "doc_type": doc_type, **(metadata or {})})


def ingest_memory(content: str, session_id: str = "", tags: list[str] | None = None) -> str:
    import json as _json
    entry_id = str(uuid.uuid4())
    db.execute(
        "INSERT INTO memory_entries (id, session_id, content, tags, created_at, embedding_status) "
        "VALUES (?, ?, ?, ?, ?, 'dirty')",
        (entry_id, session_id, content,
         _json.dumps(tags or []),
         datetime.now(timezone.utc).isoformat()),
    )
    db.commit()
    return entry_id


# ── Search (original vector search — unchanged) ───────────────────────────────

def search_documents(
    query_text: str,
    top_k: int = 10,
    doc_type: str | None = None,
) -> list[dict]:
    """Semantic (vector-only) search over all indexed documents."""
    if not _initialized or not _documents_col:
        return []
    if not query_text.strip():
        return []

    where = {"doc_type": doc_type} if doc_type else None
    try:
        count = _documents_col.count()
        if count == 0:
            return []
        results = _documents_col.query(
            query_texts=[query_text],
            n_results=min(top_k, count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        log.error(f"Document search failed: {exc}")
        return []

    out = []
    ids   = results.get("ids",       [[]])[0]
    docs  = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for rec_id, doc, meta, dist in zip(ids, docs, metas, dists):
        score = round(1.0 - dist, 3)
        out.append({
            "doc_id":       rec_id,
            "content":      doc,
            "score":        score,
            "file_source":  meta.get("source", ""),
            "doc_type":     meta.get("doc_type", "text"),
            "result_source": "semantic",
        })
    return out


def search_memories(
    query_text: str,
    top_k: int = 5,
    tags: list[str] | None = None,
) -> list[dict]:
    """Semantic search over indexed memory entries."""
    if not _initialized or not _memory_col:
        return []
    if not query_text.strip():
        return []

    try:
        count = _memory_col.count()
        if count == 0:
            return []
        results = _memory_col.query(
            query_texts=[query_text],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:
        log.error(f"Memory search failed: {exc}")
        return []

    out  = []
    ids  = results.get("ids",       [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas",[[]])[0]
    dists = results.get("distances",[[]])[0]
    now   = datetime.now(timezone.utc).isoformat()

    for rec_id, doc, meta, dist in zip(ids, docs, metas, dists):
        score = round(1.0 - dist, 3)
        out.append({
            "entry_id":   rec_id,
            "content":    doc,
            "score":      score,
            "session_id": meta.get("session_id", ""),
            "source":     "semantic",
        })
        try:
            db.execute(
                "UPDATE memory_entries SET last_accessed = ? WHERE id = ?",
                (now, rec_id),
            )
        except Exception:
            pass

    if out:
        try:
            db.commit()
        except Exception:
            pass

    return out


def get_stale_memories(days: int = 30) -> list[dict]:
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        rows = db.fetchall(
            """
            SELECT id, content, category, source, created_at, last_accessed
            FROM memory_entries
            WHERE last_accessed IS NULL OR last_accessed < ?
            ORDER BY COALESCE(last_accessed, created_at) ASC
            LIMIT 200
            """,
            (cutoff,),
        )
        return [dict(r) for r in rows]
    except Exception as exc:
        log.error("get_stale_memories failed: %s", exc)
        return []


def delete_memory_entry(entry_id: str) -> bool:
    try:
        db.execute("DELETE FROM memory_entries WHERE id = ?", (entry_id,))
        db.commit()
    except Exception as exc:
        log.error("delete_memory_entry (SQL) failed for %s: %s", entry_id, exc)
        return False

    if _initialized and _memory_col is not None:
        try:
            _memory_col.delete(ids=[entry_id])
        except Exception as exc:
            log.debug("delete_memory_entry (chroma) for %s: %s", entry_id, exc)

    return True

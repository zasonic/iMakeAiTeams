"""
tests/test_semantic_search.py — Unit tests for BM25/RRF semantic search core.

Covers: tokenize(), reciprocal_rank_fusion(), _bm25_search() (with mocked
in-memory state), and hybrid search degradation to vector-only when BM25 is
unavailable.
"""

import pytest
from unittest.mock import patch, MagicMock


# ── tokenize() ────────────────────────────────────────────────────────────────

def test_tokenize_lowercases():
    from services.semantic_search import tokenize
    tokens = tokenize("Hello World")
    assert "hello" in tokens
    assert "world" in tokens


def test_tokenize_removes_stopwords():
    from services.semantic_search import tokenize
    tokens = tokenize("the quick brown fox")
    assert "the" not in tokens
    assert "quick" in tokens
    assert "fox" in tokens


def test_tokenize_min_length_filter():
    from services.semantic_search import tokenize
    tokens = tokenize("a an it")
    assert tokens == []  # all stopwords / single-char


def test_tokenize_empty_string():
    from services.semantic_search import tokenize
    assert tokenize("") == []


def test_tokenize_preserves_numbers():
    from services.semantic_search import tokenize
    tokens = tokenize("version 42 released")
    assert "42" in tokens
    assert "version" in tokens


# ── reciprocal_rank_fusion() ─────────────────────────────────────────────────

def test_rrf_single_list():
    from services.semantic_search import reciprocal_rank_fusion
    result = reciprocal_rank_fusion([["a", "b", "c"]])
    ids = [r[0] for r in result]
    assert ids[0] == "a"  # highest rank gets highest score
    assert ids[-1] == "c"


def test_rrf_deduplicates_across_lists():
    from services.semantic_search import reciprocal_rank_fusion
    result = reciprocal_rank_fusion([["x", "y"], ["y", "x"]])
    ids = [r[0] for r in result]
    assert len(ids) == 2
    assert len(set(ids)) == 2  # no duplicates


def test_rrf_combines_scores():
    from services.semantic_search import reciprocal_rank_fusion, RRF_K
    # doc "b" ranked 1st in both lists: 1/(k+1) + 1/(k+1)
    # doc "a" ranked 2nd in both lists: 1/(k+2) + 1/(k+2)
    result = reciprocal_rank_fusion([["b", "a"], ["b", "a"]])
    scores = dict(result)
    assert scores["b"] > scores["a"]


def test_rrf_empty_lists():
    from services.semantic_search import reciprocal_rank_fusion
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


def test_rrf_formula_correct():
    from services.semantic_search import reciprocal_rank_fusion, RRF_K
    result = reciprocal_rank_fusion([["only"]])
    score = dict(result)["only"]
    expected = 1.0 / (RRF_K + 1)
    assert abs(score - expected) < 1e-9


# ── _bm25_search() ───────────────────────────────────────────────────────────

def _inject_bm25_corpus(doc_ids, corpus, contents):
    """Directly inject state into the semantic_search module globals."""
    import services.semantic_search as ss
    ss._bm25_doc_ids = doc_ids
    ss._bm25_corpus = corpus
    ss._bm25_contents = contents
    if ss._bm25_available and corpus:
        from rank_bm25 import BM25Okapi
        ss._bm25_index = BM25Okapi(corpus)
    else:
        ss._bm25_index = None


@pytest.mark.skipif(
    __import__("importlib.util", fromlist=["find_spec"]).find_spec("rank_bm25") is None,
    reason="rank-bm25 not installed",
)
def test_bm25_search_returns_ranked_results():
    from services.semantic_search import _bm25_search, tokenize
    _inject_bm25_corpus(
        ["doc1", "doc2", "doc3"],
        [tokenize("python machine learning"), tokenize("machine learning deep"), tokenize("cooking recipes")],
        {"doc1": "python machine learning", "doc2": "machine learning deep", "doc3": "cooking recipes"},
    )
    results = _bm25_search("python learning", n=5)
    assert len(results) > 0
    ids = [r[0] for r in results]
    assert "doc1" in ids
    # cooking docs should score 0 and be excluded
    assert "doc3" not in ids


@pytest.mark.skipif(
    __import__("importlib.util", fromlist=["find_spec"]).find_spec("rank_bm25") is None,
    reason="rank-bm25 not installed",
)
def test_bm25_search_empty_corpus():
    import services.semantic_search as ss
    ss._bm25_doc_ids = []
    ss._bm25_corpus = []
    ss._bm25_contents = {}
    ss._bm25_index = None
    from services.semantic_search import _bm25_search
    assert _bm25_search("anything") == []


def test_bm25_search_unavailable_returns_empty(monkeypatch):
    import services.semantic_search as ss
    monkeypatch.setattr(ss, "_bm25_available", False)
    monkeypatch.setattr(ss, "_bm25_index", None)
    from services.semantic_search import _bm25_search
    assert _bm25_search("query") == []


# ── Hybrid search degradation ─────────────────────────────────────────────────

def test_hybrid_falls_back_to_vector_when_bm25_unavailable(monkeypatch):
    import services.semantic_search as ss
    monkeypatch.setattr(ss, "_bm25_available", False)
    monkeypatch.setattr(ss, "_bm25_index", None)
    monkeypatch.setattr(ss, "_initialized", True)

    fake_col = MagicMock()
    fake_col.count.return_value = 5  # must be int for min(top_k, count)
    fake_col.query.return_value = {
        "ids": [["doc1"]],
        "documents": [["content1"]],
        "metadatas": [[{"source": "test.txt", "doc_type": "text"}]],
        "distances": [[0.1]],
    }
    monkeypatch.setattr(ss, "_documents_col", fake_col)

    from services.semantic_search import search_documents_hybrid
    results = search_documents_hybrid("any query", top_k=1, method="hybrid")
    # Should return results via vector path even though BM25 is disabled
    assert isinstance(results, list)
    fake_col.query.assert_called_once()

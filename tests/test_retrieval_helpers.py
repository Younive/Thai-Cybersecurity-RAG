"""Unit tests for retrieval.py pure helpers + the multilingual orchestrator.

Importable because conftest patches VectorStoreManager.get_exist_cromadb, so the
module-level `vectorstore` is a MagicMock we drive per test.
"""
import pytest
import retrieval
from langchain_core.documents import Document


# --- detect_language -----------------------------------------------------

def test_detect_language_thai():
    assert retrieval.detect_language("มาตรฐานความปลอดภัยเว็บไซต์") == "th"


def test_detect_language_english():
    assert retrieval.detect_language("broken access control") == "en"


def test_detect_language_empty_is_english():
    assert retrieval.detect_language("") == "en"
    assert retrieval.detect_language("!!! 123 ???") == "en"  # no alnum letters -> en


def test_detect_language_mixed_below_threshold():
    # A few Thai chars in a mostly-English string stays 'en' (ratio <= 0.3).
    assert retrieval.detect_language("web security standard ไทย") == "en"


# --- expand_query_with_translations --------------------------------------

def test_expand_english_appends_thai():
    out = retrieval.expand_query_with_translations("web security controls", "en")
    assert out[0] == "web security controls"  # original always first
    assert len(out) > 1
    assert any("ความปลอดภัยเว็บไซต์" in q for q in out[1:])  # Thai term injected


def test_expand_no_keyword_match_returns_original_only():
    assert retrieval.expand_query_with_translations("hello world", "en") == ["hello world"]


def test_expand_thai_appends_english():
    out = retrieval.expand_query_with_translations("มาตรฐานความปลอดภัยเว็บไซต์", "th")
    assert out[0] == "มาตรฐานความปลอดภัยเว็บไซต์"
    assert any("web security" in q or "security standard" in q for q in out[1:])


# --- is_thai_related_query ----------------------------------------------

@pytest.mark.parametrize("q,expected", [
    ("Thailand Web Security Standard", True),
    ("รายงานภาครัฐ พ.ศ. 2568", True),
    ("What is SQL injection?", False),
    ("NCSA requirements", True),  # 'ncsa'
])
def test_is_thai_related_query(q, expected):
    assert retrieval.is_thai_related_query(q) is expected


# --- filter_irrelevant_pages --------------------------------------------

def test_filter_drops_bibliography():
    keep = (Document(page_content="Real content about access control", metadata={}), 0.1)
    drop = (Document(page_content="References\n[1] Smith bibliography", metadata={}), 0.2)
    out = retrieval.filter_irrelevant_pages([keep, drop], "q")
    assert out == [keep]


def test_filter_all_irrelevant_returns_original():
    # Fallback: if everything is filtered, keep the original list unchanged.
    rows = [
        (Document(page_content="สารบัญ table of contents", metadata={}), 0.1),
        (Document(page_content="bibliography references", metadata={}), 0.2),
    ]
    assert retrieval.filter_irrelevant_pages(rows, "q") == rows


# --- retrieve_documents_multilingual ------------------------------------

def _rows(n):
    """n distinct (doc, score) tuples, ascending score = descending relevance."""
    return [
        (Document(page_content=f"chunk {i} unique body text", metadata={"page": i}), i * 0.1)
        for i in range(n)
    ]


def test_multilingual_truncates_to_original_k(monkeypatch):
    # Thai query bumps internal k, but result is capped at the requested k.
    monkeypatch.setattr(
        retrieval.vectorstore, "similarity_search_with_score",
        lambda q, k: _rows(8), raising=False,
    )
    out = retrieval.retrieve_documents_multilingual("มาตรฐานความปลอดภัย", k=3)
    assert len(out) == 3
    assert all(isinstance(d, Document) for d in out)


def test_multilingual_sorted_by_score(monkeypatch):
    shuffled = [
        (Document(page_content="c high", metadata={}), 0.9),
        (Document(page_content="a low", metadata={}), 0.1),
        (Document(page_content="b mid", metadata={}), 0.5),
    ]
    monkeypatch.setattr(
        retrieval.vectorstore, "similarity_search_with_score",
        lambda q, k: shuffled, raising=False,
    )
    out = retrieval.retrieve_documents_multilingual("plain english query", k=3,
                                                    adaptive_k=False, filter_pages=False)
    assert [d.page_content for d in out] == ["a low", "b mid", "c high"]


def test_multilingual_dedupes_by_prefix(monkeypatch):
    dupes = [
        (Document(page_content="identical prefix content", metadata={}), 0.1),
        (Document(page_content="identical prefix content", metadata={}), 0.2),
    ]
    monkeypatch.setattr(
        retrieval.vectorstore, "similarity_search_with_score",
        lambda q, k: dupes, raising=False,
    )
    out = retrieval.retrieve_documents_multilingual("plain query", k=5,
                                                    adaptive_k=False, filter_pages=False)
    assert len(out) == 1

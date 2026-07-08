"""Unit tests for the pure eval helpers (no store, no network)."""
import run_eval as ev
import generate_eval_dataset as gen


class _Doc:
    """Duck-typed stand-in for a langchain Document."""
    def __init__(self, content, source, page):
        self.page_content = content
        self.metadata = {"source": source, "page": page}


# --- normalize_source ----------------------------------------------------

def test_normalize_source_forward_and_backslash():
    assert ev.normalize_source("dataset/owasp.pdf") == "owasp.pdf"
    assert ev.normalize_source("dataset\\owasp.pdf") == "owasp.pdf"


# --- match_rank ----------------------------------------------------------

def _gold(**kw):
    base = {"source": "owasp.pdf", "page": 4, "chunk_fingerprint": "hello world fingerprint"}
    base.update(kw)
    return base


def test_match_rank_exact_chunk():
    # fingerprint match is page_content[:100] == gold fingerprint (exact).
    docs = [_Doc("hello world fingerprint", "owasp.pdf", 4)]
    assert ev.match_rank(_gold(), docs) == {"chunk": 1, "page": 1, "source": 1}


def test_match_rank_page_within_tolerance():
    # page off by 1 still counts for 'page'; fingerprint differs so 'chunk' misses.
    docs = [_Doc("different text", "owasp.pdf", 5)]
    r = ev.match_rank(_gold(), docs)
    assert r["chunk"] is None and r["page"] == 1 and r["source"] == 1


def test_match_rank_source_only():
    # page too far (>1) -> source matches, page does not.
    docs = [_Doc("different text", "owasp.pdf", 40)]
    r = ev.match_rank(_gold(), docs)
    assert r["source"] == 1 and r["page"] is None


def test_match_rank_miss():
    docs = [_Doc("nope", "mitre.pdf", 1)]
    assert ev.match_rank(_gold(), docs) == {"chunk": None, "page": None, "source": None}


# --- aggregate_retrieval (incl. new by_doc_type slice) ------------------

def test_aggregate_retrieval_hit_and_mrr():
    rows = [
        {"retriever": "ml", "lang": "en", "doc_type": "textbook",
         "rank": {"chunk": 1, "page": 1, "source": 1}},
        {"retriever": "ml", "lang": "en", "doc_type": "thai_pdf",
         "rank": {"chunk": None, "page": 2, "source": 1}},
    ]
    out = ev.aggregate_retrieval(rows)
    en = out["ml"]["en"]
    assert en["n"] == 2
    assert en["hit_at_k"]["page"] == 1.0       # both found page
    assert en["hit_at_k"]["chunk"] == 0.5      # one of two
    assert en["mrr"]["page"] == round((1 / 1 + 1 / 2) / 2, 3)


def test_aggregate_retrieval_by_doc_type_slice():
    rows = [
        {"retriever": "ml", "lang": "en", "doc_type": "textbook",
         "rank": {"chunk": 1, "page": 1, "source": 1}},
        {"retriever": "ml", "lang": "en", "doc_type": "thai_pdf",
         "rank": {"chunk": None, "page": None, "source": None}},
    ]
    by_type = ev.aggregate_retrieval(rows)["ml"]["en"]["by_doc_type"]
    assert by_type["textbook"]["hit_at_k"]["page"] == 1.0
    assert by_type["thai_pdf"]["hit_at_k"]["page"] == 0.0  # Thai doc misses


# --- aggregate_citations -------------------------------------------------

def test_aggregate_citations():
    rows = [
        {"lang": "en", "n_citations": 2, "n_grounded": 1, "gold_source_cited": True},
        {"lang": "en", "n_citations": 0, "n_grounded": 0, "gold_source_cited": False},
    ]
    en = ev.aggregate_citations(rows)["en"]
    assert en["pct_answers_with_citation"] == 0.5
    assert en["pct_citations_grounded_micro"] == 0.5   # 1 grounded / 2 total
    assert en["pct_citations_grounded_macro"] == 0.5   # only the cited answer counts
    assert en["pct_gold_source_cited"] == 0.5


# --- generate_eval_dataset: is_eligible + sample_chunks ------------------

def test_is_eligible_rejects_short():
    assert gen.is_eligible("too short", {"page": 3, "content_type": "text"}) is False


def test_is_eligible_rejects_page_zero():
    assert gen.is_eligible("x" * 300, {"page": 0, "content_type": "text"}) is False


def test_is_eligible_rejects_non_text():
    assert gen.is_eligible("x" * 300, {"page": 3, "content_type": "table"}) is False


def test_is_eligible_rejects_junk():
    assert gen.is_eligible("bibliography " + "x" * 300, {"page": 3, "content_type": "text"}) is False


def test_is_eligible_accepts_good():
    assert gen.is_eligible("x" * 300, {"page": 3, "content_type": "text"}) is True


def test_sample_chunks_deterministic():
    chunks = [
        {"text": "x" * 300, "meta": {"source": f"dataset/doc{d}.pdf", "page": p, "content_type": "text"}}
        for d in range(2) for p in range(1, 11)
    ]
    a = gen.sample_chunks(chunks, per_doc=3, seed=42)
    b = gen.sample_chunks(chunks, per_doc=3, seed=42)
    assert [c["text"] for c in a] == [c["text"] for c in b]  # same seed -> same sample
    assert len(a) == 6  # 3 per doc × 2 docs

"""Unit tests for prompt_template — fully import-safe (no chroma, no keys)."""
from prompt_template import (
    extract_citations,
    build_rag_prompt,
    _format_retrieved_docs,
    GENERATION_CONFIG,
)


# --- extract_citations ---------------------------------------------------

def test_extract_citations_english():
    text = "SQL injection is bad [Source: owasp-top-10.pdf, Page 4]."
    assert extract_citations(text) == [{"source": "owasp-top-10.pdf", "page": "4"}]


def test_extract_citations_thai():
    text = "มาตรฐาน [แหล่งที่มา: thailand-web-security-standard-2025.pdf, หน้า 5]"
    assert extract_citations(text) == [
        {"source": "thailand-web-security-standard-2025.pdf", "page": "5"}
    ]


def test_extract_citations_mixed_en_first():
    text = "A [Source: a.pdf, Page 1] และ B [แหล่งที่มา: b.pdf, หน้า 2]"
    # EN matches appended before TH.
    assert extract_citations(text) == [
        {"source": "a.pdf", "page": "1"},
        {"source": "b.pdf", "page": "2"},
    ]


def test_extract_citations_page_unknown():
    assert extract_citations("[Source: x.pdf, Page unknown]") == [
        {"source": "x.pdf", "page": "unknown"}
    ]


def test_extract_citations_multiple_same_source():
    text = "[Source: m.pdf, Page 12] ... [Source: m.pdf, Page 13]"
    assert extract_citations(text) == [
        {"source": "m.pdf", "page": "12"},
        {"source": "m.pdf", "page": "13"},
    ]


def test_extract_citations_none():
    assert extract_citations("No citations here.") == []


def test_extract_citations_comma_in_filename_fails():
    # Documented limitation: [^,]+ can't cross a comma, so a filename with one
    # breaks the whole match -> no citation extracted at all.
    assert extract_citations("[Source: a,b.pdf, Page 1]") == []


# --- build_rag_prompt ----------------------------------------------------

def test_build_rag_prompt_detects_thai(make_doc):
    doc = make_doc("เนื้อหา", source="th.pdf", page=1)
    prompt = build_rag_prompt("มาตรฐานความปลอดภัยเว็บไซต์คืออะไร", [doc], language="auto")
    assert "ตอบเป็นภาษาไทย" in prompt  # Thai citation-format block


def test_build_rag_prompt_detects_english(make_doc):
    doc = make_doc("content", source="en.pdf", page=1)
    prompt = build_rag_prompt("What is broken access control?", [doc], language="auto")
    assert "Answer in English" in prompt


def test_build_rag_prompt_empty_docs():
    prompt = build_rag_prompt("anything", [], language="en")
    assert "[No documents retrieved]" in prompt


# --- _format_retrieved_docs ---------------------------------------------

def test_format_strips_forward_slash_path(make_doc):
    doc = make_doc("body", source="dataset/owasp-top-10.pdf", page=4)
    out = _format_retrieved_docs([doc])
    assert "Source: owasp-top-10.pdf, Page: 4" in out


def test_format_does_not_strip_backslash_path(make_doc):
    # Documented limitation: only '/' is split, so a Windows path stays whole.
    doc = make_doc("body", source="dataset\\owasp.pdf", page=4)
    out = _format_retrieved_docs([doc])
    assert "dataset\\owasp.pdf" in out


def test_format_empty():
    assert _format_retrieved_docs([]) == "[No documents retrieved]"


def test_generation_config_shape():
    assert set(GENERATION_CONFIG) == {"temperature", "top_p", "max_tokens"}
    assert "top_k" not in GENERATION_CONFIG  # not an OpenAI param

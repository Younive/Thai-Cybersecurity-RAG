"""Unit tests for rag_pipeline chunk routing/sizing + data_model backfill.

RAGPipeline construction builds OpenAIEmbeddings (dummy key, no network until an
embed call — never made here). No chroma_db needed.
"""
import pytest
from langchain_core.documents import Document

from rag_pipeline import RAGPipeline
from extractors.utils.data_model import ExtractedContent, ContentType


@pytest.fixture(scope="module")
def pipeline():
    return RAGPipeline()


# --- routing in _chunk_all_content --------------------------------------

def _short(content, **md):
    # short (<1000 chars) -> exactly one chunk per doc, so counts are exact.
    return Document(page_content=content, metadata=md)


def test_routing_all_four_buckets(pipeline):
    docs = [
        _short("textbook body", doc_type="textbook"),
        _short("slide body", slide_num=1, content_type="text"),
        _short("thai body", extraction_method="pytesseract_ocr"),
        _short("generic body"),  # -> other
    ]
    chunks = pipeline._chunk_all_content(docs)
    bodies = {c.page_content for c in chunks}
    assert {"textbook body", "slide body", "thai body", "generic body"} <= bodies


def test_routing_double_bucket_double_chunks(pipeline):
    # Documented trap: a doc matching two buckets (textbook + slide_num) is
    # chunked by both routes -> appears twice.
    doc = _short("dual routed", doc_type="textbook", slide_num=1, content_type="text")
    chunks = pipeline._chunk_all_content([doc])
    assert sum(1 for c in chunks if c.page_content == "dual routed") == 2


# --- individual chunkers -------------------------------------------------

def test_slide_chunker_keeps_table_atomic(pipeline):
    text = Document(page_content="slide text", metadata={"slide_num": 2, "content_type": "text"})
    table = Document(page_content="<table>...</table>", metadata={"slide_num": 2, "content_type": "table"})
    chunks = pipeline._chunk_slide_style([text, table])
    table_chunks = [c for c in chunks if c.metadata.get("content_type") == "table"]
    assert len(table_chunks) == 1
    assert table_chunks[0].page_content == "<table>...</table>"  # unchanged, not split


def test_textbook_chunk_size(pipeline):
    long = Document(page_content="word " * 600, metadata={"doc_type": "textbook"})  # ~3000 chars
    chunks = pipeline._chunk_textbook_style([long])
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 1000 for c in chunks)


def test_thai_chunk_size(pipeline):
    long = Document(page_content="ก" * 3000, metadata={"extraction_method": "pytesseract_ocr"})
    chunks = pipeline._chunk_thai_style([long])
    assert all(len(c.page_content) <= 800 for c in chunks)


# --- _to_langchain_docs --------------------------------------------------

def test_to_langchain_docs_table_and_image(pipeline):
    table = ExtractedContent(
        content="table text", content_type=ContentType.TABLE,
        metadata={"source": "s.pdf", "page": 1, "table_html": "<table/>"},
    )
    image = ExtractedContent(
        content="img desc", content_type=ContentType.IMAGE,
        metadata={"source": "s.pdf", "page": 1}, image_base64="abc123",
    )
    docs = pipeline._to_langchain_docs([table, image])
    tmeta, imeta = docs[0].metadata, docs[1].metadata
    assert tmeta["content_type"] == "table" and tmeta["table_html"] == "<table/>"
    assert imeta["has_image"] is True and imeta["image_base64"] == "abc123"


# --- ExtractedContent.__post_init__ backfill -----------------------------

def test_post_init_backfills_doc_id_and_page():
    c = ExtractedContent(
        content="x", content_type=ContentType.TEXT,
        metadata={"source": "dataset/owasp-top-10.pdf", "page_number": 7},
    )
    assert c.metadata["doc_id"] == "owasp-top-10"
    assert c.metadata["page"] == 7


def test_post_init_defaults_when_source_missing():
    c = ExtractedContent(content="x", content_type=ContentType.TEXT, metadata={})
    assert c.metadata["doc_id"] == "unknown"
    assert c.metadata["page"] == 0

"""FastAPI contract tests. Imports `api` under the conftest patch (no real
chroma_db / key); retrieval + chat model are stubbed per test so nothing hits
the network. Covers every response branch + source stripping + rate limit.
"""
import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIError, APIStatusError
from langchain_core.documents import Document

import api


@pytest.fixture(autouse=True)
def isolate_limiter():
    # Disable the 20/hour limiter for every test except the one that opts back
    # in, so accumulated counts don't leak across tests.
    api.limiter.enabled = False
    yield
    api.limiter.enabled = True


@pytest.fixture
def client():
    return TestClient(api.app)


class _Msg:
    def __init__(self, content):
        self.content = content


class _FakeModel:
    def __init__(self, *, content=None, exc=None):
        self._content, self._exc = content, exc

    def invoke(self, prompt):
        if self._exc is not None:
            raise self._exc
        return _Msg(self._content)


def _stub_model(monkeypatch, *, content=None, exc=None):
    # ChatOpenAI is a frozen pydantic model — replace the whole global.
    monkeypatch.setattr(api, "model", _FakeModel(content=content, exc=exc))


# --- request validation --------------------------------------------------

def test_empty_question_400(client, monkeypatch):
    monkeypatch.setattr(api, "retrieve_documents_multilingual", lambda *a, **k: [])
    r = client.post("/query", json={"question": "   "})
    assert r.status_code == 400
    assert r.json()["detail"] == "Question is empty."


# --- retrieval branch ----------------------------------------------------

def test_retrieval_failure_502(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("chroma down")
    monkeypatch.setattr(api, "retrieve_documents_multilingual", boom)
    r = client.post("/query", json={"question": "q"})
    assert r.status_code == 502
    assert "Retrieval failed" in r.json()["detail"]


def test_no_results_canned_200(client, monkeypatch):
    monkeypatch.setattr(api, "retrieve_documents_multilingual", lambda *a, **k: [])
    r = client.post("/query", json={"question": "q"})
    assert r.status_code == 200
    body = r.json()
    assert body["sources"] == [] and body["citations"] == []
    assert "No relevant documents" in body["answer"]


# --- chat model errors ---------------------------------------------------

def test_api_status_error_passes_through(client, monkeypatch):
    monkeypatch.setattr(api, "retrieve_documents_multilingual",
                        lambda *a, **k: [Document(page_content="x", metadata={})])
    resp = httpx.Response(429, request=httpx.Request("POST", "http://openrouter"))
    _stub_model(monkeypatch, exc=APIStatusError("rate limited", response=resp, body=None))
    r = client.post("/query", json={"question": "q"})
    assert r.status_code == 429


def test_api_error_502(client, monkeypatch):
    monkeypatch.setattr(api, "retrieve_documents_multilingual",
                        lambda *a, **k: [Document(page_content="x", metadata={})])
    req = httpx.Request("POST", "http://openrouter")
    _stub_model(monkeypatch, exc=APIError("connection dropped", request=req, body=None))
    r = client.post("/query", json={"question": "q"})
    assert r.status_code == 502
    assert "Chat model unreachable" in r.json()["detail"]


# --- happy path ----------------------------------------------------------

def test_happy_path_shape_and_source_stripping(client, monkeypatch):
    doc = Document(
        page_content="Broken access control\nis the top risk.",
        metadata={"source": "dataset/owasp-top-10.pdf", "page": 12},
    )
    monkeypatch.setattr(api, "retrieve_documents_multilingual", lambda *a, **k: [doc])
    _stub_model(monkeypatch, content="It is broken access control [Source: owasp-top-10, Page 12].")
    r = client.post("/query", json={"question": "top owasp risk?"})
    assert r.status_code == 200
    body = r.json()
    src = body["sources"][0]
    assert src["source"] == "owasp-top-10"          # dataset/ and .pdf stripped
    assert src["page"] == 12
    assert src["preview"] == "Broken access control is the top risk."  # newline -> space
    assert body["citations"] == [{"source": "owasp-top-10", "page": "12"}]


def test_default_k_is_8(client, monkeypatch):
    captured = {}
    def spy(question, k, adaptive_k, filter_pages):
        captured["k"] = k
        return []
    monkeypatch.setattr(api, "retrieve_documents_multilingual", spy)
    client.post("/query", json={"question": "q"})  # no k in body
    assert captured["k"] == 8


# --- rate limit ----------------------------------------------------------

def test_rate_limit_429_after_20(client, monkeypatch):
    api.limiter.enabled = True
    api.limiter.reset()
    monkeypatch.setattr(api, "retrieve_documents_multilingual", lambda *a, **k: [])
    codes = [client.post("/query", json={"question": "q"}).status_code for _ in range(21)]
    assert codes[-1] == 429
    assert codes[:20] == [200] * 20

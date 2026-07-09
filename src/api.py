"""
FastAPI wrapper exposing the RAG pipeline to the Next.js frontend.

Run from repo root so `src/` is on sys.path and ./chroma_db resolves:
    uv run uvicorn --app-dir src api:app --port 8000

NOTE: importing `retrieval` loads the Chroma vector store at import time, so
chroma_db/ must exist and OPENROUTER_API_KEY must be set before uvicorn starts
(same constraint as app.py — see CLAUDE.md).
"""
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from openai import APIError, APIStatusError
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from retrieval import retrieve_documents_multilingual
from prompt_template import build_rag_prompt, extract_citations, GENERATION_CONFIG

load_dotenv(".env.local")

model = ChatOpenAI(
    model=os.getenv("OPENROUTER_GENERAL_MODEL"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    **GENERATION_CONFIG,
)

app = FastAPI(title="Thai Cybersecurity RAG API")

# Per-IP rate limit. Each request fires paid OpenRouter calls; throttle abuse.
# ponytail: in-memory (per-process) — single uvicorn worker only. Multi-worker
# needs a shared backend (redis via Limiter(storage_uri=...)).
def _client_ip(request: Request) -> str:
    # Behind Vercel rewrite + Render proxy; first X-Forwarded-For hop is the real client.
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else get_remote_address(request)

limiter = Limiter(key_func=_client_ip)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Next.js dev server. The rewrite in web/next.config.ts is the primary path;
# this CORS rule is belt-and-braces for direct browser calls to :8000.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    k: int = 8


@app.post("/query")
@limiter.limit("20/hour")
def query(request: Request, req: QueryRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty.")

    try:
        results = retrieve_documents_multilingual(
            question, k=req.k, adaptive_k=True, filter_pages=True
        )
    except Exception as e:
        # Retrieval hits the embedding API (OpenRouter) + Chroma; surface, don't 500.
        raise HTTPException(status_code=502, detail=f"Retrieval failed: {e}") from e

    if not results:
        return {
            "answer": "No relevant documents found. Try rephrasing or raising k.",
            "sources": [],
            "citations": [],
        }

    prompt = build_rag_prompt(question, results, language="auto")
    try:
        answer = model.invoke(prompt).content  # ChatOpenAI returns AIMessage
    except APIStatusError as e:
        # Pass through the upstream status (429 rate limit, 5xx, etc.) with its message.
        raise HTTPException(status_code=e.status_code, detail=str(e)) from e
    except APIError as e:
        # Network/timeout/connection errors have no HTTP status of their own.
        raise HTTPException(status_code=502, detail=f"Chat model unreachable: {e}") from e

    citations = extract_citations(answer)

    sources = [
        {
            "source": doc.metadata.get("source", "unknown")
            .replace("dataset/", "")
            .replace(".pdf", ""),
            "page": doc.metadata.get("page", "unknown"),
            "preview": doc.page_content[:120].replace("\n", " ").strip(),
        }
        for doc in results
    ]

    return {"answer": answer, "sources": sources, "citations": citations}

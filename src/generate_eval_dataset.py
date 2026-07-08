"""
Generate a gold QA dataset for RAG evaluation.

Samples chunks straight from the Chroma collection (no retrieval.py import —
that module loads the vector store at import time), asks the chat model to
write one bilingual Q/A pair per chunk, and writes eval/dataset.json for
human spot-check/editing before use with run_eval.py.

Run from repo root:
    uv run python src/generate_eval_dataset.py --limit 3 --out eval/dataset-smoke.json
    uv run python src/generate_eval_dataset.py
"""
import argparse
import json
import os
import random
import re
from datetime import datetime, timezone
from itertools import zip_longest

import chromadb
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from prompt_template import GENERATION_CONFIG

load_dotenv(".env.local")

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "rag_knowledge_base"

# Copied from retrieval.py::filter_irrelevant_pages (local variable there,
# not importable without triggering the store load).
JUNK_INDICATORS = [
    'บรรณานุกรม',  # Bibliography in Thai
    'bibliography',
    'references',
    'สารบัญ',  # Table of contents in Thai
    'table of contents',
    'อ้างอิง',  # References in Thai
]

QA_PROMPT = """You are building an evaluation dataset for a cybersecurity RAG system.

Given this excerpt from "{source}" page {page}, write ONE factual question that is answerable ONLY from this excerpt, plus a concise reference answer.

Excerpt:
---
{text}
---

Return strict JSON, no code fences, exactly these keys:
{{"question_en": "...", "question_th": "...", "answer_en": "..."}}

question_th must be a natural Thai rendering of the same question, not a transliteration."""


def normalize_source(s: str) -> str:
    """Strip path prefixes (dataset/, backslashes) down to the bare filename."""
    return s.replace("\\", "/").split("/")[-1]


def load_chunks() -> list:
    collection = chromadb.PersistentClient(CHROMA_PATH).get_collection(COLLECTION_NAME)
    data = collection.get(include=["documents", "metadatas"])
    return [
        {"text": doc, "meta": meta}
        for doc, meta in zip(data["documents"], data["metadatas"])
    ]


def is_eligible(text: str, meta: dict) -> bool:
    if len(text.strip()) < 250:
        return False  # too short for a meaningful question
    page = meta.get("page")
    if not isinstance(page, int) or page <= 0:
        return False  # page defaults to 0 when unknown -> no reliable gold label
    if meta.get("content_type") != "text":
        return False  # table/image chunks are often just captions
    lowered = text.lower()
    return not any(ind in lowered for ind in JUNK_INDICATORS)


def sample_chunks(chunks: list, per_doc: int, seed: int) -> list:
    by_source = {}
    for c in chunks:
        if is_eligible(c["text"], c["meta"]):
            by_source.setdefault(normalize_source(c["meta"]["source"]), []).append(c)

    rng = random.Random(seed)
    per_source = []
    for source, group in sorted(by_source.items()):
        if len(group) < per_doc:
            print(f"Warning: {source} has only {len(group)} eligible chunks (< {per_doc}), taking all")
            per_source.append(group)
        else:
            per_source.append(rng.sample(group, per_doc))
    # Interleave across docs so a --limit slice stays representative (doc1, doc2, doc3, doc1, ...)
    return [c for row in zip_longest(*per_source) for c in row if c is not None]


def generate_qa(chunk: dict, model: ChatOpenAI) -> dict | None:
    meta = chunk["meta"]
    prompt = QA_PROMPT.format(
        source=normalize_source(meta["source"]),
        page=meta["page"],
        text=chunk["text"][:3000],
    )
    for attempt in range(2):
        raw = model.invoke(prompt).content
        # Strip ```json ... ``` fences if the model adds them anyway
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        try:
            qa = json.loads(cleaned)
            if all(k in qa for k in ("question_en", "question_th", "answer_en")):
                return qa
        except json.JSONDecodeError:
            pass
        if attempt == 0:
            print(f"  Parse failed, retrying (page {meta['page']})")
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate gold QA dataset from vector store chunks")
    parser.add_argument("--per-doc", type=int, default=14, help="questions per source document")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="eval/dataset.json")
    parser.add_argument("--limit", type=int, default=None, help="cap total chunks processed (smoke test)")
    args = parser.parse_args()

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {COLLECTION_NAME}")

    sampled = sample_chunks(chunks, args.per_doc, args.seed)
    if args.limit:
        sampled = sampled[: args.limit]
    print(f"Sampled {len(sampled)} chunks, generating Q/A pairs...")

    model = ChatOpenAI(
        model=os.getenv("OPENROUTER_GENERAL_MODEL"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        **GENERATION_CONFIG,
    )

    questions = []
    for i, chunk in enumerate(sampled, 1):
        meta = chunk["meta"]
        source = normalize_source(meta["source"])
        print(f"[{i}/{len(sampled)}] {source} p{meta['page']}")
        qa = generate_qa(chunk, model)
        if qa is None:
            print("  Skipped: could not parse LLM output")
            continue
        questions.append({
            "id": f"q{len(questions) + 1:03d}",
            "question_en": qa["question_en"],
            "question_th": qa["question_th"],
            "answer_en": qa["answer_en"],
            "gold": {
                "source": source,
                "page": meta["page"],
                # Same fingerprint key production dedupe uses (retrieval.py).
                # NOTE: goes stale if rag_pipeline.py re-chunks; source/page survive.
                "chunk_fingerprint": chunk["text"][:100],
                # doc_type isn't stored; derive like the chunk router in rag_pipeline
                "doc_type": ("slides" if meta.get("slide_num") is not None
                             else "thai_pdf" if meta.get("extraction_method") == "pytesseract_ocr"
                             else "textbook"),
            },
            "chunk_preview": chunk["text"][:300],
        })

    dataset = {
        "created": datetime.now(timezone.utc).isoformat(),
        "generator_model": os.getenv("OPENROUTER_GENERAL_MODEL"),
        "seed": args.seed,
        "note": "Human spot-check before use. Thai doc pages are OCR render indices, "
                "not printed page numbers. chunk_fingerprint stales if store is rebuilt.",
        "questions": questions,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(questions)} questions to {args.out}")


if __name__ == "__main__":
    main()

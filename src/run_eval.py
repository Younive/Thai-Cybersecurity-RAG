"""
RAG evaluation harness: retrieval metrics + citation accuracy, split EN vs TH.

Default run is retrieval-only (embedding calls, pennies). --generate adds
answer generation + citation grounding (one chat call per question per lang).

Run from repo root:
    uv run python src/run_eval.py --help                      # no store touch
    uv run python src/run_eval.py                             # retrieval metrics
    uv run python src/run_eval.py --generate --lang en        # + citation eval
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time
from datetime import datetime, timezone


def retry(fn, tries=3, delay=2):
    """Retry an API-hitting call. OpenRouter intermittently returns empty
    data (embed) or rate-limits; one hiccup shouldn't kill a 168-call batch."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            if attempt == tries - 1:
                raise
            print(f"\n  retry {attempt + 1}/{tries - 1} after error: {e}")
            time.sleep(delay * (attempt + 1))


def normalize_source(s: str) -> str:
    return str(s).replace("\\", "/").split("/")[-1]


def load_dataset(path: str) -> dict:
    if not os.path.isfile(path):
        sys.exit(f"Dataset not found: {path} — run `uv run python src/generate_eval_dataset.py` first.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@contextlib.contextmanager
def quiet():
    """Suppress the per-call prints inside retrieve_documents_multilingual."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def match_rank(gold: dict, docs: list) -> dict:
    """1-indexed rank of first match at each level, None if absent.

    chunk:  exact chunk retrieved (fingerprint = page_content[:100], same key
            as production dedupe). Strictest; penalizes neighbor chunks.
    page:   source match AND page within ±1 (chunks overlap/straddle pages).
            The headline metric.
    source: filename only. Sanity floor — 3 docs, random ≈ 33%.
    """
    ranks = {"chunk": None, "page": None, "source": None}
    for i, doc in enumerate(docs, 1):
        src = normalize_source(doc.metadata.get("source", ""))
        page = doc.metadata.get("page")
        if ranks["chunk"] is None and doc.page_content[:100] == gold["chunk_fingerprint"]:
            ranks["chunk"] = i
        if src == gold["source"]:
            if ranks["source"] is None:
                ranks["source"] = i
            if ranks["page"] is None and isinstance(page, int) and abs(page - gold["page"]) <= 1:
                ranks["page"] = i
    return ranks


def eval_retrieval(questions: list, k: int, retrievers: dict, langs: list) -> list:
    rows = []
    total = len(questions) * len(langs) * len(retrievers)
    done = 0
    try:
        for q in questions:
            for lang in langs:
                query = q.get(f"question_{lang}")
                if not query:
                    print(f"\n  skip {q['id']}: missing question_{lang}")
                    continue
                for name, fn in retrievers.items():
                    done += 1
                    print(f"\r  retrieval {done}/{total}", end="", flush=True)
                    with quiet():
                        docs = retry(lambda: fn(query, k=k))
                    rows.append({
                        "id": q["id"], "lang": lang, "retriever": name,
                        "doc_type": q["gold"].get("doc_type"),
                        "rank": match_rank(q["gold"], docs),
                    })
    except Exception as e:
        print(f"\n  retrieval aborted at {done}/{total}: {e} — keeping {len(rows)} rows")
    print()
    return rows


def eval_citations(questions: list, k: int, langs: list) -> list:
    # Mirrors api.py exactly: multilingual retrieval -> build_rag_prompt -> chat -> extract_citations
    from langchain_openai import ChatOpenAI
    from retrieval import retrieve_documents_multilingual
    from prompt_template import build_rag_prompt, extract_citations, GENERATION_CONFIG

    model = ChatOpenAI(
        model=os.getenv("OPENROUTER_GENERAL_MODEL"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        **GENERATION_CONFIG,
    )

    rows = []
    total = len(questions) * len(langs)
    done = 0
    try:
        for q in questions:
            for lang in langs:
                done += 1
                print(f"\r  citations {done}/{total}", end="", flush=True)
                query = q.get(f"question_{lang}")
                if not query:
                    print(f"\n  skip {q['id']}: missing question_{lang}")
                    continue
                with quiet():
                    docs = retry(lambda: retrieve_documents_multilingual(query, k=k))
                prompt = build_rag_prompt(query, docs, language="auto")
                answer = retry(lambda: model.invoke(prompt)).content
                citations = extract_citations(answer)

                # Exact page match here (no ±1): the prompt shows literal page
                # numbers, copying them correctly is what's being measured.
                in_prompt = {
                    (normalize_source(d.metadata.get("source", "")), d.metadata.get("page"))
                    for d in docs
                }
                grounded = sum(
                    1 for c in citations
                    if c["page"].isdigit()
                    and (normalize_source(c["source"]).strip(), int(c["page"])) in in_prompt
                )
                gold_cited = any(
                    normalize_source(c["source"]).strip() == q["gold"]["source"]
                    for c in citations
                )
                rows.append({
                    "id": q["id"], "lang": lang,
                    "n_citations": len(citations), "n_grounded": grounded,
                    "gold_source_cited": gold_cited,
                })
    except Exception as e:
        print(f"\n  citations aborted at {done}/{total}: {e} — keeping {len(rows)} rows")
    print()
    return rows


def _hit_mrr(rs: list) -> dict:
    # One gold chunk per question -> recall@k == hit@k, so only hit@k reported.
    metrics = {"n": len(rs), "hit_at_k": {}, "mrr": {}}
    for level in ("chunk", "page", "source"):
        ranks = [r["rank"][level] for r in rs]
        metrics["hit_at_k"][level] = round(sum(1 for x in ranks if x) / len(ranks), 3)
        metrics["mrr"][level] = round(sum(1 / x for x in ranks if x) / len(ranks), 3)
    return metrics


def aggregate_retrieval(rows: list) -> dict:
    out = {}
    groups = {}
    for r in rows:
        groups.setdefault((r["retriever"], r["lang"]), []).append(r)
        groups.setdefault((r["retriever"], "overall"), []).append(r)
    for (retriever, lang), rs in groups.items():
        metrics = _hit_mrr(rs)
        # Per-doc-type slice: exposes how much worse the Thai-OCR doc / slides
        # retrieve than the textbook (the headline weakness this system fights).
        by_type = {}
        for r in rs:
            by_type.setdefault(r.get("doc_type") or "unknown", []).append(r)
        metrics["by_doc_type"] = {t: _hit_mrr(trs) for t, trs in sorted(by_type.items())}
        out.setdefault(retriever, {})[lang] = metrics
    return out


def aggregate_citations(rows: list) -> dict:
    out = {}
    groups = {}
    for r in rows:
        groups.setdefault(r["lang"], []).append(r)
        groups.setdefault("overall", []).append(r)
    for lang, rs in groups.items():
        n_cit = sum(r["n_citations"] for r in rs)
        cited = [r for r in rs if r["n_citations"]]  # answers with ≥1 citation
        out[lang] = {
            "n": len(rs),
            "pct_answers_with_citation": round(len(cited) / len(rs), 3),
            # micro = grounded/total citations (citation-heavy answers weigh more);
            # macro = mean per-answer grounding rate (each answer weighs equally)
            "pct_citations_grounded_micro": round(sum(r["n_grounded"] for r in rs) / n_cit, 3) if n_cit else None,
            "pct_citations_grounded_macro": round(
                sum(r["n_grounded"] / r["n_citations"] for r in cited) / len(cited), 3) if cited else None,
            "pct_gold_source_cited": round(sum(1 for r in rs if r["gold_source_cited"]) / len(rs), 3),
        }
    return out


def print_summary(results: dict, k: int):
    ret = results.get("retrieval")
    if ret:
        print(f"\nRETRIEVAL (k={k})")
        header = f"{'retriever':<14}{'lang':<9}{'n':<5}{'hit@k(page)':<13}{'MRR(page)':<11}{'hit@k(src)':<12}{'hit@k(chunk)':<13}"
        print(header)
        print("-" * len(header))
        for retriever in sorted(ret):
            for lang in ("en", "th", "overall"):
                m = ret[retriever].get(lang)
                if not m:
                    continue
                print(f"{retriever:<14}{lang:<9}{m['n']:<5}"
                      f"{m['hit_at_k']['page']:<13}{m['mrr']['page']:<11}"
                      f"{m['hit_at_k']['source']:<12}{m['hit_at_k']['chunk']:<13}")

    cit = results.get("citation")
    if cit:
        print("\nCITATIONS (multilingual path)")
        for lang in ("en", "th", "overall"):
            m = cit.get(lang)
            if not m:
                continue
            micro = m["pct_citations_grounded_micro"]
            macro = m["pct_citations_grounded_macro"]
            grounded = f"{micro:.0%}/{macro:.0%}" if micro is not None else "n/a"
            print(f"  {lang:<8} {m['pct_answers_with_citation']:.0%} answers cited | "
                  f"{grounded} grounded (micro/macro) | "
                  f"{m['pct_gold_source_cited']:.0%} cite gold source")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval + citations")
    parser.add_argument("--dataset", default="eval/dataset.json")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="cap number of questions")
    parser.add_argument("--generate", action="store_true", help="also run citation eval (chat calls, costs money)")
    parser.add_argument("--retriever", choices=["both", "multilingual", "baseline"], default="both")
    parser.add_argument("--lang", choices=["both", "en", "th"], default="both")
    parser.add_argument("--out", default=None, help="results JSON path (default eval/results/<timestamp>.json)")
    args = parser.parse_args()

    if not os.path.isdir("./chroma_db"):
        sys.exit("chroma_db/ not found — run `uv run python src/rag_pipeline.py` first (from repo root).")

    dataset = load_dataset(args.dataset)
    questions = dataset["questions"][: args.limit] if args.limit else dataset["questions"]
    langs = ["en", "th"] if args.lang == "both" else [args.lang]

    # Deferred: retrieval.py loads the Chroma store at import time.
    import retrieval

    retrievers = {}
    if args.retriever in ("both", "multilingual"):
        retrievers["multilingual"] = retrieval.retrieve_documents_multilingual
    if args.retriever in ("both", "baseline"):
        retrievers["baseline"] = retrieval.retrieve_documents

    print(f"Evaluating {len(questions)} questions × {langs} × {list(retrievers)}")
    retrieval_rows = eval_retrieval(questions, args.k, retrievers, langs)

    citation_rows = []
    if args.generate:
        print(f"Generating answers for citation eval ({len(questions) * len(langs)} chat calls)...")
        citation_rows = eval_citations(questions, args.k, langs)

    results = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "config": {"k": args.k, "limit": args.limit, "generate": args.generate,
                   "retriever": args.retriever, "lang": args.lang},
        "retrieval": aggregate_retrieval(retrieval_rows),
        "citation": aggregate_citations(citation_rows) if citation_rows else None,
        "per_question": retrieval_rows + citation_rows,
    }

    out = args.out or f"eval/results/{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S')}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print_summary(results, args.k)
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()

"""
run_eval.py

Runs the question set through the RAG pipeline and reports metrics.

  python run_eval.py --retrieval-only     # free, no API key: retrieval accuracy, latency, context size
  python run_eval.py --mock               # offline test of the whole pipeline (canned answers, NOT real)
  python run_eval.py                      # live: real Claude answers + faithfulness judging (needs API key)

Results are saved to results/ as JSON. Mock results are labelled and must never be quoted
as real.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from rag.generator import MODEL
from rag.judge import PASS_SCORE, judge_faithfulness, mock_judge
from rag.metrics import key_facts_present, summarize
from rag.pipeline import RagPipeline

ROOT = Path(__file__).resolve().parent


def load_questions(path):
    questions = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions:
        raise ValueError(f"No questions found in {path}")
    return questions


def evaluate(pipeline, questions, client=None):
    records = []
    for q in questions:
        record = {
            "id": q["id"], "question": q["question"], "style": q.get("style", "direct"),
            "answerable": bool(q["answerable"]), "gold_doc": q.get("gold_doc"),
            "key_facts": q.get("key_facts", []), "answer": None,
        }
        try:
            result, chunks = pipeline.ask(q["question"])
            record.update({k: result[k] for k in
                           ("retrieved", "retrieval_ms", "context_words", "full_context_words")})
            gen = result["generation"]
            if gen is not None:
                parsed = gen["parsed"]
                record.update({
                    "answer": parsed["answer"], "answered": parsed["answerable"],
                    "parse_error": parsed["parse_error"], "sources": parsed["sources"],
                    "usage": gen["usage"], "generation_latency_seconds": gen["latency_seconds"],
                })
                retrieved_ids = {r["chunk_id"] for r in result["retrieved"]}
                retrieved_docs = {r["chunk_id"]: r["doc_id"] for r in result["retrieved"]}
                record["citations_valid"] = bool(parsed["sources"]) and all(s in retrieved_ids for s in parsed["sources"])
                record["cites_gold_doc"] = any(retrieved_docs.get(s) == record["gold_doc"] for s in parsed["sources"])
                record["correct"] = bool(q["answerable"] and parsed["answerable"]
                                         and key_facts_present(parsed["answer"], record["key_facts"]))
                if pipeline.mode == "mock":
                    score, reason, jusage = mock_judge(q["question"], parsed["answer"], chunks)
                else:
                    score, reason, jusage = judge_faithfulness(client, q["question"], parsed["answer"], chunks, pipeline.model)
                record.update({"judge_score": score, "judge_reason": reason, "judge_usage": jusage})
        except Exception as exc:  # one failed call should not lose the whole run
            record["error"] = f"{type(exc).__name__}: {exc}"
            record.setdefault("retrieved", [])
            record.setdefault("retrieval_ms", 0.0)
            record.setdefault("context_words", 0)
            record.setdefault("full_context_words", pipeline.full_context_words)
        records.append(record)
    return records


def print_report(mode, metrics):
    def line(label, m):
        return f"  {label:<28}: {m['count']}/{m['total']} ({m['rate'] * 100:.1f}%)" if m["rate"] is not None else f"  {label:<28}: n/a"

    print("\n=== Results ===")
    print(f"Questions: {metrics['questions']} ({metrics['answerable']} answerable, {metrics['unanswerable']} not answerable), top-{metrics['k']} retrieval")
    print("\nRetrieval (real measurements)")
    print(line(f"recall@{metrics['k']} (all)", metrics["retrieval_recall_at_k"]))
    for style in ("direct", "paraphrased"):
        if f"retrieval_recall_at_k_{style}" in metrics:
            print(line(f"  recall@{metrics['k']} {style}", metrics[f"retrieval_recall_at_k_{style}"]))
    print(f"  {'MRR':<28}: {metrics['retrieval_mrr']}")
    lat = metrics["retrieval_latency_ms"]
    print(f"  {'latency (ms)':<28}: mean {lat['mean']}, p50 {lat['p50']}, p95 {lat['p95']}, max {lat['max']}")
    cr = metrics["context_reduction"]
    print(f"  {'context sent to model':<28}: {cr['mean_context_words']} words vs {cr['full_handbook_words']} for the whole handbook "
          f"-> {cr['reduction'] * 100:.1f}% smaller ({cr['note']})")
    if "answer_correctness" in metrics:
        label = "Generation (CANNED mock values, not real)" if mode == "mock" else "Generation and judging (live)"
        print(f"\n{label}")
        print(line("answer correctness", metrics["answer_correctness"]))
        print(line("abstains when unanswerable", metrics["abstention_accuracy"]))
        print(line("wrongly abstains", metrics["false_abstain_rate"]))
        print(line("citation accuracy", metrics["citation_accuracy"]))
        if "faithfulness_rate" in metrics:
            print(line(f"faithfulness (judge >= {PASS_SCORE})", metrics["faithfulness_rate"]))
            print(f"  {'avg faithfulness score':<28}: {metrics['faithfulness_avg_score']}/5")
        print(f"  {'parse errors / API errors':<28}: {metrics['answer_parse_errors']} / {metrics['generation_errors']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--questions", default=str(ROOT / "eval" / "questions.yaml"))
    parser.add_argument("--docs", default=str(ROOT / "data" / "docs"))
    parser.add_argument("--k", type=int, default=3, help="chunks retrieved per question")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--retrieval-only", action="store_true", help="no API calls; real retrieval metrics")
    mode_group.add_argument("--mock", action="store_true", help="canned offline generation and judging")
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    args = parser.parse_args(argv)

    mode = "retrieval" if args.retrieval_only else "mock" if args.mock else "live"
    client = None
    if mode == "live":
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("No ANTHROPIC_API_KEY found. Put it in a .env file, or try --retrieval-only (free) or --mock.")
            return 1
        import anthropic
        client = anthropic.Anthropic()

    banners = {
        "retrieval": "Mode: RETRIEVAL-ONLY (no API calls; retrieval numbers are real)",
        "mock": "Mode: MOCK (canned offline answers and judge scores; NOT real results, do not quote them)",
        "live": f"Mode: LIVE ({MODEL} for answers and judging)",
    }
    print(banners[mode])

    pipeline = RagPipeline(args.docs, k=args.k, mode=mode, client=client)
    questions = load_questions(args.questions)
    print(f"Handbook: {len(pipeline.docs)} documents, {len(pipeline.chunks)} chunks, {pipeline.full_context_words} words")
    records = evaluate(pipeline, questions, client)
    metrics = summarize(records, args.k)
    print_report(mode, metrics)

    out = {"mode": mode, "model": MODEL if mode == "live" else None,
           "run_timestamp": datetime.now(timezone.utc).isoformat(), "metrics": metrics, "records": records}
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = results_dir / f"run_{mode}_{stamp}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
metrics.py

Pure functions that turn per-question results into the numbers we report.
No API calls here, so they are easy to test.
"""

import math


def key_facts_present(answer, key_facts):
    """Every entry in key_facts is a list of alternatives; at least one alternative of each must appear."""
    text = (answer or "").lower()
    return all(any(alt.lower() in text for alt in group) for group in key_facts)


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, math.ceil(pct / 100 * len(ordered)) - 1)
    return ordered[rank]


def rate(count, total):
    return {"count": count, "total": total, "rate": round(count / total, 3) if total else None}


def retrieval_rank(retrieved, gold_doc):
    """1-based rank of the first retrieved chunk from gold_doc, or None if it was not retrieved."""
    for i, item in enumerate(retrieved, start=1):
        if item["doc_id"] == gold_doc:
            return i
    return None


def summarize(records, k):
    """records: list of per-question dicts built by run_eval. Returns the metrics dict."""
    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]
    metrics = {"k": k, "questions": len(records),
               "answerable": len(answerable), "unanswerable": len(unanswerable)}

    # ---- retrieval (real in every mode)
    ranks = [retrieval_rank(r["retrieved"], r["gold_doc"]) for r in answerable]
    metrics["retrieval_recall_at_k"] = rate(sum(1 for x in ranks if x), len(ranks))
    metrics["retrieval_mrr"] = round(sum(1 / x for x in ranks if x) / len(ranks), 3) if ranks else None
    for style in ("direct", "paraphrased"):
        sub = [rank for r, rank in zip(answerable, ranks) if r.get("style", "direct") == style]
        if sub:
            metrics[f"retrieval_recall_at_k_{style}"] = rate(sum(1 for x in sub if x), len(sub))

    latencies = [r["retrieval_ms"] for r in records]
    metrics["retrieval_latency_ms"] = {
        "mean": round(sum(latencies) / len(latencies), 3) if latencies else None,
        "p50": round(percentile(latencies, 50), 3) if latencies else None,
        "p95": round(percentile(latencies, 95), 3) if latencies else None,
        "max": round(max(latencies), 3) if latencies else None,
    }

    ctx = [r["context_words"] for r in records]
    full = records[0]["full_context_words"] if records else 0
    mean_ctx = sum(ctx) / len(ctx) if ctx else 0
    metrics["context_reduction"] = {
        "mean_context_words": round(mean_ctx, 1),
        "full_handbook_words": full,
        "reduction": round(1 - mean_ctx / full, 3) if full else None,
        "note": "measured in words, not model tokens",
    }

    # ---- generation (only if generation was attempted)
    errors = [r for r in records if r.get("error")]
    ok = [r for r in records if r.get("answer") is not None and not r.get("error")]
    if not ok and not errors:
        return metrics

    # A question whose API call failed counts AGAINST the accuracy metrics (it is never
    # silently dropped), and is reported in generation_errors.
    ok_ids = {r["id"] for r in ok}
    metrics["generation_errors"] = len(errors)
    metrics["answer_parse_errors"] = sum(1 for r in ok if r["parse_error"])

    correct = [r for r in answerable if r["id"] in ok_ids and r["answered"]
               and key_facts_present(r["answer"], r["key_facts"])]
    metrics["answer_correctness"] = rate(len(correct), len(answerable))
    metrics["false_abstain_rate"] = rate(
        sum(1 for r in answerable if r["id"] in ok_ids and r["answered"] is False), len(answerable))
    metrics["abstention_accuracy"] = rate(
        sum(1 for r in unanswerable if r["id"] in ok_ids and r["answered"] is False), len(unanswerable))

    cited = [r for r in answerable if r["id"] in ok_ids and r["answered"]]
    good_citation = [r for r in cited if r["citations_valid"] and r["cites_gold_doc"]]
    metrics["citation_accuracy"] = rate(len(good_citation), len(cited))

    judged = [r for r in ok if r.get("judge_score") is not None]
    if judged:
        metrics["faithfulness_rate"] = rate(sum(1 for r in judged if r["judge_score"] >= 4), len(judged))
        metrics["faithfulness_avg_score"] = round(sum(r["judge_score"] for r in judged) / len(judged), 2)

    in_tok = sum(r["usage"]["input_tokens"] + r.get("judge_usage", {}).get("input_tokens", 0) for r in ok)
    out_tok = sum(r["usage"]["output_tokens"] + r.get("judge_usage", {}).get("output_tokens", 0) for r in ok)
    metrics["tokens"] = {"input": in_tok, "output": out_tok}
    return metrics

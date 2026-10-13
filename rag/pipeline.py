"""
pipeline.py

Ties the pieces together: chunk the handbook, retrieve the top-k chunks for a
question, and (optionally) generate an answer from them.

Modes
  retrieval : retrieval only. No API, no generation. Numbers are REAL.
  mock      : canned offline generation and judging. Tests the code, says nothing about Claude.
  live      : real Claude calls for generation (and judging in run_eval.py).
"""

from rag.chunker import build_chunks, word_count
from rag.generator import MODEL, generate, mock_generate
from rag.retriever import BM25Retriever

MODES = ("retrieval", "mock", "live")


class RagPipeline:
    def __init__(self, docs_dir="data/docs", k=3, mode="retrieval", client=None, model=MODEL):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "live" and client is None:
            raise ValueError("live mode needs an Anthropic client")
        self.k, self.mode, self.client, self.model = k, mode, client, model
        self.docs, self.chunks = build_chunks(docs_dir)
        self.retriever = BM25Retriever(self.chunks)
        # What you would have to send without retrieval: the whole handbook.
        self.full_context_words = sum(word_count(c["text"]) for c in self.chunks)

    def ask(self, question):
        results, retrieval_ms = self.retriever.search(question, self.k)
        chunks = [c for c, _ in results]
        out = {
            "question": question,
            "retrieved": [{"chunk_id": c["chunk_id"], "doc_id": c["doc_id"], "score": round(s, 3)}
                          for c, s in results],
            "retrieval_ms": retrieval_ms,
            "context_words": sum(word_count(c["text"]) for c in chunks),
            "full_context_words": self.full_context_words,
            "generation": None,
        }
        if self.mode == "retrieval":
            return out, chunks
        if self.mode == "mock":
            parsed, raw, latency, usage = mock_generate(question, chunks)
        else:
            parsed, raw, latency, usage = generate(self.client, question, chunks, self.model)
        out["generation"] = {"parsed": parsed, "raw": raw, "latency_seconds": round(latency, 3), "usage": usage}
        return out, chunks

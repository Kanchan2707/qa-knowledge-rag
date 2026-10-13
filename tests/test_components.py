"""Unit tests for chunking, retrieval, parsing and metrics (no API, no network).

Run with:  python -m unittest discover -s tests -t . -v
"""

import unittest
from pathlib import Path

from rag.chunker import build_chunks, chunk_doc, word_count
from rag.generator import build_user_prompt, parse_answer
from rag.judge import _parse as parse_judge
from rag.metrics import key_facts_present, percentile, retrieval_rank, summarize
from rag.retriever import BM25Retriever, tokenize

DOCS = Path(__file__).resolve().parent.parent / "data" / "docs"


class ChunkerTests(unittest.TestCase):
    def test_every_document_becomes_chunks_with_ids(self):
        docs, chunks = build_chunks(DOCS)
        self.assertEqual(len(docs), 14)
        self.assertGreater(len(chunks), len(docs))
        self.assertEqual(len({c["chunk_id"] for c in chunks}), len(chunks))
        for c in chunks:
            self.assertTrue(c["chunk_id"].startswith(c["doc_id"] + "#"))
            self.assertIn(c["title"], c["text"])

    def test_long_sections_are_split(self):
        doc = {"doc_id": "d", "title": "T",
               "text": "# T\n\n## S\n" + " ".join(f"Sentence number {i} is here." for i in range(60))}
        chunks = chunk_doc(doc, max_words=40)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(word_count(c["text"]) < 70 for c in chunks))

    def test_missing_folder_is_a_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            build_chunks(DOCS / "does-not-exist")


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        _, self.chunks = build_chunks(DOCS)
        self.retriever = BM25Retriever(self.chunks)

    def test_tokenize_drops_stopwords_and_plurals(self):
        self.assertEqual(tokenize("The tests are flaky"), ["test", "flaky"])
        self.assertIn("p1", tokenize("How fast is a P1 bug fixed?"))

    def test_finds_obvious_answer_first(self):
        results, _ = self.retriever.search("Which tool is used for load testing k6?", k=3)
        self.assertEqual(results[0][0]["doc_id"], "performance-testing")

    def test_scores_are_sorted_and_positive(self):
        results, _ = self.retriever.search("quarantine flaky test", k=5)
        scores = [s for _, s in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertTrue(all(s > 0 for s in scores))

    def test_no_matching_words_returns_nothing(self):
        results, latency = self.retriever.search("zzzz qqqq", k=3)
        self.assertEqual(results, [])
        self.assertGreaterEqual(latency, 0)

    def test_k_limits_results(self):
        self.assertLessEqual(len(self.retriever.search("test", k=2)[0]), 2)


class ParsingTests(unittest.TestCase):
    def test_parses_plain_json(self):
        p = parse_answer('{"answerable": true, "answer": "4 hours", "sources": ["a#1"]}')
        self.assertEqual((p["answerable"], p["answer"], p["sources"], p["parse_error"]),
                         (True, "4 hours", ["a#1"], False))

    def test_parses_code_fenced_json_with_text_around_it(self):
        p = parse_answer('Here you go:\n```json\n{"answerable": false, "answer": "n/a", "sources": []}\n```')
        self.assertFalse(p["parse_error"])
        self.assertIs(p["answerable"], False)

    def test_garbage_is_flagged_not_crashed(self):
        for bad in ["not json at all", "", None, '{"answer": "missing flag"}', '{"answerable": "yes"}']:
            with self.subTest(bad=bad):
                self.assertTrue(parse_answer(bad)["parse_error"])

    def test_judge_parser_rejects_out_of_range_scores(self):
        self.assertEqual(parse_judge('{"score": 5, "reason": "ok"}')[0], 5)
        self.assertEqual(parse_judge('{"score": 9, "reason": "x"}')[0], 0)
        self.assertEqual(parse_judge("nonsense")[0], 0)

    def test_prompt_lists_passages_with_ids(self):
        prompt = build_user_prompt("Q?", [{"chunk_id": "x#1", "text": "hello"}])
        self.assertIn("[x#1] hello", prompt)
        self.assertIn("no passages", build_user_prompt("Q?", []))


class MetricsTests(unittest.TestCase):
    def test_key_facts_need_one_alternative_from_every_group(self):
        facts = [["monday"], ["5 pm", "17:00"]]
        self.assertTrue(key_facts_present("Code freeze is Monday at 5 PM.", facts))
        self.assertFalse(key_facts_present("Code freeze is on Monday.", facts))
        self.assertFalse(key_facts_present(None, facts))

    def test_percentile_and_rank(self):
        self.assertEqual(percentile([1, 2, 3, 4, 100], 50), 3)
        self.assertEqual(percentile([1, 2, 3, 4, 100], 95), 100)
        self.assertIsNone(percentile([], 50))
        retrieved = [{"doc_id": "a"}, {"doc_id": "b"}]
        self.assertEqual(retrieval_rank(retrieved, "b"), 2)
        self.assertIsNone(retrieval_rank(retrieved, "z"))

    def test_summarize_retrieval_numbers(self):
        def rec(rid, answerable, gold, retrieved_docs, ms=1.0, words=100):
            return {"id": rid, "answerable": answerable, "gold_doc": gold, "style": "direct",
                    "retrieved": [{"doc_id": d, "chunk_id": d + "#1"} for d in retrieved_docs],
                    "retrieval_ms": ms, "context_words": words, "full_context_words": 1000, "answer": None}
        records = [rec("1", True, "a", ["a"]), rec("2", True, "b", ["x", "b"]),
                   rec("3", True, "c", ["x"]), rec("4", False, None, [])]
        m = summarize(records, k=3)
        self.assertEqual(m["retrieval_recall_at_k"]["count"], 2)
        self.assertEqual(m["retrieval_recall_at_k"]["total"], 3)
        self.assertAlmostEqual(m["retrieval_mrr"], round((1 + 0.5 + 0) / 3, 3))
        self.assertEqual(m["context_reduction"]["reduction"], 0.9)
        self.assertNotIn("answer_correctness", m)


if __name__ == "__main__":
    unittest.main()

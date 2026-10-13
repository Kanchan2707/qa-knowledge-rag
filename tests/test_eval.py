"""End-to-end tests of run_eval: retrieval-only, mock, and the live code path with a FAKE client.

The fake client stands in for the Anthropic API (the same idea as the framework's mock mode),
so no key, network or cost is involved. It checks request building, JSON handling, error
handling and metric maths, NOT how real Claude behaves.
"""

import contextlib
import io
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import run_eval
from rag.pipeline import RagPipeline

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "data" / "docs"
QUESTIONS = run_eval.load_questions(ROOT / "eval" / "questions.yaml")


def reply(text):
    return SimpleNamespace(content=[SimpleNamespace(text=text)],
                           usage=SimpleNamespace(input_tokens=100, output_tokens=20))


class FakeClient:
    """Answers from the top passage when it contains the key words, abstains otherwise."""

    def __init__(self, broken_question=None, failing_question=None, bad_judge=False):
        self.calls = []
        self.broken_question, self.failing_question, self.bad_judge = broken_question, failing_question, bad_judge
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        prompt = kwargs["messages"][0]["content"]
        if "FAITHFULNESS" in prompt:  # judge call
            return reply("garbage" if self.bad_judge else '{"score": 5, "reason": "supported"}')
        question = prompt.split("Question: ")[-1]
        if self.failing_question and self.failing_question in question:
            raise RuntimeError("simulated API outage")
        if self.broken_question and self.broken_question in question:
            return reply("I think the answer is probably 4 hours")  # not JSON
        passages = re.findall(r"\[([^\]]+)\] (.+)", prompt.split("Question:")[0])
        if not passages:
            return reply('{"answerable": false, "answer": "Not in the handbook.", "sources": []}')
        chunk_id, text = passages[0]
        return reply(json.dumps({"answerable": True, "answer": text, "sources": [chunk_id]}))


def run_main(args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = run_eval.main(args)
    return code, buf.getvalue()


class RetrievalOnlyTests(unittest.TestCase):
    def test_runs_without_any_client_and_reports_real_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = run_main(["--retrieval-only", "--results-dir", tmp])
            self.assertEqual(code, 0)
            self.assertIn("RETRIEVAL-ONLY", out)
            saved = json.loads(next(Path(tmp).glob("run_retrieval_*.json")).read_text())
            m = saved["metrics"]
            self.assertEqual(saved["mode"], "retrieval")
            self.assertEqual(m["questions"], 33)
            self.assertNotIn("answer_correctness", m)
            self.assertGreater(m["retrieval_recall_at_k"]["rate"], 0.8)
            self.assertGreater(m["context_reduction"]["reduction"], 0.5)


class MockModeTests(unittest.TestCase):
    def test_mock_run_is_labelled_and_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = run_main(["--mock", "--results-dir", tmp])
            self.assertEqual(code, 0)
            self.assertIn("NOT real results", out)
            self.assertIn("CANNED", out)
            saved = json.loads(next(Path(tmp).glob("run_mock_*.json")).read_text())
            self.assertEqual(saved["mode"], "mock")
            self.assertIn("faithfulness_rate", saved["metrics"])


class LiveCodePathTests(unittest.TestCase):
    def evaluate(self, client, questions=None):
        pipeline = RagPipeline(DOCS, k=3, mode="live", client=client)
        records = run_eval.evaluate(pipeline, questions or QUESTIONS, client)
        return records, run_eval.summarize(records, 3)

    def test_requests_are_built_correctly(self):
        client = FakeClient()
        self.evaluate(client, QUESTIONS[:1])
        gen_call, judge_call = client.calls
        self.assertEqual(gen_call["model"], "claude-sonnet-4-6")
        self.assertIn("ONLY the numbered passages", gen_call["system"])
        self.assertIn("Question: How quickly must a P1 bug be fixed?", gen_call["messages"][0]["content"])
        self.assertIn("FAITHFULNESS", judge_call["messages"][0]["content"])

    def test_full_metrics_from_a_cooperative_client(self):
        records, m = self.evaluate(FakeClient())
        self.assertEqual(m["generation_errors"], 0)
        self.assertEqual(m["answer_parse_errors"], 0)
        self.assertEqual(m["faithfulness_rate"]["count"], 33)
        self.assertGreater(m["answer_correctness"]["count"], 0)
        self.assertEqual(m["tokens"]["input"], 33 * 100 * 2)
        self.assertGreater(m["citation_accuracy"]["rate"], 0.8)

    def test_unparseable_reply_is_counted_not_fatal(self):
        records, m = self.evaluate(FakeClient(broken_question="P1 bug"))
        self.assertEqual(m["answer_parse_errors"], 1)
        bad = next(r for r in records if r["id"] == "Q01")
        self.assertIsNone(bad["answered"])
        self.assertFalse(bad["correct"])

    def test_api_error_on_one_question_does_not_lose_the_run(self):
        records, m = self.evaluate(FakeClient(failing_question="code freeze"))
        self.assertEqual(m["generation_errors"], 1)
        failed = next(r for r in records if "RuntimeError" in r.get("error", ""))
        self.assertEqual(failed["id"], "Q10")
        self.assertEqual(len(records), 33)
        # the failed question counts as wrong: denominator is still all 27 answerable questions
        self.assertEqual(m["answer_correctness"]["total"], 27)
        _, healthy = self.evaluate(FakeClient())
        self.assertLess(m["answer_correctness"]["count"], healthy["answer_correctness"]["count"])

    def test_unparseable_judge_reply_counts_as_unfaithful(self):
        _, m = self.evaluate(FakeClient(bad_judge=True))
        self.assertEqual(m["faithfulness_rate"]["count"], 0)

    def test_abstaining_on_unanswerable_questions_is_measured(self):
        unanswerable = [q for q in QUESTIONS if not q["answerable"]]
        _, m = self.evaluate(FakeClient(), unanswerable)
        self.assertEqual(m["abstention_accuracy"]["total"], 6)


class CliTests(unittest.TestCase):
    def test_live_mode_without_key_explains_what_to_do(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch("dotenv.load_dotenv"):
            code, out = run_main(["--results-dir", tempfile.gettempdir()])
        self.assertEqual(code, 1)
        self.assertIn("--retrieval-only", out)


if __name__ == "__main__":
    unittest.main()

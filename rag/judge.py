"""
judge.py

Faithfulness judge: a second Claude call scores (1-5) whether every claim in an answer
is supported by the passages the model was given. This is the same LLM-as-judge idea
used in claude-eval-framework.

A score of 0 means the judge's reply could not be parsed; it counts as NOT faithful,
so a broken judge never produces a silent pass.
"""

import json
import re

from rag.generator import MODEL
from rag.retriever import tokenize

PASS_SCORE = 4

JUDGE_PROMPT = """You are grading an assistant's answer for FAITHFULNESS to the passages it was given.

Passages:
{passages}

Question: {question}

Assistant's answer: {answer}

Score 1-5:
5 = every claim in the answer is supported by the passages (an answer that says the passages do not contain the information counts as supported when the passages really do not contain it)
4 = supported, with only trivial wording differences
3 = mostly supported but includes one unsupported detail
2 = contains a significant claim that is not in the passages
1 = mostly unsupported or contradicts the passages

Reply with only JSON: {{"score": <1-5>, "reason": "<one sentence>"}}"""


def _parse(text):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            score = int(data.get("score", 0))
            if 1 <= score <= 5:
                return score, str(data.get("reason", ""))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return 0, f"Could not parse judge reply: {cleaned[:120]!r}"


def judge_faithfulness(client, question, answer, chunks, model=MODEL):
    """Live call. Returns (score, reason, usage)."""
    passages = "\n".join(f"[{c['chunk_id']}] {c['text']}" for c in chunks) or "(none)"
    prompt = JUDGE_PROMPT.format(passages=passages, question=question, answer=answer)
    response = client.messages.create(
        model=model, max_tokens=200, messages=[{"role": "user", "content": prompt}]
    )
    score, reason = _parse(response.content[0].text)
    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    return score, reason, usage


def mock_judge(question, answer, chunks):
    """CANNED stand-in: 5 if most answer words appear in the passages, else 2."""
    context = set(tokenize(" ".join(c["text"] for c in chunks)))
    words = tokenize(answer)
    if not words:
        return 2, "Mock judge: empty answer.", {"input_tokens": 0, "output_tokens": 0}
    overlap = sum(1 for w in words if w in context) / len(words)
    score = 5 if overlap >= 0.8 else 2
    return score, f"Mock judge: {overlap:.0%} of answer words appear in the passages.", {"input_tokens": 0, "output_tokens": 0}

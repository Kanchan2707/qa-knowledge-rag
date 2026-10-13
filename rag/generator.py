"""
generator.py

Asks Claude to answer a question using ONLY the retrieved passages, and to return a
small JSON object so the answer can be checked automatically:

  {"answerable": true/false, "answer": "...", "sources": ["chunk_id", ...]}

`answerable: false` is how the model says "the passages don't contain the answer".
Also contains a canned offline stand-in (mock_generate) for testing without an API key.
"""

import json
import re
import time

from rag.retriever import tokenize

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered passages provided. "
    "If the passages do not contain the answer, set answerable to false. "
    "Never use outside knowledge. Reply with a single JSON object and nothing else:\n"
    '{"answerable": true or false, "answer": "<short answer, under 60 words>", '
    '"sources": ["<chunk id>", ...]}\n'
    "In sources, list the ids of the passages you used. Use [] if answerable is false."
)


def build_user_prompt(question, chunks):
    if chunks:
        passages = "\n".join(f"[{c['chunk_id']}] {c['text']}" for c in chunks)
    else:
        passages = "(no passages were retrieved)"
    return f"Passages:\n{passages}\n\nQuestion: {question}"


def parse_answer(text):
    """Parse the model's JSON reply. Tolerates code fences and extra text around the JSON."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict) and isinstance(data.get("answerable"), bool):
                sources = data.get("sources", [])
                return {
                    "answerable": data["answerable"],
                    "answer": str(data.get("answer", "")),
                    "sources": [str(s) for s in sources] if isinstance(sources, list) else [],
                    "parse_error": False,
                }
        except json.JSONDecodeError:
            pass
    return {"answerable": None, "answer": cleaned, "sources": [], "parse_error": True}


def generate(client, question, chunks, model=MODEL):
    """Live call. Returns (parsed_answer, raw_text, latency_seconds, usage)."""
    start = time.perf_counter()
    response = client.messages.create(
        model=model,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": build_user_prompt(question, chunks)}],
    )
    latency = time.perf_counter() - start
    raw = response.content[0].text
    usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    return parse_answer(raw), raw, latency, usage


def mock_generate(question, chunks):
    """CANNED stand-in: picks the sentence that overlaps most with the question.

    It says nothing about how Claude behaves. It exists so the pipeline and the
    metrics code can be tested offline.
    """
    if not chunks:
        parsed = {"answerable": False, "answer": "The handbook does not say.", "sources": [], "parse_error": False}
        return parsed, json.dumps(parsed), 0.0, {"input_tokens": 0, "output_tokens": 0}
    q_tokens = set(tokenize(question))
    best_sentence, best_overlap, best_chunk = "", 0, chunks[0]
    for chunk in chunks:
        body = chunk["text"].split(": ", 1)[-1]
        for sentence in re.split(r"(?<=[.!?])\s+", body):
            overlap = len(q_tokens & set(tokenize(sentence)))
            if overlap > best_overlap:
                best_sentence, best_overlap, best_chunk = sentence, overlap, chunk
    if best_overlap < 2:
        parsed = {"answerable": False, "answer": "The handbook does not say.", "sources": [], "parse_error": False}
    else:
        parsed = {"answerable": True, "answer": best_sentence, "sources": [best_chunk["chunk_id"]], "parse_error": False}
    usage = {"input_tokens": len(question.split()), "output_tokens": len(parsed["answer"].split())}
    return parsed, json.dumps(parsed), 0.0, usage

# QA Handbook RAG Assistant

A Retrieval-Augmented Generation (RAG) system that answers questions from a QA team
handbook, **with an evaluation harness that measures it**: retrieval accuracy, context
size, latency, answer correctness, abstention, citations and faithfulness.

The handbook ("Northwind QA Handbook", 14 short documents) is **fictional**, written for
this project. It contains specific, checkable facts (for example "a P1 bug must be fixed
within 4 hours"), so answers can be verified and the model cannot rely on training data.

## How it works

```
question -> BM25 retrieval (top-k chunks) -> Claude answers from those chunks only
         -> JSON {answerable, answer, sources} -> automatic checks + LLM faithfulness judge
```

| Step | File | Notes |
|---|---|---|
| Chunking | `rag/chunker.py` | One chunk per handbook section (long sections split on sentences). Title and heading stay in the chunk text. |
| Retrieval | `rag/retriever.py` | BM25 written from scratch in plain Python. Keyword-based, so it does not understand synonyms. |
| Generation | `rag/generator.py` | Claude must answer using only the passages, cite chunk ids, and set `answerable: false` when the passages don't contain the answer. |
| Judging | `rag/judge.py` | A second Claude call scores faithfulness 1-5 (pass = 4 or more). An unparseable judge reply counts as a fail. |
| Metrics | `rag/metrics.py` | Pure functions, unit tested. A failed API call counts against accuracy; it is never dropped. |

## What is measured

| Metric | Meaning |
|---|---|
| `recall@k` | For answerable questions, how often the document that holds the answer is among the top-k retrieved chunks. Reported overall and split into **direct** questions (use the handbook's own wording) and **paraphrased** ones (deliberately worded differently). |
| `MRR` | How high the right document ranks (1.0 = always first). |
| `retrieval latency` | Time to rank all chunks for a question (mean, p50, p95). |
| `context reduction` | How much smaller the context sent to the model is than sending the whole handbook. Measured in **words**, not model tokens. |
| `answer correctness` | The answer contains every required key fact (checked with simple text matching). |
| `abstention accuracy` | For questions the handbook cannot answer, how often the model says so instead of guessing. |
| `citation accuracy` | The cited chunks were really retrieved and include the document that holds the answer. |
| `faithfulness` | The judge finds every claim in the answer supported by the retrieved passages. |

There are 33 questions: 18 direct, 9 paraphrased, and 6 that the handbook cannot answer
(`eval/questions.yaml`).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                # then put your Anthropic API key in .env (live runs only)
```

## Run it

```bash
# Free, no API key. Real retrieval metrics.
python run_eval.py --retrieval-only

# Offline test of the whole pipeline. Canned answers: NOT real results.
python run_eval.py --mock

# Live: real Claude answers and faithfulness judging (uses a small amount of API credit).
python run_eval.py

# Ask one question and see what was retrieved.
python ask.py "How quickly must a P1 bug be fixed?"
python ask.py "How quickly must a P1 bug be fixed?" --retrieval-only
```

Each run is saved to `results/` as JSON, labelled with its mode. Mock runs are labelled
`mock` and must never be quoted as real results.

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

25 tests: chunking, BM25 ranking, JSON parsing (including garbage replies), metric maths,
and end-to-end runs in retrieval-only and mock modes. The live code path is tested with a
fake Claude client (good answers, an unparseable reply, an API outage, a broken judge), so
the tests need no API key. A test like that checks the plumbing, not how real Claude behaves.

## Results

Live run on 2026-10-13: `claude-sonnet-4-6` for answering and judging, top-3 retrieval,
33 questions (27 answerable, 6 not answerable), each run once. 0 parse errors, 0 API errors.
Raw output: `results/run_live_*.json`.

| Metric | Result |
|---|---|
| Retrieval recall@3 (answer's document is in the top 3) | 26/27 (96.3%), MRR 0.963 |
| - direct questions (handbook's own wording) | 18/18 (100%) |
| - paraphrased questions | 8/9 (88.9%) |
| Retrieval latency | mean 0.235 ms, p95 0.326 ms (33 chunks, MacBook Air) |
| Context sent to the model | 112 words vs 1,239 for the whole handbook: 90.9% smaller (words, not tokens) |
| Faithfulness (judge score 4 or more) | 32/33 (97.0%), average 4.91/5 |
| Citation accuracy | 23/23 (100%) |
| Abstains on the 6 unanswerable questions | 6/6 (100%) |
| Answer correctness (answerable questions) | 21/27 (77.8%) |
| Wrongly abstains on answerable questions | 4/27 (14.8%) |

### Error analysis: all 6 incorrect answers were on reworded questions

None of the 18 direct questions failed. The six misses:

| Question | What happened | Cause |
|---|---|---|
| P01 "...carrying the pager...respond?" | Said the handbook doesn't say | Right passage retrieved; model did not connect "pager" with "on-call" |
| P06 "...severe vulnerability?" | Said the handbook doesn't say | Model did not equate "severe" with the handbook's "critical" |
| P07 "...automate tests in the browser?" | Said the handbook doesn't say | Model did not connect "browser" with "web UI tests" |
| P09 "...people with disabilities..." | Said the handbook doesn't say | **Retrieval miss:** the accessibility document was not in the top 3 (no shared keywords), so abstaining was correct given what the model saw |
| P03 "...sometimes passes and sometimes fails..." | Answered with the flaky-test definition; omitted the quarantine rule | Right document, wrong chunk |
| P04 "Who has to give the go-ahead before we ship?" | Said the QA lead approves post-freeze fixes; missed that sign-off needs the QA lead **and** the product owner | Right document, wrong chunk |

What this shows:
- **Faithfulness is not correctness.** P04's answer is supported by the passage it was
  given, so it passed the faithfulness judge, but it answers the wrong question.
- **Recall at document level hides chunk-level misses.** P03 and P04 count as retrieval
  hits because the right *document* was retrieved, yet the right *chunk* was not. A
  chunk-level recall metric would be stricter and is a next step.
- **Keyword retrieval struggles with reworded questions** (P09), and the model stays
  cautious when wording differs (P01, P06, P07). Embedding-based or hybrid retrieval, and
  retrieving more chunks, are the natural fixes. They were not tried here, so their effect
  is unmeasured.

## Limitations

- **Tiny corpus.** 14 documents, about 1,200 words, 33 questions. Retrieval is fast and
  accurate here partly because the corpus is small. These numbers do not predict
  performance on a large knowledge base.
- **Keyword retrieval only.** BM25 cannot match a question to a passage that uses
  different words for the same idea. Embedding-based retrieval (or a hybrid) is the
  natural next step, and the `search(query, k)` interface was kept small for that reason.
- **Faithfulness is judged by Claude.** The judge is the same model family as the answerer
  and was not cross-checked against human review.
- **Context reduction is measured in words**, a proxy for tokens, against the naive option
  of sending the entire handbook.
- **Each question was run once.** Results can vary between runs, and with only 33 questions one question moves a rate by about 3 points.
- **Reworded questions are only 9 of 33.** The overall numbers are flattering because most questions reuse the handbook's own wording; the paraphrased split above is the more honest view of difficulty.
- The handbook is fictional, and this is a learning and demonstration project, not a
  deployed production system.

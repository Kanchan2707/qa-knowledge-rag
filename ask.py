"""
ask.py

Ask the handbook a question and see exactly what was retrieved and answered.

  python ask.py "How quickly must a P1 bug be fixed?"            # live (needs API key)
  python ask.py "How quickly must a P1 bug be fixed?" --retrieval-only
  python ask.py "How quickly must a P1 bug be fixed?" --mock
"""

import argparse
import os
import sys
from pathlib import Path

from rag.pipeline import RagPipeline

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ask the QA handbook a question.")
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=3)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--retrieval-only", action="store_true")
    group.add_argument("--mock", action="store_true")
    args = parser.parse_args(argv)

    mode = "retrieval" if args.retrieval_only else "mock" if args.mock else "live"
    client = None
    if mode == "live":
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("No ANTHROPIC_API_KEY found. Put it in a .env file, or add --retrieval-only or --mock.")
            return 1
        import anthropic
        client = anthropic.Anthropic()

    pipeline = RagPipeline(ROOT / "data" / "docs", k=args.k, mode=mode, client=client)
    result, chunks = pipeline.ask(args.question)

    print(f"Question: {args.question}")
    if mode == "mock":
        print("(MOCK mode: canned answer, not from Claude)")
    print(f"\nRetrieved {len(chunks)} passages in {result['retrieval_ms']:.2f} ms "
          f"({result['context_words']} words instead of {result['full_context_words']}):")
    for item, chunk in zip(result["retrieved"], chunks):
        print(f"  [{item['chunk_id']}] score {item['score']}: {chunk['text'][:110]}...")

    gen = result["generation"]
    if gen:
        parsed = gen["parsed"]
        print("\nAnswer:")
        if parsed["parse_error"]:
            print(f"  (could not parse the model reply) {parsed['answer']}")
        elif parsed["answerable"]:
            print(f"  {parsed['answer']}")
            print(f"  Sources: {', '.join(parsed['sources']) or 'none cited'}")
        else:
            print("  The handbook does not contain the answer to that question.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

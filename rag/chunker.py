"""
chunker.py

Loads the handbook documents and splits them into chunks.

Each document is a markdown file with a "# Title" and several "## Section"
headings. A section becomes one chunk (the section heading and document title
are kept in the chunk text, because they help retrieval). A section longer than
MAX_WORDS is split on sentence boundaries into smaller chunks.
"""

import re
from pathlib import Path

MAX_WORDS = 120


def load_docs(docs_dir):
    """Return [{'doc_id', 'title', 'text'}] for every .md file, sorted by name."""
    folder = Path(docs_dir)
    if not folder.is_dir():
        raise FileNotFoundError(f"Documents folder not found: {folder}")
    docs = []
    for path in sorted(folder.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
        title = match.group(1).strip() if match else path.stem
        docs.append({"doc_id": path.stem, "title": title, "text": text})
    if not docs:
        raise FileNotFoundError(f"No .md files found in {folder}")
    return docs


def _split_long(text, max_words):
    """Split text into pieces of at most ~max_words words, on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    pieces, current = [], []
    for sentence in sentences:
        words = len(sentence.split())
        if current and sum(len(s.split()) for s in current) + words > max_words:
            pieces.append(" ".join(current))
            current = []
        current.append(sentence)
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_doc(doc, max_words=MAX_WORDS):
    """Split one document into chunks, each tagged with its doc_id and a chunk id."""
    sections = re.split(r"^##\s+", doc["text"], flags=re.MULTILINE)
    chunks = []
    for section in sections[1:]:  # sections[0] is the title block before the first '##'
        heading, _, body = section.partition("\n")
        body = " ".join(body.split())
        if not body:
            continue
        for piece in _split_long(body, max_words):
            chunks.append({
                "chunk_id": f"{doc['doc_id']}#{len(chunks) + 1}",
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "heading": heading.strip(),
                "text": f"{doc['title']} - {heading.strip()}: {piece}",
            })
    return chunks


def build_chunks(docs_dir, max_words=MAX_WORDS):
    docs = load_docs(docs_dir)
    chunks = []
    for doc in docs:
        chunks.extend(chunk_doc(doc, max_words))
    return docs, chunks


def word_count(text):
    return len(text.split())

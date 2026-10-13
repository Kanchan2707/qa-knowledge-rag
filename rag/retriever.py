"""
retriever.py

BM25 retrieval written from scratch in plain Python (no extra libraries).

BM25 ranks chunks by how well their words match the question, giving more weight
to rare words and less to very long chunks. It is a lexical (keyword) method: it
does not understand synonyms the way embedding models do. The Retriever interface
(`search(query, k)`) is small, so an embedding-based retriever could be swapped in.
"""

import math
import re
import time

STOPWORDS = frozenset("""
a an and are as at be by for from has have how in is it its of on or that the this to
was what when where which who why will with does do did can could should would about
""".split())


def tokenize(text):
    tokens = []
    for token in re.findall(r"[a-z0-9_]+", text.lower()):
        if token in STOPWORDS:
            continue
        if len(token) > 4 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return tokens


class BM25Retriever:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(c["text"]) for c in chunks]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.avg_len = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self.term_freqs = []
        df = {}
        for tokens in self.doc_tokens:
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.term_freqs.append(tf)
            for t in tf:
                df[t] = df.get(t, 0) + 1
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def score(self, query_tokens, index):
        tf, length = self.term_freqs[index], self.doc_len[index]
        total = 0.0
        for t in query_tokens:
            f = tf.get(t)
            if not f:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * length / self.avg_len)
            total += self.idf[t] * f * (self.k1 + 1) / denom
        return total

    def search(self, query, k=3):
        """Return (results, latency_ms). results = [(chunk, score)] best first, score > 0 only."""
        start = time.perf_counter()
        query_tokens = tokenize(query)
        scored = [(self.score(query_tokens, i), i) for i in range(len(self.chunks))]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(key=lambda s: (-s[0], s[1]))
        results = [(self.chunks[i], score) for score, i in scored[:k]]
        latency_ms = (time.perf_counter() - start) * 1000
        return results, latency_ms

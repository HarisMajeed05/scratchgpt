"""FAISS document retrieval with stable offline feature hashes or optional local semantic embeddings."""

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from ..data.quality import read_rows
from ..safeio import atomic_write


def hash_embeddings(texts, dimension=1024):
    out = np.zeros((len(texts), dimension), dtype=np.float32)
    for i, text in enumerate(texts):
        words = re.findall(r"\w+", text.casefold())
        features = words + [a + " " + b for a, b in zip(words, words[1:])]
        for word, count in Counter(features).items():
            h = hashlib.blake2b(word.encode(), digest_size=8).digest()
            j = int.from_bytes(h[:4], "little") % dimension
            out[i, j] += (1 + math.log(count)) * (1 if h[4] % 2 else -1)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-8)


class FaissRetriever:
    def __init__(self, texts, embedding_model=None):
        import faiss

        self.encoder = None
        if embedding_model:
            from sentence_transformers import SentenceTransformer

            self.encoder = SentenceTransformer(embedding_model, device="cpu", local_files_only=True)
        vectors = self.encode(texts)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def encode(self, texts):
        if self.encoder:
            return np.asarray(self.encoder.encode(texts, normalize_embeddings=True), dtype=np.float32)
        return hash_embeddings(texts)

    def search(self, query, k):
        if self.index.ntotal == 0:
            return []
        scores, indices = self.index.search(self.encode([query]), min(k, self.index.ntotal))
        return [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if i >= 0]


class DocumentMemory:
    def __init__(self, path, embedding_model=None):
        self.path = Path(path)
        self.embedding_model = embedding_model
        self._signature, self._rows, self._index = None, [], None

    def retrieve(self, query, k=3, threshold=0.15):
        if not self.path.exists() or not query.strip() or k < 1:
            return []
        stat = self.path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if signature != self._signature:
            self._rows = list(read_rows(self.path))
            if not self._rows:
                return []
            self._index = FaissRetriever([r["text"] for r in self._rows], self.embedding_model)
            self._signature = signature
        return [dict(self._rows[i], score=score) for i, score in self._index.search(query, k) if score >= threshold]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Create chunked JSONL document memory from UTF-8 .txt/.md files")
    ap.add_argument("--input", required=True, help="file or directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-chars", type=int, default=2000)
    ap.add_argument("--overlap", type=int, default=200)
    a = ap.parse_args(argv)
    if a.chunk_chars < 100 or not 0 <= a.overlap < a.chunk_chars:
        ap.error("chunk-chars >= 100 and 0 <= overlap < chunk-chars required")
    source = Path(a.input)
    files = sorted(p for p in source.rglob("*") if p.suffix.lower() in (".txt", ".md")) if source.is_dir() else [source]
    out = Path(a.out).resolve()

    def write(f):
        for path in files:
            if path.resolve() == out:
                continue
            text = path.read_text(encoding="utf-8")
            for start in range(0, len(text), a.chunk_chars - a.overlap):
                chunk = text[start : start + a.chunk_chars].strip()
                if chunk:
                    row = {"source": str(path), "offset": start, "text": chunk}
                    f.write((json.dumps(row, ensure_ascii=False) + "\n").encode())

    atomic_write(out, write)
    print("created", out)


if __name__ == "__main__":
    main()

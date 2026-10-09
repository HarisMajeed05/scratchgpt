"""Long-term memory of your corrections: stored on disk, retrieved by plain TF-IDF (no pretrained parts)."""
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

from .. import paths
from ..safeio import atomic_write


def words(s):
    return re.findall(r"\w+", s.lower())


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    return " ".join(p.get("text", "") for p in content if p.get("type") == "text")


def last_user_text(messages) -> str:
    for m in reversed(messages):
        if m["role"] == "user":
            return text_of(m["content"])
    return ""


class CorrectionStore:
    """corrections.jsonl: one {"ts", "query", "context", "reply"} per line. It is the permanent record: `rebuild`
    can re-apply every line onto a newer base model."""

    def __init__(self, path=None):
        self.path = Path(path or paths.CORRECTIONS)

    def all(self):
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:      # a power cut mid-append can leave a partial last line
                continue
        return rows

    def __len__(self):
        return len(self.all())

    def add(self, context, reply):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": time.time(), "query": last_user_text(context), "context": context, "reply": reply}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def drop_last(self):
        rows = self.all()[:-1]
        body = "".join(json.dumps(r) + "\n" for r in rows).encode("utf-8")
        atomic_write(self.path, lambda f: f.write(body), keep_prev=False)

    def retrieve(self, query, k=2, thresh=0.3):
        """The k past corrections whose question is most similar to `query` (cosine over TF-IDF)."""
        mem = self.all()
        if not mem or not query.strip():
            return []
        docs = [Counter(words(m["query"])) for m in mem]
        df = Counter(w for d in docs for w in d)
        n = len(docs)

        def vec(c):
            return {w: cnt * (math.log((n + 1) / (df.get(w, 0) + 1)) + 1) for w, cnt in c.items()}

        def cos(a, b):
            num = sum(a[w] * b.get(w, 0) for w in a)
            den = math.sqrt(sum(x * x for x in a.values())) * math.sqrt(sum(x * x for x in b.values()))
            return num / den if den else 0.0

        q = vec(Counter(words(query)))
        scored = sorted(((cos(q, vec(d)), m) for d, m in zip(docs, mem)), key=lambda t: -t[0])
        return [m for s, m in scored[:k] if s >= thresh]

"""Auditable local data filtering, exact deduplication and prompt-grouped train/val split."""

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from ..safeio import atomic_json, atomic_write


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


def digest(text):
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def read_rows(path):
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{n}: expected JSON object")
                yield row


def messages_of(row):
    if "messages" in row:
        return row["messages"]
    return [
        {"role": "user", "content": row["prompt"]},
        {"role": "assistant", "content": row.get("response", row.get("chosen", ""))},
    ]


def rejection(text, min_chars=20, max_chars=200000):
    if not min_chars <= len(text.strip()) <= max_chars:
        return "length"
    if "\x00" in text or text.count("\ufffd") > 2:
        return "encoding"
    lines = [normalize(s) for s in text.splitlines() if s.strip()]
    if len(lines) > 8 and max(Counter(lines).values()) / len(lines) > 0.5:
        return "repetition"
    if re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text):
        return "email_pii"  # conservative; inspect rejects if emails are part of intended data
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-chars", type=int, default=20)
    ap.add_argument("--max-chars", type=int, default=1000000)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--heldout", help="JSONL evaluation prompts: exclude matching prompt groups")
    a = ap.parse_args(argv)
    if not 0 < a.val_fraction < 1:
        ap.error("val-fraction must lie between 0 and 1")
    excluded = {digest(r["prompt"]) for r in read_rows(a.heldout)} if a.heldout else set()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    counts, seen = Counter(), set()

    # Atomic streams keep memory bounded; dedup hashes remain in memory.
    def write_all(train):
        def write_val(val):
            def write_reject(reject):
                for i, row in enumerate(read_rows(a.input), 1):
                    try:
                        if "text" in row and "messages" not in row and "prompt" not in row:
                            if not isinstance(row["text"], str):
                                raise ValueError("text must be a string")
                            text, prompt = row["text"], row["text"]
                            msgs = None
                        else:
                            msgs = messages_of(row)
                        valid = msgs is None or (
                            msgs
                            and msgs[-1]["role"] == "assistant"
                            and any(m["role"] == "user" for m in msgs)
                            and all(
                                m["role"] in ("system", "user", "assistant") and isinstance(m["content"], str)
                                for m in msgs
                            )
                        )
                        if not valid:
                            raise ValueError("invalid conversation")
                        if msgs is not None:
                            text = "\n".join(m["content"] for m in msgs)
                            prompt = "\n".join(m["content"] for m in msgs if m["role"] != "assistant")
                        key = hashlib.sha256(unicodedata.normalize("NFC", text).strip().encode()).hexdigest()
                        group = digest(prompt)
                        reason = rejection(text, a.min_chars, a.max_chars)
                        if key in seen:
                            reason = "duplicate"
                        elif group in excluded:
                            reason = "heldout_overlap"
                    except (KeyError, TypeError, ValueError):
                        reason = "schema"
                    if reason:
                        counts[reason] += 1
                        reject.write((json.dumps({"line": i, "reason": reason}) + "\n").encode())
                        continue
                    seen.add(key)
                    split = "val" if int(group[:8], 16) / 2**32 < a.val_fraction else "train"
                    row = dict(row, content_hash=key, prompt_hash=group, split=split)
                    if msgs is not None:
                        row["messages"] = msgs
                    (val if split == "val" else train).write((json.dumps(row, ensure_ascii=False) + "\n").encode())
                    counts[split] += 1

            atomic_write(out / "rejected.jsonl", write_reject, keep_prev=False)

        atomic_write(out / "val.jsonl", write_val, keep_prev=False)

    atomic_write(out / "train.jsonl", write_all, keep_prev=False)
    atomic_json(out / "audit.json", dict(counts))
    print(json.dumps(dict(counts), indent=2))


if __name__ == "__main__":
    main()

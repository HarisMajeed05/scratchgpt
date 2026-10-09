"""Train the 32,768-entry byte-level BPE on a sample of the text + code mix -> artifacts/tokenizer/tokenizer.json

Run once, BEFORE preparing any data. Takes roughly an hour (it streams ~400k documents) and a few GB of RAM.
"""
import argparse

from .. import paths
from ..sources import PRETRAIN_SOURCES
from ..tokens import TEXT_VOCAB


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs", type=int, default=400_000, help="documents to sample across all text sources")
    ap.add_argument("--max-chars", type=int, default=20_000)
    a = ap.parse_args(argv)

    from datasets import load_dataset
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    srcs = {k: v for k, v in PRETRAIN_SOURCES.items() if v["kind"] == "text"}
    total_share = sum(v["share"] for v in srcs.values())

    def docs():
        for k, v in srcs.items():
            n = int(a.docs * v["share"] / total_share)
            print(f"sampling {n:,} docs from {k}", flush=True)
            ds = load_dataset(**v["kw"], split="train", streaming=True).shuffle(seed=0, buffer_size=10_000)
            for i, ex in enumerate(ds):
                if i >= n:
                    break
                yield ex[v["field"]][: a.max_chars]

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=TEXT_VOCAB, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=True)
    tok.train_from_iterator(docs(), trainer, length=a.docs)

    paths.TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    tok.save(str(paths.TOKENIZER_FILE))
    print("vocab size:", tok.get_vocab_size(), "->", paths.TOKENIZER_FILE)
    for name, s in [("python", "def add(a, b):\n    if a > b:\n        return a + b\n    return 0\n"),
                    ("c++", "#include <vector>\nint main() {\n    std::vector<int> v;\n    return 0;\n}\n"),
                    ("text", "The quick brown fox jumps over the lazy dog.")]:
        ids = tok.encode(s).ids
        assert tok.decode(ids) == s
        print(f"{name}: {len(s)} chars -> {len(ids)} tokens")


if __name__ == "__main__":
    main()

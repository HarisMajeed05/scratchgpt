"""Text tokenizer wrapper (byte-level BPE, 32,768 entries, trained by `scratchgpt train-tokenizer`)."""
from pathlib import Path

from . import paths
from .tokens import TEXT_VOCAB


class TextTokenizer:
    def __init__(self, path=None):
        self._path = path
        self._tok = None

    def _load(self):
        if self._tok is None:
            from tokenizers import Tokenizer

            p = Path(self._path or paths.TOKENIZER_FILE)
            if not p.exists():
                raise FileNotFoundError(f"{p} not found. Run: scratchgpt train-tokenizer")
            self._tok = Tokenizer.from_file(str(p))
        return self._tok

    def encode(self, text: str) -> list[int]:
        return self._load().encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self._load().encode_batch(texts)]

    def decode(self, ids) -> str:
        """Decodes text ids only; image/audio/special ids are skipped."""
        return self._load().decode([int(i) for i in ids if i < TEXT_VOCAB])


tokenizer = TextTokenizer()

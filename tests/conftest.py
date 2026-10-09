import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture(scope="session")
def tiny_tok(tmp_path_factory):
    """A small real BPE tokenizer, so chat-template tests don't need the 32k one."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    from scratchgpt.tokenizer import TextTokenizer

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=400, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    corpus = ["hello world", "def add(a, b): return a + b", "draw a cat on a mat", "what is the capital of France?"] * 30
    tok.train_from_iterator(corpus, trainer)
    p = tmp_path_factory.mktemp("tok") / "tokenizer.json"
    tok.save(str(p))
    return TextTokenizer(p)

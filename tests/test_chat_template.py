import pytest

from scratchgpt.chat_template import decode_reply, encode_conversation, encode_example, fit_conversation, image_tokens
from scratchgpt.tokens import ASSISTANT, BOI, END_TURN, EOI, IMG_BASE, IMG_TOKENS, SYSTEM, USER

CODES = list(range(IMG_TOKENS))


def test_roles_and_mask(tiny_tok):
    msgs = [{"role": "system", "content": "hello"}, {"role": "user", "content": "draw a cat"},
            {"role": "assistant", "content": "hello world"}]
    ids, mask = encode_conversation(msgs, tok=tiny_tok)
    assert len(ids) == len(mask)
    assert ids[0] == SYSTEM and USER in ids and ASSISTANT in ids and ids[-1] == END_TURN
    i = ids.index(ASSISTANT)
    assert sum(mask[:i + 1]) == 0                      # system, user and the role token are never learned
    assert all(mask[i + 1:]) and mask[-1] == 1         # assistant text and its closing <|end|> are learned


def test_generation_prompt_ends_with_assistant(tiny_tok):
    ids, mask = encode_conversation([{"role": "user", "content": "hello"}], add_generation_prompt=True, tok=tiny_tok)
    assert ids[-1] == ASSISTANT and len(ids) == len(mask) and sum(mask) == 0


def test_image_parts_round_trip(tiny_tok):
    toks = image_tokens(CODES)
    assert toks[0] == BOI and toks[-1] == EOI and len(toks) == IMG_TOKENS + 2 and toks[1] == IMG_BASE
    with pytest.raises(ValueError):
        image_tokens(CODES[:-1])
    with pytest.raises(ValueError):
        image_tokens([99999] * IMG_TOKENS)
    reply = tiny_tok.encode("hello") + toks + tiny_tok.encode(" world")
    text, images = decode_reply(reply, tiny_tok)
    assert text == "hello world" and images == [CODES]


def test_fit_conversation_cuts_on_an_assistant_turn(tiny_tok):
    turn = [{"role": "user", "content": "hello world"}, {"role": "assistant", "content": "hello world"}]
    one, _ = encode_conversation(turn, tok=tiny_tok)
    ids, mask = fit_conversation(turn * 3, max_len=len(one) * 2 - 1, tok=tiny_tok)
    assert ids == one and ids[-1] == END_TURN and len(mask) == len(ids)
    assert fit_conversation(turn, max_len=3, tok=tiny_tok) == ([], [])


def test_encode_example_trains_only_on_the_target(tiny_tok):
    ctx = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "WRONG"},
           {"role": "user", "content": "draw a cat"}]
    ids, mask = encode_example(ctx, "hello world", tiny_tok)
    body = tiny_tok.encode("hello world")
    assert len(ids) == len(mask) and sum(mask) == len(body) + 1
    assert ids[-1] == END_TURN and ids[-len(body) - 2] == ASSISTANT

from scratchgpt.inference.memory import CorrectionStore, last_user_text


def _ctx(q):
    return [{"role": "user", "content": q}]


def test_add_retrieve_and_drop(tmp_path):
    s = CorrectionStore(tmp_path / "c.jsonl")
    s.add(_ctx("what is the capital of France"), "Paris")
    s.add(_ctx("how do I reverse a list in python"), "Use my_list[::-1] or list.reverse()")
    assert len(s) == 2
    hit = s.retrieve("capital of france?")
    assert hit and hit[0]["reply"] == "Paris"
    assert s.retrieve("something entirely unrelated xyz") == []
    s.drop_last()
    assert len(s) == 1 and s.all()[0]["reply"] == "Paris"


def test_partial_last_line_from_a_power_cut_is_ignored(tmp_path):
    s = CorrectionStore(tmp_path / "c.jsonl")
    s.add(_ctx("hello"), "hi")
    with open(s.path, "a", encoding="utf-8") as f:
        f.write('{"ts": 1, "que')
    assert len(s) == 1


def test_last_user_text_handles_parts():
    msgs = [{"role": "user", "content": [{"type": "image", "codes": []}, {"type": "text", "text": "describe"}]}]
    assert last_user_text(msgs) == "describe"

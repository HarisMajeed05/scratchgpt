"""HTTP layer with a stub engine (no torch, no checkpoint)."""
import json
import threading

from fastapi.testclient import TestClient

from scratchgpt.server.app import create_app


class StubEngine:
    model_name = "stub"

    def __init__(self):
        self.lock = threading.RLock()

    def info(self):
        return {"model": "stub", "checkpoint": "none", "vision": True, "parameters": 1, "corrections": 0}

    def parse_messages(self, raw):
        if not raw or raw[-1]["role"] != "user":
            raise ValueError("the last message must be from the user")
        return raw

    def generate(self, messages, **kw):
        yield ("text", "Hel")
        yield ("text", "lo")
        yield ("image", list(range(256)))

    def image_to_data_url(self, codes):
        return "data:image/png;base64,AAAA"

    def image_to_b64(self, codes):
        return "AAAA"

    def draw(self, prompt, **kw):
        return list(range(256))


class StubLearner:
    def __init__(self):
        self.fixed, self.undone = [], 0

    def fix(self, context, correction):
        self.fixed.append((context, correction))

    def undo(self):
        self.undone += 1
        return self.undone == 1


def client(api_key=None, learner=None):
    return TestClient(create_app(StubEngine(), learner, api_key=api_key or ""))


USER = [{"role": "user", "content": "hi"}]


def test_health_and_models_and_ui():
    c = client()
    assert c.get("/health").json()["status"] == "ok"
    assert c.get("/v1/models").json()["data"][0]["id"] == "stub"
    assert "scratchgpt" in c.get("/").text


def test_api_key_is_enforced_except_for_open_paths():
    c = client(api_key="secret")
    assert c.get("/health").status_code == 200
    assert c.post("/v1/chat/completions", json={"messages": USER}).status_code == 401
    assert c.post("/v1/chat/completions", json={"messages": USER}, headers={"x-api-key": "wrong"}).status_code == 401
    assert c.post("/v1/chat/completions", json={"messages": USER}, headers={"x-api-key": "secret"}).status_code == 200
    assert c.post("/v1/chat/completions", json={"messages": USER},
                  headers={"Authorization": "Bearer secret"}).status_code == 200


def test_chat_completion_text_and_markdown_image():
    body = client().post("/v1/chat/completions", json={"messages": USER}).json()
    msg = body["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert msg["content"].startswith("Hello") and "![generated image](data:image/png;base64,AAAA)" in msg["content"]


def test_chat_completion_streaming_is_openai_sse():
    r = client().post("/v1/chat/completions", json={"messages": USER, "stream": True})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = [e[6:] for e in r.text.split("\n\n") if e.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text.startswith("Hello") and "data:image/png" in text


def test_bad_request_is_a_400_with_openai_error_shape():
    r = client().post("/v1/chat/completions", json={"messages": [{"role": "assistant", "content": "x"}]})
    assert r.status_code == 400 and "last message" in r.json()["error"]["message"]


def test_image_generation_endpoint():
    d = client().post("/v1/images/generations", json={"prompt": "a cat", "n": 2}).json()
    assert len(d["data"]) == 2 and d["data"][0]["b64_json"] == "AAAA"


def test_fix_uses_the_last_conversation_and_undo_works():
    learner = StubLearner()
    c = client(learner=learner)
    assert c.post("/api/fix", json={"correction": "x"}).status_code == 400          # nothing to correct yet
    c.post("/v1/chat/completions", json={"messages": USER})
    assert c.post("/api/fix", json={"correction": "Paris"}).status_code == 200
    assert learner.fixed == [(USER, "Paris")]
    assert c.post("/api/fix", json={"correction": "  "}).status_code == 400
    assert c.post("/api/undo").status_code == 200
    assert c.post("/api/undo").status_code == 400


def test_learning_can_be_disabled():
    c = client(learner=None)
    c.post("/v1/chat/completions", json={"messages": USER})
    assert c.post("/api/fix", json={"correction": "x"}).status_code == 503

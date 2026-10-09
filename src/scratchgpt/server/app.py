"""OpenAI-compatible HTTP API plus continual-learning endpoints.

  GET  /                        chat page (placeholder UI)
  GET  /health                  status
  GET  /v1/models
  POST /v1/chat/completions     OpenAI format: streaming (SSE), multi-turn, images in (data URLs) and out (markdown)
  POST /v1/images/generations   text -> image (b64_json)
  POST /api/fix                 teach it the right answer to the last (or a given) conversation
  POST /api/undo                undo the last fix
Auth: set API_KEY; clients send `x-api-key: <key>` or `Authorization: Bearer <key>`.

The engine and learner are injected, so this module imports no torch (and can be tested with a stub engine).
"""
import json
import os
import secrets
import time
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

STATIC = Path(__file__).parent / "static"
OPEN_PATHS = {"/", "/health"}


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    stream: bool = False
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    use_memory: bool = True


class ImageRequest(BaseModel):
    prompt: str
    n: int = 1
    response_format: str = "b64_json"


class FixRequest(BaseModel):
    correction: str
    messages: list[dict] | None = None


def _error(message, status=400, kind="invalid_request_error"):
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status)


def create_app(engine, learner=None, api_key=None) -> FastAPI:
    app = FastAPI(title="scratchgpt")
    api_key = api_key if api_key is not None else os.environ.get("API_KEY")
    state = {"last": None}                       # internal messages of the last chat request (for /api/fix)

    @app.middleware("http")
    async def check_key(request, call_next):
        if api_key and request.url.path not in OPEN_PATHS:
            sent = request.headers.get("x-api-key") or request.headers.get("authorization", "").removeprefix("Bearer ")
            if not secrets.compare_digest(sent.encode(), api_key.encode()):
                return _error("missing or wrong API key", 401, "authentication_error")
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    def home():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health():
        return {"status": "ok", **engine.info()}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": engine.model_name, "object": "model", "owned_by": "you"}]}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        try:
            with engine.lock:
                messages = engine.parse_messages(req.messages)
        except (ValueError, KeyError, TypeError) as e:
            return _error(str(e))
        except RuntimeError as e:
            return _error(str(e), 400)
        state["last"] = messages
        opts = dict(max_new=req.max_tokens, temperature=req.temperature, top_p=req.top_p, top_k=req.top_k,
                    use_memory=req.use_memory)
        cid, created = "chatcmpl-" + uuid.uuid4().hex[:24], int(time.time())

        def piece(ev):
            kind, val = ev
            if kind == "text":
                return val
            return f"\n\n![generated image]({engine.image_to_data_url(val)})\n\n"

        if req.stream:
            def sse():
                def chunk(delta, finish=None):
                    body = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": engine.model_name,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                    return "data: " + json.dumps(body) + "\n\n"

                yield chunk({"role": "assistant", "content": ""})
                with engine.lock:
                    for ev in engine.generate(messages, **opts):
                        yield chunk({"content": piece(ev)})
                yield chunk({}, "stop")
                yield "data: [DONE]\n\n"

            return StreamingResponse(sse(), media_type="text/event-stream")

        with engine.lock:
            text = "".join(piece(ev) for ev in engine.generate(messages, **opts))
        return {"id": cid, "object": "chat.completion", "created": created, "model": engine.model_name,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}

    @app.post("/v1/images/generations")
    def images(req: ImageRequest):
        data = []
        try:
            with engine.lock:
                for _ in range(max(1, min(req.n, 4))):
                    data.append({"b64_json": engine.image_to_b64(engine.draw(req.prompt))})
        except RuntimeError as e:
            return _error(str(e), 400)
        return {"created": int(time.time()), "data": data}

    @app.post("/api/fix")
    def fix(req: FixRequest):
        if learner is None:
            return _error("continual learning is not enabled on this server", 503, "server_error")
        try:
            with engine.lock:
                ctx = engine.parse_messages(req.messages) if req.messages else state["last"]
        except (ValueError, KeyError, TypeError) as e:
            return _error(str(e))
        if not ctx:
            return _error("no conversation to correct yet")
        if not req.correction.strip():
            return _error("the correction is empty")
        try:
            learner.fix(ctx, req.correction.strip())
        except Exception as e:                   # LearningError (rolled back) or anything unexpected
            return _error(str(e), 503, "server_error")
        return {"message": "learned. Ask the same thing again to check."}

    @app.post("/api/undo")
    def undo():
        if learner is None or not learner.undo():
            return _error("nothing to undo in this session")
        return {"message": "last fix undone"}

    return app

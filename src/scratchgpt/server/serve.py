"""Run the local server: `scratchgpt serve`  ->  http://127.0.0.1:8000

Listens on localhost only. To reach it from elsewhere, set API_KEY first and use a tunnel, e.g.
    $env:API_KEY = "a-long-random-string"
    scratchgpt serve
    cloudflared tunnel --url http://localhost:8000
"""
import argparse

import uvicorn

from ..inference.continual import ContinualLearner
from ..inference.engine import Engine
from .app import create_app


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--checkpoint", help="explicit floating-point or CPU INT8 checkpoint")
    ap.add_argument("--memory", help="document-memory JSONL, requires retrieval extra")
    ap.add_argument("--embedding-model", help="local sentence-transformer directory (optional)")
    ap.add_argument("--no-learning", action="store_true", help="disable /api/fix and /api/undo")
    ap.add_argument("--fix-steps", type=int, default=60, help="fine-tune steps per correction")
    a = ap.parse_args(argv)

    from ..safeio import load_torch
    checkpoint = load_torch(a.checkpoint, a.device or "cpu") if a.checkpoint else None
    if a.checkpoint and checkpoint is None:
        ap.error("explicit checkpoint not found")
    if checkpoint and checkpoint.get("quantization"):
        a.device = "cpu"
        a.no_learning = True
    memory = None
    if a.memory:
        from ..inference.retrieval import DocumentMemory
        memory = DocumentMemory(a.memory, a.embedding_model)
    engine = Engine(device=a.device, checkpoint=checkpoint, document_memory=memory)
    learner = None if a.no_learning else ContinualLearner(engine, steps=a.fix_steps)
    print(f"loaded {engine.source} checkpoint | {engine.info()['parameters'] / 1e6:.0f}M parameters | "
          f"vision: {engine.has_vision} | corrections: {len(engine.store)}")
    uvicorn.run(create_app(engine, learner), host=a.host, port=a.port)


if __name__ == "__main__":
    main()

"""Generate sequence-level distillation data using a local open-weight teacher (no remote inference)."""

import argparse
import json
from pathlib import Path

from .quality import digest, read_rows, rejection


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", required=True, help='JSONL: {"prompt": "..."}, optional kind/answer/tests metadata')
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", default="Qwen/Qwen3-0.6B", help="HF model ID or local model directory")
    ap.add_argument("--revision", default="main", help="Pin a commit for reproducibility")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--max-prompt", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--int8", action="store_true", help="CPU dynamic INT8 teacher linear layers")
    ap.add_argument("--verify", action="store_true", help="Require math answer or code tests to pass")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.max_new < 1 or a.max_prompt < 1:
        ap.error("token limits must be positive")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.teacher, revision=a.revision, local_files_only=a.offline)
    teacher = AutoModelForCausalLM.from_pretrained(
        a.teacher,
        revision=a.revision,
        local_files_only=a.offline,
        torch_dtype=torch.float32 if a.device == "cpu" else torch.bfloat16,
    )
    teacher.to(a.device).eval()
    if a.max_prompt + a.max_new > teacher.config.max_position_embeddings:
        ap.error("prompt + generation limit exceeds teacher context")
    if a.int8:
        if a.device != "cpu":
            ap.error("--int8 requires CPU")
        teacher = torch.ao.quantization.quantize_dynamic(teacher, {torch.nn.Linear}, dtype=torch.qint8)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Recover an interrupted final append; retain every complete valid record.
    if out.exists():
        data = out.read_bytes()
        if data and not data.endswith(b"\n"):
            boundary = data.rfind(b"\n") + 1
            tail = data[boundary:]
            try:
                json.loads(tail)
                data += b"\n"
            except (json.JSONDecodeError, UnicodeDecodeError):
                data = data[:boundary]
            from ..safeio import atomic_write

            atomic_write(out, lambda f: f.write(data), keep_prev=False)
    old_rows = list(read_rows(out)) if out.exists() else []
    if any(
        r.get("teacher") != a.teacher or r.get("requested_revision", r.get("revision")) != a.revision for r in old_rows
    ):
        ap.error("output contains answers from a different teacher/revision; choose a new output path")
    completed = {r["prompt_hash"] for r in old_rows}
    # One durable record per prompt, restart skips already-generated prompts.
    with out.open("a", encoding="utf-8") as f:
        for row in read_rows(a.prompts):
            key = digest(row["prompt"])
            if key in completed:
                continue
            text = tok.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            x = tok(text, return_tensors="pt", add_special_tokens=False).to(a.device)
            if x.input_ids.size(1) > a.max_prompt:
                print("skipped overlong prompt", key)
                continue
            opts = dict(
                max_new_tokens=a.max_new,
                do_sample=a.temperature > 0,
                pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
            )
            if a.temperature > 0:
                opts["temperature"] = a.temperature
            with torch.inference_mode():
                y = teacher.generate(**x, **opts)
            response = tok.decode(y[0, x.input_ids.size(1) :], skip_special_tokens=True).strip()
            if rejection(response, min_chars=1 if row.get("kind") == "math" else 20):
                print("skipped low-quality answer", key)
                continue
            if a.verify:
                from ..training.verifiers import verify

                if verify(row, response) != 1:
                    print("skipped failed verification", key)
                    continue
            rec = dict(
                row,
                response=response,
                prompt_hash=key,
                source="local_teacher",
                teacher=a.teacher,
                revision=getattr(teacher.config, "_commit_hash", None) or a.revision,
                requested_revision=a.revision,
                verified=a.verify,
                messages=[{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": response}],
            )
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            import os

            os.fsync(f.fileno())
            completed.add(key)
            print("generated", key, flush=True)


if __name__ == "__main__":
    main()

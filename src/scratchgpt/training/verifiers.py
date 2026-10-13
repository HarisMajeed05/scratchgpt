"""Binary rewards for explicit numeric final answers or trusted Python unit tests."""

import math
import re

from ..inference.tools import run_python


def numeric_answer(text):
    # Prefer explicit final-answer markers; otherwise require the final line to be numeric.
    boxes = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxes:
        candidate = boxes[-1].strip()
    else:
        candidate = text.strip().splitlines()[-1] if text.strip() else ""
        candidate = re.sub(r"^(?:final answer|answer)\s*[:=]\s*", "", candidate, flags=re.I).strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", candidate):
        return None
    value = float(candidate)
    return value if math.isfinite(value) else None


def verify(row, response):
    kind = row.get("kind", "math")
    if kind == "math":
        if "answer" not in row:
            raise ValueError("math verification needs a trusted numeric answer")
        actual = numeric_answer(response)
        expected = float(row["answer"])
        return float(
            actual is not None
            and math.isfinite(expected)
            and math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6)
        )
    if kind == "code":
        if not row.get("tests"):
            raise ValueError("code verification requires trusted tests")
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", response, re.S)
        code = blocks[-1] if blocks else response
        import secrets

        marker = secrets.token_hex(16)
        harness = code + "\n" + row["tests"] + "\nprint(" + repr(marker) + ")\n"
        result = run_python(harness)
        if result.get("returncode") == 125:
            raise RuntimeError("Docker unavailable or local Python image missing; fix infrastructure before RL")
        return float(result["ok"] and marker in result["output"])
    raise ValueError("kind must be math or code")

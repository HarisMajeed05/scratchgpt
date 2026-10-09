"""Chat in the terminal. Commands: /fix <correct answer>  /undo  /clear  /draw <prompt>  /quit"""
import argparse

from . import paths
from .inference.continual import ContinualLearner, LearningError
from .inference.engine import Engine
from .vision.imageio import png_bytes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    eng = Engine(device=a.device)
    learner = ContinualLearner(eng)
    print(f"model: {eng.source} checkpoint | vision: {eng.has_vision} | corrections so far: {len(eng.store)}")
    print(__doc__.strip().splitlines()[-1])
    history, context, n_img = [], None, 0
    while True:
        try:
            s = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not s:
            continue
        if s in ("/quit", "/exit"):
            break
        if s == "/clear":
            history, context = [], None
            print("(conversation cleared)")
            continue
        if s == "/undo":
            print("undone" if learner.undo() else "nothing to undo in this session")
            continue
        if s.startswith("/fix "):
            if context is None:
                print("nothing to fix yet")
                continue
            try:
                learner.fix(context, s[5:].strip())
                history[-1] = {"role": "assistant", "content": s[5:].strip()}
                print("learned. Ask again to check.")
            except LearningError as e:
                print(e)
            continue
        if s.startswith("/draw "):
            try:
                with eng.lock:
                    arr = eng.image_array(eng.draw(s[6:].strip()))
                paths.OUTPUTS.mkdir(parents=True, exist_ok=True)
                n_img += 1
                p = paths.OUTPUTS / f"draw_{n_img}.png"
                p.write_bytes(png_bytes(arr))
                print(f"saved {p}")
            except RuntimeError as e:
                print(e)
            continue
        history.append({"role": "user", "content": s})
        with eng.lock:
            context = eng.parse_messages(history)      # internal format; restores generated images exactly
        reply = ""
        print("model> ", end="", flush=True)
        with eng.lock:
            for kind, val in eng.generate(context, max_new=a.max_tokens, temperature=a.temperature):
                if kind == "text":
                    reply += val
                    print(val, end="", flush=True)
                else:
                    n_img += 1
                    paths.OUTPUTS.mkdir(parents=True, exist_ok=True)
                    p = paths.OUTPUTS / f"image_{n_img}.png"
                    p.write_bytes(png_bytes(eng.image_array(val)))
                    reply += f"\n![generated image]({eng.image_to_data_url(val)})\n"
                    print(f"\n[image saved to {p}]", end="", flush=True)
        print()
        history.append({"role": "assistant", "content": reply.strip()})


if __name__ == "__main__":
    main()

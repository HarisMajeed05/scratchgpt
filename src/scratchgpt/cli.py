"""`scratchgpt <command>` (or `python -m scratchgpt <command>`): one entry point for the whole pipeline."""
import importlib
import sys

COMMANDS = {
    "train-tokenizer": ("scratchgpt.data.train_tokenizer", "1. train the 32k BPE text tokenizer"),
    "prepare-pretrain": ("scratchgpt.data.prepare_pretrain", "2. stream + tokenize the pretraining mix into shards"),
    "vq-cache": ("scratchgpt.training.train_vqvae", "3a. cache images for the image tokenizer (use: vq-cache)"),
    "vq-train": ("scratchgpt.training.train_vqvae", "3b. train the image tokenizer (VQ-VAE)"),
    "pretrain": ("scratchgpt.training.pretrain", "4. pretrain the language model"),
    "prepare-sft": ("scratchgpt.data.prepare_sft", "5. build chat / image-chat instruction data"),
    "sft": ("scratchgpt.training.sft", "6. instruction-tune into a chat model"),
    "serve": ("scratchgpt.server.serve", "7. run the OpenAI-compatible server (with continual learning)"),
    "chat": ("scratchgpt.chat_cli", "   chat in the terminal (with /fix, /undo, /draw)"),
    "rebuild": ("scratchgpt.inference.continual", "   re-apply all saved corrections onto the current base model"),
}


def _usage():
    print("usage: scratchgpt <command> [options]\n\ncommands:")
    for name, (_, desc) in COMMANDS.items():
        print(f"  {name:<17} {desc}")
    print("\nRun `scratchgpt <command> --help` for a command's options.")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        _usage()
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in COMMANDS:
        print(f"unknown command: {cmd}\n")
        _usage()
        return 2
    module = importlib.import_module(COMMANDS[cmd][0])
    if cmd in ("vq-cache", "vq-train"):          # one module, two sub-commands
        return module.main([cmd.split("-")[1]] + rest)
    return module.main(rest)

# Roadmap

Done: tokenizer, VQ-VAE, crash-safe data pipeline, pretraining, chat SFT, image in/out, continual learning, OpenAI-compatible
server, tests + CI.

Next, in rough order:

1. **ChatGPT-style web UI**: conversation list, streaming, markdown + code blocks, image upload, per-message "Fix" button.
   The server already speaks the OpenAI protocol, so any OpenAI-compatible UI works meanwhile.
2. **Audio**: train a small neural audio codec (IDs 40960-45055 are reserved), add speech-to-text (audio -> text) and
   text-to-speech (text -> audio) data, extend the chat template with audio parts.
3. **Better images**: add a patch-GAN discriminator to the VQ-VAE for sharper reconstructions; 256px with 1024 codes
   (needs a longer context).
4. **Implemented in v0.2:** DPO, local teacher distillation, FAISS retrieval, tools, verified-reward REINFORCE and CPU INT8. Next: benchmark quality and replace the deprecated PTQ backend.
5. **Scale**: a 350M variant (`GPTConfig(n_layer=24, n_head=16, n_embd=1024)`) once the modern default pipeline is benchmarked.
6. **Publishing**: weights on the Hugging Face Hub, plus a read-only Space for demos.

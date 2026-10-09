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
4. **Preference tuning** (DPO) on top of SFT.
5. **Scale**: a 350M variant (`GPTConfig(n_layer=24, n_head=16, n_embd=1024)`) once the 124M pipeline is proven.
6. **Publishing**: weights on the Hugging Face Hub, plus a read-only Space for demos.

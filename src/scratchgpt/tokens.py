"""Token ID layout. Everything fits in uint16. Never renumber: shards on disk depend on it.

    0      .. 32767   text (byte-level BPE)
    32768  .. 40959   image codes (VQ-VAE codebook, 8192 entries)
    40960  .. 45055   audio codes (reserved for the audio codec)
    45056  .. 45119   special tokens
"""
TEXT_VOCAB = 32768
IMG_BASE, IMG_CODES = 32768, 8192
AUD_BASE, AUD_CODES = 40960, 4096
SPECIAL_BASE = 45056

# document / media markers
EOT, BOI, EOI, BOA, EOA, PAD = range(SPECIAL_BASE, SPECIAL_BASE + 6)
# chat markers
SYSTEM, USER, ASSISTANT, END_TURN = range(SPECIAL_BASE + 6, SPECIAL_BASE + 10)

VOCAB_SIZE = 45120                         # multiple of 64; = GPTConfig.vocab_size

IMG_SIZE, IMG_DOWN = 128, 8                # 128px, 8x downsample -> 16x16 = 256 tokens per image
IMG_TOKENS = (IMG_SIZE // IMG_DOWN) ** 2

SPECIAL_NAMES = {
    EOT: "<|endoftext|>", BOI: "<|img|>", EOI: "<|/img|>", BOA: "<|aud|>", EOA: "<|/aud|>", PAD: "<|pad|>",
    SYSTEM: "<|system|>", USER: "<|user|>", ASSISTANT: "<|assistant|>", END_TURN: "<|end|>",
}

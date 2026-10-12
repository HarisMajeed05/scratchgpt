# Pipeline, step by step

All outputs go under `artifacts/` (set `SCRATCHGPT_HOME` to use another drive). Every long step is **resumable and
crash-safe**: Ctrl+C saves, and a power cut costs at most one shard (about 100M tokens) of data prep or `--save-every`
steps of training. Just run the same command again.

| # | Command | Writes | Time / space (rough) |
|---|---|---|---|
| 1 | `scratchgpt train-tokenizer` | `artifacts/tokenizer/tokenizer.json` | ~1 h, a few GB RAM |
| 2 | `scratchgpt prepare-pretrain --until 90e9` | `artifacts/data/pretrain/*.bin`, `state.json` | days (network bound); 2 bytes/token, 90B tokens = 180 GB |
| 3a | `scratchgpt vq-cache --n 200000` | `artifacts/data/vq_images/images.npy` | ~30 min, ~10 GB (deletable afterwards) |
| 3b | `scratchgpt vq-train --steps 60000` | `artifacts/checkpoints/vq/vq.pt`, `recon_*.png` | hours |
| 2' | `scratchgpt prepare-pretrain` (no flag) | adds image-caption pairs | runs to 110B tokens |
| 4 | `scratchgpt pretrain` | `artifacts/checkpoints/lm/latest.pt` | weeks; ~2-4B tokens/day on an RTX 5070 (estimate) |
| 5 | `scratchgpt prepare-sft` | `artifacts/data/sft/*.npz` | ~1 h |
| 6 | `scratchgpt sft` | `artifacts/checkpoints/sft/latest.pt` | hours |
| 7 | `scratchgpt serve` / `scratchgpt chat` | `artifacts/live/` (live model + corrections) | |

## Rules that matter

1. **Train the tokenizer first.** Token IDs are baked into every shard; changing the tokenizer means redoing the data.
2. **Freeze the VQ-VAE before writing image tokens.** Image codes from different VQ-VAE versions are incompatible. The
   data preparation records the VQ-VAE step and refuses to continue if it changes.
3. **Don't start `pretrain` with only a few shards** unless you want to overfit them: it needs one `val_*` and one
   `train_*` shard and picks up new shards as they appear, weighted by size.
4. **Learning while pretraining runs:** `/fix` needs a few GB of VRAM. If pretraining is using nearly all of it, pause
   pretraining first; a failed fix is rolled back automatically.
5. **Finishing pretraining:** add `--decay-start <current step>` to start the final learning-rate decay.

## Changing the data mix

Edit the `share` values in `src/scratchgpt/sources.py` (they must sum to 1.0). The scheduler always pulls from whichever
source is furthest behind its share, so changes apply to data written from then on. When a source runs out (e.g. the
image-caption sets), the others fill in.

## Disk budget example (300 GB)

180 GB pretraining shards + ~10 GB VQ cache (deletable) + ~1 GB per checkpoint (latest + .prev) + ~5 GB SFT data.

## Why image data is slower, and the status line

Each source is downloaded, decoded and tokenized by its own background thread, a few batches ahead. The main loop takes
from the most-behind source that has a batch **ready**, so a slow source never stalls the fast ones. Images cost roughly
100x more bytes per token than text (a whole photo for ~290 tokens), so on a 20 Mbps link expect on the order of
1B image tokens per day while text keeps flowing at the same time. The image share you configured is a target, not a
guarantee: the real share is whatever the bandwidth allows.

Once a minute `prepare-pretrain` prints a status line, for example:

```
status: 310k tok/s | web:240k py:40k cpp:20k cc12m:3k | sources ready 9/11
```

How to read it: `cc12m` stuck at a few k tok/s with the network near 20 Mbps (Task Manager, Performance) means you are
bandwidth-bound; low network use with a busy CPU means tokenizing is the limit; `sources ready` near 0 means everything
is waiting on downloads. Prefetched batches that were not yet written are simply re-read after a restart (progress is
saved per source only when a batch is consumed), so crashes still lose and repeat nothing.

## Giving images the whole connection: `--only` and `--add`

Text and images share your download bandwidth, and text will happily use almost all of it. To let images catch up, run
them on their own for a while, then go back to the full mix:

```powershell
scratchgpt prepare-pretrain --only images --add 2e9    # write 2B NEW image tokens, nothing else, then stop
scratchgpt prepare-pretrain                            # back to the normal mix (resumes where it stopped)
```

`--only` accepts `images`, `text`, or source names (`--only cc12m,cc3m`). `--add N` stops after N new tokens in this run;
without it `--until` is a cumulative total as before. Everything stays resumable and crash-safe.

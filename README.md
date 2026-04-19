# PS6-CHEME-5820-S2026

Problem Set 6 (final PS of the semester): Transformers — tokenization, scale, and interpretability. Picks up the NanoGPT code from Lab 13d.

## Tasks

1. **Tokenizer swap** — retrain NanoGPT on Tiny Shakespeare with the pretrained GPT-2 BPE tokenizer and measure the change in perplexity and effective sequence length.
2. **Scale** — train a tiny NanoGPT on [TinyStories](https://arxiv.org/abs/2305.07759) and compare its generated samples to a larger reference checkpoint trained by the instructor.
3. **Induction-head hunt** — implement an induction-score scan across `(layer, head)` pairs of the reference checkpoint, rank the heads, and visualize the top head's attention pattern.

## Repo layout

```
Project.toml                 Julia dependencies (NanoGPT + BytePairEncoding)
Include.jl                   environment bootstrap (paths, packages, src includes)
src/
  CausalAttention.jl         \
  DecoderBlock.jl             >  copied verbatim from L13d
  NanoGPT.jl                 /
  Sample.jl                  /
  Shakespeare.jl             /
  Compute.jl                 PS6 helpers (BPE, training, induction scan primitives)
  Types.jl                   PS6-specific types (TrainingConfig)
  Autograder.jl              rubric-style check!/score! framework (from PS5)
scripts/
  train_reference.jl         instructor-run training for data/reference_checkpoint.jld2
data/                        corpora, token caches, model checkpoints (gitignored)
PS6-CHEME-5820-Student-Transformers-S2026.ipynb
PS6-CHEME-5820-Solution-Transformers-S2026.ipynb
```

## Running

### Students

Install dependencies once, then open the student notebook:

```
julia --project=. -e 'using Pkg; Pkg.instantiate()'
jupyter notebook PS6-CHEME-5820-Student-Transformers-S2026.ipynb
```

Task 2 and Task 3 require `data/reference_checkpoint.jld2`. The notebook fetches it automatically on first run by calling [`fetch_reference_checkpoint`](src/Compute.jl), which downloads the file from the course **Releases** page into your `data/` folder. If the automatic fetch fails (e.g., offline), grab the file manually from the Releases page and drop it into `data/` yourself.

### Submission: do not commit anything in `data/`

The `data/` folder holds downloads (~2 GB), token caches (~400 MB), and trained checkpoints (~110 MB each). GitHub rejects any single push containing a file over 100 MB, so if you accidentally try to commit something in `data/` your `git push` will fail.

The repo's `.gitignore` already blocks `data/`, plus `*.jld2` and `*.log` anywhere in the repo, so this should never happen. If `git status` ever shows anything in `data/` as tracked or staged, unstage it with `git restore --staged data/` before you commit.

Only commit the notebook, your edits to `src/` (if any), and the answer cells. Leave `data/` alone.

### Instructor

Train the reference checkpoint once per semester:

```
julia --project=. --threads=auto scripts/train_reference.jl
```

The script downloads TinyStories, trains a 4-layer / d=256 NanoGPT for 15 000 steps, saves the checkpoint to `data/reference_checkpoint.jld2`, and prints an induction-head scan. The checkpoint ships with the repo for student runs.

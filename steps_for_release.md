# Shipping the Reference Checkpoint via GitHub Release

Once training finishes, follow these steps to publish `data/reference_checkpoint.jld2` so students can download it automatically from inside the notebook.

## 0. Prerequisites

- Training has completed cleanly (`scripts/train_reference.jl` printed the top induction score and wrote `data/reference_checkpoint.jld2`), or you have promoted a progress checkpoint with `scripts/finalize_reference.jl`.
- You have verified quality locally by running the Solution notebook end to end (samples look reasonable, the induction scan runs and produces a heatmap). Task 3 no longer gates on a minimum induction score.
- Your code is committed and pushed. `data/` is gitignored, so the checkpoint is **not** in git — that is correct.

## 1. Create the release on GitHub

1. Open your repo on [github.com](https://github.com).
2. In the right sidebar, click **Releases** (or go directly to `https://github.com/<owner>/<repo>/releases`).
3. Click **Draft a new release**.
4. Fill in the form:
   - **Choose a tag**: type a new tag name such as `v1.0-s2026`. GitHub offers to create it on the latest commit; accept.
   - **Target**: leave at `main`.
   - **Release title**: `PS6 Spring 2026 — reference checkpoint` (or similar).
   - **Describe this release**: one or two sentences, e.g., *"Contains `reference_checkpoint.jld2` (~110 MB). Downloaded automatically by the notebook via `fetch_reference_checkpoint`."*
5. Near the bottom, drag `data/reference_checkpoint.jld2` from Finder into the **"Attach binaries by dropping them here or selecting them"** box. Wait for the upload to finish (progress bar).
6. Click **Publish release** (green button at the bottom).

## 2. Copy the asset URL

After publishing, the release page shows the attached file. Right-click the filename and copy the link address. It will look like:

```
https://github.com/<owner>/<repo>/releases/download/<tag>/reference_checkpoint.jld2
```

Example: `https://github.com/varnerlab/PS6-CHEME-5820-S2026/releases/download/v1.0-s2026/reference_checkpoint.jld2`

## 3. Update the URL constant in the code

Open [`src/Compute.jl`](src/Compute.jl) and find the constant near the bottom:

```julia
const REFERENCE_CHECKPOINT_URL = "https://github.com/varnerlab/PS6-CHEME-5820-S2026/releases/download/v1.0-s2026/reference_checkpoint.jld2"
```

Replace the string with the actual URL you just copied. Commit and push the change.

## 4. Verify the end-to-end flow

On your local machine (or a fresh clone), delete the local copy of the checkpoint and run the helper from Julia:

```
rm data/reference_checkpoint.jld2
julia --project=. -e 'include("Include.jl"); fetch_reference_checkpoint()'
```

Expected output:

```
[ Info: Downloading reference checkpoint... url=... target=data/reference_checkpoint.jld2
[ Info: Download complete path=data/reference_checkpoint.jld2 size_mb=... elapsed_s=...
```

If that works, a student running the notebook for the first time will get the same automatic download when they reach Task 2c.

## Gotchas

- **Template repo must be public** for GitHub Classroom students to download the release asset without authentication. Your PS3/PS4/PS5 repos were public, so following that pattern works.
- **Tags are unique per repo**. If you retrain later and want to ship an updated checkpoint, use a new tag (`v1.1-s2026` etc.). Do not overwrite a tag students may already have fetched from.
- **Do not commit the checkpoint to git.** The `.gitignore` blocks `data/` and `*.jld2`, but if you ever find yourself fighting a push error about file size, check `git status` and run `git restore --staged data/` to back out.
- **GitHub Classroom does not copy releases to student forks.** Students fetch from *your* template repo's release URL, not their own. The `fetch_reference_checkpoint` helper uses the absolute URL, so this works automatically.

## One-time setup, zero recurring work

After you publish the release and push the URL update once, every student running the notebook gets the checkpoint automatically. You do not need to re-upload or re-distribute the file for subsequent semesters unless the training recipe changes.

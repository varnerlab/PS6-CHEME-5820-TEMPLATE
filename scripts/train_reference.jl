# train_reference.jl -- instructor training script for the PS6 reference checkpoint.
#
# Run from the repository root:
#
#     julia --project=. --threads=auto scripts/train_reference.jl
#
# The script
#   1. downloads a TinyStories split to `data/`
#   2. tokenizes it with the GPT-2 BPE (caching the encoded vector)
#   3. trains a 4-layer NanoGPT on the encoded corpus
#   4. saves the checkpoint to `data/reference_checkpoint.jld2`
#   5. prints an induction-head score heatmap so we can verify the checkpoint
#      is usable for Task 3 before shipping it to students.
#
# Budget roughly a few hours on an M-series Mac with 20 cores. Adjust the
# hyperparameters in `REF_CFG` below to trade training time for model quality.

# ─── environment ────────────────────────────────────────────────────────────
const _SCRIPT_DIR = @__DIR__
const _REPO_ROOT = abspath(joinpath(_SCRIPT_DIR, ".."))
cd(_REPO_ROOT)
include(joinpath(_REPO_ROOT, "Include.jl"))

# ─── reference-model configuration ───────────────────────────────────────────
# Embedding tables dominate the param count under the GPT-2 vocab (50257), so
# the transformer body stays relatively small. This sizing is chosen to train
# in a few hours on an M-series Mac while still producing clean induction
# heads in at least one `(layer, head)` pair.
const REF_CFG = TrainingConfig(
    50257,      # vocab_size  (GPT-2 BPE)
    256,        # d_model
    4,          # n_heads
    4,          # n_layers
    256,        # ctx_len
    1024,       # d_ff (= 4 * d_model)
    32,         # batch_size
    6f-4,       # learning rate (peak, cosine schedule)
    15_000,     # training steps
)

# Cap the TinyStories text at this many characters before tokenization to keep
# training memory bounded. ~200 MB of text tokenizes to ~50 M tokens, which
# lets the training loop sample many diverse windows without holding the full
# 2 GB file in memory.
const MAX_TRAIN_CHARS = 200_000_000
const MAX_VAL_CHARS   =   5_000_000

const CHECKPOINT_PATH   = joinpath(_PATH_TO_DATA, "reference_checkpoint.jld2")
const PROGRESS_PATH     = joinpath(_PATH_TO_DATA, "reference_checkpoint_progress.jld2")
const TRAIN_TOKENS_PATH = joinpath(_PATH_TO_DATA, "tinystories_train_tokens.jld2")
const VAL_TOKENS_PATH   = joinpath(_PATH_TO_DATA, "tinystories_val_tokens.jld2")

# ─── induction-head scan (reference implementation for the sanity check) ────
# Students implement this themselves in the PS6 notebook; the script needs its
# own copy so the checkpoint can be gated on the Task 3 threshold before it is
# shipped.
function induction_score(model::NanoGPT, vocab_size::Int;
                          n_trials::Int = 32, seq_len::Int = 32,
                          rng::AbstractRNG = Random.default_rng())::Matrix{Float32}
    n_layers = length(model.blocks)
    n_heads = model.blocks[1].attn.n_heads
    scores = zeros(Float32, n_layers, n_heads)
    for _ in 1:n_trials
        tokens = rand(rng, 1:vocab_size, seq_len)
        seq = vcat(tokens, tokens)
        X = reshape(seq, 2 * seq_len, 1)
        ws = collect_attention_weights(model, X)
        for l in 1:n_layers, h in 1:n_heads
            acc = 0.0f0
            for i in 1:(seq_len - 1)
                acc += ws[l][seq_len + i, i + 1, h, 1]
            end
            scores[l, h] += acc / (seq_len - 1)
        end
    end
    return scores ./ n_trials
end

# ─── data pipeline ──────────────────────────────────────────────────────────
function prepare_tokens(enc, split::String, cache_path::String, max_chars::Int)
    if isfile(cache_path)
        @info "Loading cached token stream" split=split path=cache_path
        return load(cache_path, "ids")::Vector{Int}
    end

    text_path = download_tinystories(_PATH_TO_DATA; split = split)
    @info "Tokenizing TinyStories split" split=split text=text_path max_chars=max_chars
    t0 = time()
    ids = load_and_encode(text_path, enc; max_chars = max_chars)
    @info "Tokenization done" n_tokens=length(ids) elapsed_s=round(time() - t0, digits=1)

    jldsave(cache_path; ids = ids)
    @info "Cached token stream" path=cache_path
    return ids
end

# ─── main ──────────────────────────────────────────────────────────────────
function main()
    @info "Loading GPT-2 tokenizer..."
    enc = load_gpt2_encoder()
    @assert vocab_size(enc) == REF_CFG.vocab_size

    train_ids = prepare_tokens(enc, "train", TRAIN_TOKENS_PATH, MAX_TRAIN_CHARS)
    val_ids   = prepare_tokens(enc, "valid", VAL_TOKENS_PATH,   MAX_VAL_CHARS)

    @info "Building reference NanoGPT" vocab_size=REF_CFG.vocab_size d_model=REF_CFG.d_model n_heads=REF_CFG.n_heads n_layers=REF_CFG.n_layers ctx_len=REF_CFG.ctx_len
    model = NanoGPT(REF_CFG.vocab_size, REF_CFG.d_model, REF_CFG.n_heads,
                     REF_CFG.n_layers, REF_CFG.ctx_len; d_ff = REF_CFG.d_ff)
    @info "Model constructed" n_params=n_parameters(model)

    @info "Starting training" n_steps=REF_CFG.n_steps batch_size=REF_CFG.batch_size lr=REF_CFG.lr
    losses = train_nanogpt!(model, train_ids, REF_CFG;
                             val_data = val_ids, log_every = 200,
                             checkpoint_path = PROGRESS_PATH,
                             checkpoint_every = 500)

    # persist the checkpoint alongside the config so it can be rebuilt later
    meta = Dict{Symbol, Any}(
        :d_model   => REF_CFG.d_model,
        :n_heads   => REF_CFG.n_heads,
        :n_layers  => REF_CFG.n_layers,
        :ctx_len   => REF_CFG.ctx_len,
        :d_ff      => REF_CFG.d_ff,
        :vocab_size => REF_CFG.vocab_size,
        :n_steps   => REF_CFG.n_steps,
    )
    save_checkpoint(CHECKPOINT_PATH, model; meta = meta, losses = losses)
    @info "Checkpoint saved" path=CHECKPOINT_PATH size_mb=round(filesize(CHECKPOINT_PATH) / 2^20; digits=1)
    isfile(PROGRESS_PATH) && rm(PROGRESS_PATH)  # clean up the progress checkpoint

    # ─── sanity checks ──────────────────────────────────────────────────────
    println("\n=== sample generation ===")
    sample = generate_text(model, enc, "Once upon a time", 80;
                            temperature = 0.8, top_k = 40)
    println(sample)

    println("\n=== induction-head scan ===")
    scores = induction_score(model, REF_CFG.vocab_size; n_trials = 32, seq_len = 32)
    for l in 1:size(scores, 1), h in 1:size(scores, 2)
        @printf("  layer %d head %d  score = %.3f\n", l, h, scores[l, h])
    end
    top_l, top_h = Tuple(argmax(scores))
    top_score = scores[top_l, top_h]
    @printf("\n  best head: layer %d, head %d  (score = %.3f)\n", top_l, top_h, top_score)

    if top_score < 0.3
        @warn "Best induction score is below the Task 3 target of 0.3; consider training longer or using a larger model."
    else
        @info "Reference checkpoint clears the Task 3 induction threshold." threshold=0.3 top_score=top_score
    end
end

main()

# finalize_reference.jl -- promote a progress checkpoint to the shipped one
# and report its induction-head scan.
#
# Run from the repository root:
#     julia --project=. --threads=auto scripts/finalize_reference.jl
#
# Loads `data/reference_checkpoint_progress.jld2`, runs a sample generation and
# induction-head scan, prints the scores, and copies the checkpoint to
# `data/reference_checkpoint.jld2` with full architecture metadata. Task 3 in
# the student notebook no longer gates on a minimum induction score, so this
# script always promotes; the scan is retained as a sanity check on the shipped
# checkpoint.

const _SCRIPT_DIR = @__DIR__
const _REPO_ROOT = abspath(joinpath(_SCRIPT_DIR, ".."))
cd(_REPO_ROOT)
include(joinpath(_REPO_ROOT, "Include.jl"))

const REF_D_MODEL   = 256
const REF_N_HEADS   = 4
const REF_N_LAYERS  = 4
const REF_CTX_LEN   = 256
const REF_D_FF      = 1024
const REF_VOCAB     = 50257

const CHECKPOINT_PATH   = joinpath(_PATH_TO_DATA, "reference_checkpoint.jld2")
const PROGRESS_PATH     = joinpath(_PATH_TO_DATA, "reference_checkpoint_progress.jld2")
const TRAIN_TOKENS_PATH = joinpath(_PATH_TO_DATA, "tinystories_train_tokens.jld2")

"""
Score induction heads on synthetic repeat sequences. Tokens are drawn from
`pool` (a vector of in-distribution token ids) rather than uniformly over the
full vocabulary — otherwise most random tokens never appeared in training and
their embeddings are near-random, which degrades the QK match inside the
induction circuit even when the circuit exists.
"""
function induction_score(model::NanoGPT, pool::AbstractVector{<:Integer};
                          n_trials::Int = 32, seq_len::Int = 32,
                          rng::AbstractRNG = Random.default_rng())::Matrix{Float32}
    n_layers = length(model.blocks)
    n_heads  = model.blocks[1].attn.n_heads
    scores = zeros(Float32, n_layers, n_heads)
    for _ in 1:n_trials
        tokens = rand(rng, pool, seq_len)                  # sample from in-vocab pool
        seq    = vcat(tokens, tokens)
        X      = reshape(seq, 2 * seq_len, 1)
        ws     = collect_attention_weights(model, X)
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

function main()
    @assert isfile(PROGRESS_PATH) "progress checkpoint not found: $PROGRESS_PATH"

    enc = load_gpt2_encoder()
    @info "Loading progress checkpoint..." path=PROGRESS_PATH size_mb=round(filesize(PROGRESS_PATH) / 2^20; digits=1)

    model = NanoGPT(REF_VOCAB, REF_D_MODEL, REF_N_HEADS, REF_N_LAYERS, REF_CTX_LEN; d_ff = REF_D_FF)
    _, meta, losses = load_checkpoint(PROGRESS_PATH, model)
    last_step = get(meta, :step, length(losses))
    @info "Progress checkpoint loaded" resumed_step=last_step final_train_loss=round(losses[end]; digits=4)

    println("\n=== sample generation ===")
    sample = generate_text(model, enc, "Once upon a time", 80;
                            temperature = 0.8, top_k = 40)
    println(sample)

    # Build the in-vocab sampling pool from the unique TinyStories training tokens.
    @info "Loading training-token pool for in-vocab induction scan..."
    train_ids = load(TRAIN_TOKENS_PATH, "ids")::Vector{Int}
    pool = unique(train_ids)
    @info "Pool built" n_unique_tokens=length(pool) fraction_of_vocab=round(length(pool) / REF_VOCAB; digits=3)

    println("\n=== induction-head scan (32 trials, seq_len 32, in-vocab tokens) ===")
    scores = induction_score(model, pool; n_trials = 32, seq_len = 32)
    for l in 1:size(scores, 1), h in 1:size(scores, 2)
        @printf("  layer %d head %d  score = %.3f\n", l, h, scores[l, h])
    end
    top_l, top_h = Tuple(argmax(scores))
    top_score = scores[top_l, top_h]
    @printf("\n  best head: layer %d, head %d  (score = %.3f)\n", top_l, top_h, top_score)

    final_meta = Dict{Symbol, Any}(
        :d_model    => REF_D_MODEL,
        :n_heads    => REF_N_HEADS,
        :n_layers   => REF_N_LAYERS,
        :ctx_len    => REF_CTX_LEN,
        :d_ff       => REF_D_FF,
        :vocab_size => REF_VOCAB,
        :n_steps    => last_step,
        :promoted_from_progress => true,
    )
    save_checkpoint(CHECKPOINT_PATH, model; meta = final_meta, losses = losses)
    @info "Final checkpoint written" path=CHECKPOINT_PATH size_mb=round(filesize(CHECKPOINT_PATH) / 2^20; digits=1)
    @info "Top induction head reported (no threshold gate)" top_l=top_l top_h=top_h top_score=top_score

    println("\nReady to ship. Next steps:")
    println("  1. (Optional) run the Solution notebook end-to-end as the acceptance test.")
    println("  2. Upload $(CHECKPOINT_PATH) as a GitHub Release asset.")
    println("  3. Commit and push the repo.")
end

main()

# diagnose_induction.jl -- probe the reference checkpoint with several induction
# scoring variants to understand why the random-token score didn't fire.
#
# Tries three scoring regimes on the progress checkpoint:
#   A. random uniform-vocab tokens
#   B. random in-vocab tokens (sampled from unique TinyStories training tokens)
#   C. natural-text prefixes sampled from the TinyStories training stream, duplicated
#
# For each, reports both the canonical "induction" score W[N+i, i+1] and the
# "prefix-match" score W[N+i, i] (attention to the past occurrence itself).

const _SCRIPT_DIR = @__DIR__
const _REPO_ROOT = abspath(joinpath(_SCRIPT_DIR, ".."))
cd(_REPO_ROOT)
include(joinpath(_REPO_ROOT, "Include.jl"))

const REF_D_MODEL, REF_N_HEADS, REF_N_LAYERS, REF_CTX_LEN, REF_D_FF, REF_VOCAB =
    256, 4, 4, 256, 1024, 50257
const PROGRESS_PATH     = joinpath(_PATH_TO_DATA, "reference_checkpoint_progress.jld2")
const TRAIN_TOKENS_PATH = joinpath(_PATH_TO_DATA, "tinystories_train_tokens.jld2")

# Return two (n_layers, n_heads) matrices: induction W[N+i, i+1] and prefix-match W[N+i, i].
function two_scores(model::NanoGPT, seqs::Vector{<:AbstractVector{<:Integer}})
    n_layers = length(model.blocks)
    n_heads = model.blocks[1].attn.n_heads
    ind  = zeros(Float32, n_layers, n_heads)
    pref = zeros(Float32, n_layers, n_heads)
    n_trials = length(seqs)
    for seq in seqs
        N = length(seq) ÷ 2
        X = reshape(seq, 2N, 1)
        ws = collect_attention_weights(model, X)
        for l in 1:n_layers, h in 1:n_heads
            a, b = 0.0f0, 0.0f0
            for i in 1:(N - 1)
                a += ws[l][N + i, i + 1, h, 1]  # induction: next-after-past-copy
                b += ws[l][N + i, i,     h, 1]  # prefix match: past copy itself
            end
            ind[l, h]  += a / (N - 1)
            pref[l, h] += b / (N - 1)
        end
    end
    return ind ./ n_trials, pref ./ n_trials
end

function print_scores(tag, ind, pref)
    println("\n--- $tag ---")
    n_layers, n_heads = size(ind)
    println("  (layer, head): induction W[N+i, i+1]  |  prefix-match W[N+i, i]")
    for l in 1:n_layers, h in 1:n_heads
        @printf("  L%d h%d:  %.3f  |  %.3f\n", l, h, ind[l, h], pref[l, h])
    end
    best_ind = argmax(ind); best_pref = argmax(pref)
    @printf("  best induction:    L%d h%d = %.3f\n", best_ind[1], best_ind[2], ind[best_ind])
    @printf("  best prefix-match: L%d h%d = %.3f\n", best_pref[1], best_pref[2], pref[best_pref])
end

function main()
    @info "Loading progress checkpoint..."
    model = NanoGPT(REF_VOCAB, REF_D_MODEL, REF_N_HEADS, REF_N_LAYERS, REF_CTX_LEN; d_ff = REF_D_FF)
    load_checkpoint(PROGRESS_PATH, model)

    train_ids = load(TRAIN_TOKENS_PATH, "ids")::Vector{Int}
    pool = unique(train_ids)
    @info "Loaded training pool" n_unique=length(pool) n_train_tokens=length(train_ids)

    rng = MersenneTwister(0)
    seq_len = 32
    n_trials = 32

    # A. random uniform-vocab tokens
    seqs_A = [vcat(tokens, tokens) for tokens in (rand(rng, 1:REF_VOCAB, seq_len) for _ in 1:n_trials)]
    indA, prefA = two_scores(model, seqs_A)
    print_scores("A. random uniform-vocab tokens", indA, prefA)

    # B. random in-vocab tokens (sampled from pool)
    seqs_B = [vcat(tokens, tokens) for tokens in (rand(rng, pool, seq_len) for _ in 1:n_trials)]
    indB, prefB = two_scores(model, seqs_B)
    print_scores("B. random in-vocab tokens (sampled from TinyStories pool)", indB, prefB)

    # C. natural-text prefixes from TinyStories, duplicated
    seqs_C = Vector{Vector{Int}}()
    for _ in 1:n_trials
        start = rand(rng, 1:(length(train_ids) - seq_len))
        prefix = train_ids[start:start + seq_len - 1]
        push!(seqs_C, vcat(prefix, prefix))
    end
    indC, prefC = two_scores(model, seqs_C)
    print_scores("C. natural-text prefixes (duplicated TinyStories text)", indC, prefC)
end

main()

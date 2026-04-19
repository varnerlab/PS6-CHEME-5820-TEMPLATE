# Compute.jl -- helper functions for PS6: Transformers.
#
# Contents:
#   * BPE tokenizer loading (GPT-2, shared by Tasks 1 and 2)
#   * Corpus download / tokenization helpers
#   * Training and evaluation utilities (perplexity, generation)
#   * Checkpoint save/load
#   * Induction-head scoring (Task 3)

# ────────────────────────────────────────────────────────────────────────────
# Tokenizer helpers
# ────────────────────────────────────────────────────────────────────────────

"""
    load_gpt2_encoder()

Return the GPT-2 byte-pair-encoding tokenizer. The returned object exposes two
callables: `enc.encode(text::String) -> Vector{Int}` and
`enc.decode(ids::Vector{Int}) -> String`. The integer ids are 1-based and take
values in `1:vocab_size(enc)`.
"""
function load_gpt2_encoder()
    return BytePairEncoding.load_tiktoken_encoder("gpt2")
end

"""
    vocab_size(enc) -> Int

Number of tokens in the tokenizer's vocabulary, including special tokens.
"""
vocab_size(enc) = length(enc.vocab)

# ────────────────────────────────────────────────────────────────────────────
# Corpus download and tokenization
# ────────────────────────────────────────────────────────────────────────────

"""
    download_tinystories(datapath; split="valid") -> String

Download a TinyStories text file to `datapath` and return the local path.
`split` is either `"valid"` (~20 MB) or `"train"` (~2 GB). Caches the file and
skips the download on subsequent calls.
"""
function download_tinystories(datapath::String; split::String = "valid")::String
    split in ("valid", "train") || throw(ArgumentError("split must be \"valid\" or \"train\""))
    fname = split == "valid" ? "TinyStories-valid.txt" : "TinyStories-train.txt"
    local_path = joinpath(datapath, fname)
    isfile(local_path) && return local_path

    !isdir(datapath) && mkpath(datapath)
    url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/$(fname)"
    @info "Downloading TinyStories split from Hugging Face..." split=split url=url
    Downloads.download(url, local_path; timeout = 600.0)
    @info "Download complete" path=local_path size=filesize(local_path)
    return local_path
end

"""
    encode_corpus(text::AbstractString, enc) -> Vector{Int}

Tokenize `text` with the BPE encoder and return the flat vector of token ids.
The ids can be fed directly to `sample_batch`.
"""
function encode_corpus(text::AbstractString, enc)::Vector{Int}
    return Int.(enc.encode(String(text)))
end

"""
    load_and_encode(path::AbstractString, enc; max_chars=nothing) -> Vector{Int}

Read a text file and tokenize it. If `max_chars` is supplied, only the leading
`max_chars` characters of the file are tokenized (useful for capping the
TinyStories training set to a manageable size).
"""
function load_and_encode(path::AbstractString, enc;
                         max_chars::Union{Nothing, Int} = nothing)::Vector{Int}
    text = read(path, String)
    if max_chars !== nothing && length(text) > max_chars
        # Trim before tokenization so expensive BPE work only sees the prefix we
        # intend to train on.
        text = text[1:max_chars]
    end
    return encode_corpus(text, enc)
end

# ────────────────────────────────────────────────────────────────────────────
# Training helpers
# ────────────────────────────────────────────────────────────────────────────

"""
    cosine_lr(step, warmup_steps, total_steps, lr_max, lr_min=0.1*lr_max) -> Float32

Cosine learning-rate schedule with a linear warmup. Returns `lr_max * step/warmup`
during the warmup window, then decays smoothly to `lr_min` over the remaining
steps.
"""
function cosine_lr(step::Int, warmup_steps::Int, total_steps::Int,
                    lr_max::Real, lr_min::Real = 0.1 * lr_max)
    if step < warmup_steps
        return Float32(lr_max * step / max(warmup_steps, 1))
    end
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = clamp(progress, 0.0, 1.0)
    return Float32(lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * progress)))
end

"""
    train_nanogpt!(model, data, cfg; val_data=nothing, log_every=100) -> Vector{Float32}

Train `model` on a tokenized `data::Vector{Int}` for `cfg.n_steps` steps using
AdamW with a cosine learning-rate schedule. Returns a vector of per-step
training losses. If `val_data` is provided, a held-out validation loss is
computed every `log_every` steps and printed.
"""
function train_nanogpt!(model::NanoGPT,
                        data::AbstractVector{<:Integer},
                        cfg::TrainingConfig;
                        val_data::Union{Nothing, AbstractVector{<:Integer}} = nothing,
                        log_every::Int = 100,
                        checkpoint_path::Union{Nothing, AbstractString} = nothing,
                        checkpoint_every::Int = 500,
                        rng::AbstractRNG = Random.default_rng())

    # Resume from a progress checkpoint if one exists. The optimizer state is
    # not persisted — AdamW's momenta warm-start quickly — but losses and the
    # step counter are, so the cosine schedule stays continuous across restart.
    start_step = 1
    losses = Float32[]
    if checkpoint_path !== nothing && isfile(checkpoint_path)
        _, meta, saved_losses = load_checkpoint(checkpoint_path, model)
        if haskey(meta, :step)
            start_step = meta[:step] + 1
            losses = collect(Float32, saved_losses)
            @info "Resuming from checkpoint" path=checkpoint_path resumed_step=meta[:step] n_losses=length(losses)
        end
    end

    if start_step > cfg.n_steps
        @info "Checkpoint already past n_steps; nothing to do" start_step cfg_n_steps=cfg.n_steps
        return losses
    end

    opt_state = Flux.setup(AdamW(cfg.lr), model)
    # Use the first 5% of training for warmup. With small transformers this is
    # usually enough to avoid unstable early updates without complicating the schedule.
    warmup = max(1, cfg.n_steps ÷ 20)

    for step in start_step:cfg.n_steps
        # Update the optimizer's learning rate every step so the warmup/cosine
        # schedule is applied continuously across the whole run.
        lr_step = cosine_lr(step, warmup, cfg.n_steps, cfg.lr)
        Flux.adjust!(opt_state, lr_step)

        # Draw one random minibatch of contiguous token windows and backpropagate
        # the next-token prediction loss through that batch.
        X, Y = sample_batch(data, cfg.batch_size, cfg.ctx_len; rng = rng)
        loss, grads = Flux.withgradient(model) do m
            nanogpt_loss(m, X, Y)
        end
        # `grads[1]` is the gradient with respect to the first argument of the
        # closure above, namely the model itself.
        Flux.update!(opt_state, model, grads[1])
        push!(losses, loss)

        if step == start_step || step == cfg.n_steps || step % log_every == 0
            msg = @sprintf("  step %5d | lr = %.2e | train loss = %.4f", step, lr_step, loss)
            if val_data !== nothing
                # Validation uses fresh random windows so the estimate is noisy
                # but representative without scanning the full dataset.
                vloss = validation_loss(model, val_data, cfg.batch_size, cfg.ctx_len; n_batches=4, rng=rng)
                msg *= @sprintf(" | val loss = %.4f", vloss)
            end
            println(msg)
        end

        if checkpoint_path !== nothing && (step % checkpoint_every == 0 || step == cfg.n_steps)
            # Save the running loss curve too so resumed runs can plot a single
            # continuous history rather than restarting the trace from zero.
            save_checkpoint(checkpoint_path, model;
                             meta = Dict(:step => step), losses = losses)
        end
    end
    return losses
end

"""
    validation_loss(model, data, batchsize, ctx_len; n_batches=4) -> Float32

Average cross-entropy loss computed on `n_batches` random chunks drawn from
`data`. Runs the model in inference mode (no gradients).
"""
function validation_loss(model::NanoGPT,
                          data::AbstractVector{<:Integer},
                          batchsize::Int, ctx_len::Int;
                          n_batches::Int = 4,
                          rng::AbstractRNG = Random.default_rng())::Float32
    total = 0.0f0
    for _ in 1:n_batches
        # Reuse the same batching logic as training, but do not take gradients.
        X, Y = sample_batch(data, batchsize, ctx_len; rng = rng)
        total += nanogpt_loss(model, X, Y)
    end
    return total / n_batches
end

"""
    perplexity(model, data, ctx_len; n_windows=32) -> Float32

Estimate perplexity as `exp(mean cross-entropy)` over `n_windows` random
contiguous chunks of length `ctx_len + 1` drawn from `data`. Lower is better.
"""
function perplexity(model::NanoGPT,
                     data::AbstractVector{<:Integer},
                     ctx_len::Int;
                     n_windows::Int = 32,
                     rng::AbstractRNG = Random.default_rng())::Float32
    ce = validation_loss(model, data, 1, ctx_len; n_batches = n_windows, rng = rng)
    return exp(ce)
end

# ────────────────────────────────────────────────────────────────────────────
# Generation helpers
# ────────────────────────────────────────────────────────────────────────────

"""
    generate_text(model, enc, prompt::AbstractString, n_new_tokens;
                   temperature=1.0, top_k=nothing, rng=Random.default_rng()) -> String

BPE-aware wrapper around [`generate`](@ref). Encodes `prompt` with the tokenizer,
generates `n_new_tokens` additional tokens, and decodes the full id sequence
back to a string. This is the BPE analog of [`sample_text`](@ref) from L13d.
"""
function generate_text(model::NanoGPT, enc,
                        prompt::AbstractString,
                        n_new_tokens::Int;
                        temperature::Real = 1.0,
                        top_k::Union{Nothing, Int} = nothing,
                        rng::AbstractRNG = Random.default_rng())::String
    prompt_ids = Int.(enc.encode(String(prompt)))
    full_ids = generate(model, prompt_ids, n_new_tokens;
                         temperature = temperature, top_k = top_k, rng = rng)
    return enc.decode(full_ids)
end

# ────────────────────────────────────────────────────────────────────────────
# Reference-checkpoint download
# ────────────────────────────────────────────────────────────────────────────

"""
URLs of the three instructor-trained checkpoints, hosted as GitHub Release
assets on the PS6 template repo. Edit these constants to match the actual
release URLs after publishing the release.
"""
const REFERENCE_CHECKPOINT_URL  = "https://github.com/varnerlab/PS6-CHEME-5820-TEMPLATE/releases/download/v1.0-s2026/reference_checkpoint.jld2"
const TASK1_CHECKPOINT_URL      = "https://github.com/varnerlab/PS6-CHEME-5820-TEMPLATE/releases/download/v1.0-s2026/task1_bpe_model.jld2"
const TASK2_TINY_CHECKPOINT_URL = "https://github.com/varnerlab/PS6-CHEME-5820-TEMPLATE/releases/download/v1.0-s2026/task2_tiny_model.jld2"

# Shared downloader so the three fetch_*_checkpoint helpers agree on directory
# creation, timeout, logging, and "already present" semantics.
function _fetch_release_asset(path::AbstractString, url::AbstractString, label::AbstractString, force::Bool)::String
    if isfile(path) && !force
        @info "$label already present; skipping download." path=path size_mb=round(filesize(path) / 2^20; digits=1)
        return path
    end
    !isdir(dirname(path)) && mkpath(dirname(path))
    @info "Downloading $label..." url=url target=path
    t0 = time()
    Downloads.download(url, path; timeout = 600.0)
    @info "$label download complete" path=path size_mb=round(filesize(path) / 2^20; digits=1) elapsed_s=round(time() - t0; digits=1)
    return path
end

"""
    fetch_reference_checkpoint(path = joinpath(_PATH_TO_DATA, "reference_checkpoint.jld2");
                                url = REFERENCE_CHECKPOINT_URL, force = false) -> String

Download the instructor-trained reference NanoGPT checkpoint from the course
GitHub Release and save it to `path`. Returns the local path. If the file
already exists and `force` is `false`, the download is skipped and the
existing path is returned.
"""
function fetch_reference_checkpoint(path::AbstractString = joinpath(_PATH_TO_DATA, "reference_checkpoint.jld2");
                                     url::AbstractString = REFERENCE_CHECKPOINT_URL,
                                     force::Bool = false)::String
    return _fetch_release_asset(path, url, "Reference checkpoint", force)
end

"""
    fetch_task1_checkpoint(path = joinpath(_PATH_TO_DATA, "task1_bpe_model.jld2");
                            url = TASK1_CHECKPOINT_URL, force = false) -> String

Download the instructor-trained Task 1 BPE Shakespeare checkpoint from the
course GitHub Release. Same semantics as `fetch_reference_checkpoint`.
"""
function fetch_task1_checkpoint(path::AbstractString = joinpath(_PATH_TO_DATA, "task1_bpe_model.jld2");
                                 url::AbstractString = TASK1_CHECKPOINT_URL,
                                 force::Bool = false)::String
    return _fetch_release_asset(path, url, "Task 1 checkpoint", force)
end

"""
    fetch_task2_tiny_checkpoint(path = joinpath(_PATH_TO_DATA, "task2_tiny_model.jld2");
                                 url = TASK2_TINY_CHECKPOINT_URL, force = false) -> String

Download the instructor-trained Task 2 tiny TinyStories checkpoint from the
course GitHub Release. Same semantics as `fetch_reference_checkpoint`.
"""
function fetch_task2_tiny_checkpoint(path::AbstractString = joinpath(_PATH_TO_DATA, "task2_tiny_model.jld2");
                                      url::AbstractString = TASK2_TINY_CHECKPOINT_URL,
                                      force::Bool = false)::String
    return _fetch_release_asset(path, url, "Task 2 tiny checkpoint", force)
end

# ────────────────────────────────────────────────────────────────────────────
# Checkpoint I/O
# ────────────────────────────────────────────────────────────────────────────

"""
    save_checkpoint(path, model; meta::Dict = Dict(), losses::Vector{Float32} = Float32[])

Save a NanoGPT `model` (and optional training metadata) to `path` in JLD2
format. `meta` is a free-form dict of hyperparameters preserved alongside the
weights so the model can be rebuilt identically.
"""
function save_checkpoint(path::AbstractString, model::NanoGPT;
                          meta::AbstractDict = Dict{Symbol, Any}(),
                          losses::AbstractVector{<:Real} = Float32[])
    # `Flux.state(model)` walks the layer tree and extracts only the trainable
    # arrays and buffers needed to rebuild this exact parameter state.
    jldsave(path;
        model_state = Flux.state(model),
        meta = Dict(meta),
        losses = collect(Float32, losses))
    return path
end

"""
    load_checkpoint(path, model::NanoGPT) -> (model, meta, losses)

Load weights from a JLD2 checkpoint into an already-constructed `NanoGPT` of
matching shape. Returns the mutated model alongside the saved metadata and the
training-loss history.
"""
function load_checkpoint(path::AbstractString, model::NanoGPT)
    # The caller is responsible for constructing `model` with the same shape as
    # the saved checkpoint. `Flux.loadmodel!` then copies the stored arrays in.
    f = jldopen(path, "r")
    Flux.loadmodel!(model, f["model_state"])
    meta = haskey(f, "meta") ? f["meta"] : Dict{Symbol, Any}()
    losses = haskey(f, "losses") ? f["losses"] : Float32[]
    close(f)
    return model, meta, losses
end

# ────────────────────────────────────────────────────────────────────────────
# Induction-head scoring (Task 3)
# ────────────────────────────────────────────────────────────────────────────

"""
    collect_attention_weights(model::NanoGPT, X_ids) -> Vector{Array{Float32,4}}

Run `model` on the `(T, B)` integer matrix `X_ids` and return a vector of
attention-weight tensors, one per decoder layer. Each entry has shape
`(T, T, n_heads, B)` with indexing `weights[query, key, head, batch]`. The
forward pass is replicated locally rather than reusing `model(X_ids)` so that
each layer's input can be fed through [`causal_attention_weights`](@ref).
"""
function collect_attention_weights(model::NanoGPT,
                                    X_ids::AbstractMatrix{<:Integer})::Vector{Array{Float32,4}}
    T, B = size(X_ids)
    @assert T <= model.ctx_len "T=$T exceeds model ctx_len=$(model.ctx_len)"
    d_model = size(model.tok_emb, 1)

    # Recreate the exact embedding path used in `NanoGPT(::)(X_ids)` so the
    # attention maps correspond to the same hidden states the model really sees.
    flat_ids = vec(X_ids)
    tok_e = model.tok_emb[:, flat_ids]
    H = reshape(tok_e, d_model, T, B)
    pos_e = model.pos_emb[:, 1:T]
    H = H .+ reshape(pos_e, d_model, T, 1)

    out = Vector{Array{Float32, 4}}(undef, length(model.blocks))
    for (l, block) in enumerate(model.blocks)
        # Attention is computed from the normalized hidden state because this is
        # a pre-LayerNorm transformer.
        normed = block.ln1(H)
        out[l] = causal_attention_weights(block.attn, normed)
        # Then manually advance the residual block so the next layer sees the
        # same activations it would have seen in a real forward pass.
        Y = H .+ block.attn(normed)
        Z = Y .+ block.ffn(block.ln2(Y))
        H = Z
    end
    return out
end

"""
    plot_induction_heatmap(scores; title="Induction scores") -> Plots.Plot

Render an `(n_layers, n_heads)` score matrix as a heatmap with layers on the
vertical axis and heads on the horizontal axis.
"""
function plot_induction_heatmap(scores::AbstractMatrix; title::AbstractString = "Induction scores")
    n_layers, n_heads = size(scores)
    # autoscale to the top score: with small checkpoints the induction signal
    # is well under 1, so a fixed clims = (0, 1) collapses every cell to the
    # same dark color and hides the per-head structure.
    vmax = max(maximum(scores), eps(Float32))
    return heatmap(1:n_heads, 1:n_layers, scores;
        xlabel = "head", ylabel = "layer",
        xticks = 1:n_heads, yticks = 1:n_layers,
        yflip = true,
        color = :viridis, clims = (0, vmax),
        title = title, size = (560, 360),
        titlefontsize = 10, framestyle = :box,
        right_margin = 8Plots.mm)
end

"""
    plot_attention_pattern(weights_layer, head, example=1; title="") -> Plots.Plot

Visualize the `(T, T)` attention matrix for a single `(layer, head)` pair taken
from one example in a batch. The matrix entry `w[q, k]` gives the probability
that the query at position `q` attends to the key at position `k`.
"""
function plot_attention_pattern(weights_layer::AbstractArray{<:Real, 4},
                                 head::Int, example::Int = 1;
                                 title::AbstractString = "")
    W = weights_layer[:, :, head, example]
    T = size(W, 1)
    return heatmap(1:T, 1:T, W;
        xlabel = "key position", ylabel = "query position",
        yflip = true, color = :viridis, clims = (0, maximum(W)),
        title = title, size = (400, 380),
        titlefontsize = 10, framestyle = :box, aspect_ratio = :equal)
end

# Types.jl -- PS6-specific type definitions.
#
# The transformer types (NanoGPT, DecoderBlock, CausalAttention, CharVocabulary)
# are defined in the L13d source files included earlier. This file reserves a
# place for any additional types introduced by the problem set; it is included
# after the transformer sources so those types are already visible.

"""
    TrainingConfig

Bundle of hyperparameters used by the student training loops in Tasks 1 and 2.
Kept as a plain struct so it can be printed and compared in autograder checks.
"""
struct TrainingConfig
    vocab_size :: Int
    d_model    :: Int
    n_heads    :: Int
    n_layers   :: Int
    ctx_len    :: Int
    d_ff       :: Int
    batch_size :: Int
    lr         :: Float32
    n_steps    :: Int
end

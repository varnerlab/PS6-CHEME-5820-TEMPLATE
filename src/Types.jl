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
    vocab_size :: Int  # tokenizer vocabulary size / logits dimension
    d_model    :: Int  # hidden width of each token representation
    n_heads    :: Int  # number of attention heads per decoder block
    n_layers   :: Int  # depth of the transformer stack
    ctx_len    :: Int  # maximum sequence length seen at once
    d_ff       :: Int  # inner width of the feedforward network
    batch_size :: Int  # number of training sequences per gradient step
    lr         :: Float32  # peak learning rate used by AdamW
    n_steps    :: Int  # total optimizer steps to run
end

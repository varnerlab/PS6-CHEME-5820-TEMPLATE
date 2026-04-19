# setup paths -
const _ROOT = @__DIR__;
const _PATH_TO_DATA = joinpath(_ROOT, "data");
const _PATH_TO_SRC = joinpath(_ROOT, "src");
const _PATH_TO_FIGS = joinpath(_ROOT, "figs");

!isdir(_PATH_TO_DATA) && mkpath(_PATH_TO_DATA);
!isdir(_PATH_TO_FIGS) && mkpath(_PATH_TO_FIGS);

# flag to check if the include file was called -
const _DID_INCLUDE_FILE_GET_CALLED = true;

using Pkg;
Pkg.activate(_ROOT);
if (isfile(joinpath(_ROOT, "Manifest.toml")) == false)
    Pkg.resolve(); Pkg.instantiate(); Pkg.update();
end

# load external packages -
using Statistics
using LinearAlgebra
using Random
using Downloads
using Printf
using DataFrames
using PrettyTables
using Flux
using NNlib
using OneHotArrays
using Plots
using JLD2
using StatsBase
using Test
using BytePairEncoding

# set the random seed for reproducibility -
Random.seed!(42);

# load transformer source files (copied verbatim from L13d) -
include(joinpath(_PATH_TO_SRC, "Shakespeare.jl"));
include(joinpath(_PATH_TO_SRC, "CausalAttention.jl"));
include(joinpath(_PATH_TO_SRC, "DecoderBlock.jl"));
include(joinpath(_PATH_TO_SRC, "NanoGPT.jl"));
include(joinpath(_PATH_TO_SRC, "Sample.jl"));

# load problem-set-specific code -
include(joinpath(_PATH_TO_SRC, "Types.jl"));
include(joinpath(_PATH_TO_SRC, "Compute.jl"));
include(joinpath(_PATH_TO_SRC, "Autograder.jl"));

# initialize the autograder -
GRADER = Grader();

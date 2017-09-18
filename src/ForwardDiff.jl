__precompile__()

module ForwardDiff

using DiffBase
using DiffBase: DiffResult, MutableDiffResult, ImmutableDiffResult
using StaticArrays
using Compat

import NaNMath
import SpecialFunctions
import RealInterface
import CommonSubexpressions

#############################
# types/functions/constants #
#############################

const NANSAFE_MODE_ENABLED = false

const REAL_TYPES = (AbstractFloat, Irrational, Integer, Rational, Real, Irrational{:e}, Irrational{:π})

const DEFAULT_CHUNK_THRESHOLD = 10

struct Chunk{N} end

function Chunk(input_length::Integer, threshold::Integer = DEFAULT_CHUNK_THRESHOLD)
    N = pickchunksize(input_length, threshold)
    return Chunk{N}()
end

function Chunk(x::AbstractArray, threshold::Integer = DEFAULT_CHUNK_THRESHOLD)
    return Chunk(length(x), threshold)
end

# Constrained to `N <= threshold`, minimize (in order of priority):
#   1. the number of chunks that need to be computed
#   2. the number of "left over" perturbations in the final chunk
function pickchunksize(input_length, threshold = DEFAULT_CHUNK_THRESHOLD)
    if input_length <= threshold
        return input_length
    else
        nchunks = round(Int, input_length / DEFAULT_CHUNK_THRESHOLD, RoundUp)
        return round(Int, input_length / nchunks, RoundUp)
    end
end

############
# includes #
############

include("partials.jl")
include("dual.jl")
include("config.jl")
include("utils.jl")
include("derivative.jl")
include("gradient.jl")
include("jacobian.jl")
include("hessian.jl")
include("deprecated.jl")

export DiffBase

end # module

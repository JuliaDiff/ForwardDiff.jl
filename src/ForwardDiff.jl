isdefined(Base, :__precompile__) && __precompile__()

module ForwardDiff

using Compat

import Calculus
import NaNMath

#############################
# types/functions/constants #
#############################

# multithreading #
#----------------#

const IS_MULTITHREADED_JULIA = VERSION >= v"0.5.0-dev+923"

if IS_MULTITHREADED_JULIA
    const NTHREADS = Base.Threads.nthreads()
    @inline compat_threadid() = Base.Threads.threadid()
else
    const NTHREADS = 1
    @inline compat_threadid() = 1
end

# function generation #
#---------------------#

const AUTO_DEFINED_UNARY_FUNCS = map(first, Calculus.symbolic_derivatives_1arg())
const NANMATH_FUNCS = (:sin, :cos, :tan, :asin, :acos, :acosh,
                       :atanh, :log, :log2, :log10, :lgamma, :log1p)

# chunk handling #
#----------------#

const MAX_CHUNK_SIZE = 10

immutable Chunk{N}
    function Chunk()
        @assert N <= MAX_CHUNK_SIZE "cannot create Chunk{$N}: max chunk size is $(MAX_CHUNK_SIZE)"
        return new()
    end
end

@inline Base.copy(chunk::Chunk) = chunk

# This is type-unstable, which is why our docs advise users to manually enter a chunk size
# when possible. The type instability here doesn't really hurt performance, since most of
# the heavy lifting happens behind a function barrier, but it can cause inference to give up
# when predicting the final output type of API functions.
pickchunk(x) = Chunk{pickchunksize(x)}()

function pickchunksize(x)
    k = length(x)
    if k <= MAX_CHUNK_SIZE
        return k
    else
        # Constrained to chunk <= MAX_CHUNK_SIZE, minimize (in order of priority):
        #   1. the number of chunks that need to be computed
        #   2. the number of "left over" perturbations in the final chunk
        nchunks = round(Int, k / MAX_CHUNK_SIZE, RoundUp)
        return round(Int, k / nchunks, RoundUp)
    end
end

# abstract types #
#----------------#

abstract ForwardDiffResult

############
# includes #
############

include("partials.jl")
include("dual.jl")
include("cache.jl")
include("derivative.jl")
include("gradient.jl")
include("jacobian.jl")
include("hessian.jl")

###########
# exports #
###########

export Chunk, DerivativeResult, GradientResult, JacobianResult, HessianResult

end # module

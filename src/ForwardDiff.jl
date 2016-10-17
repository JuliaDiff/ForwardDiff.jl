isdefined(Base, :__precompile__) && __precompile__()

module ForwardDiff

using Compat
using DiffBase
using DiffBase: DiffResult, GradientResult, JacobianResult, HessianResult

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

# chunk settings #
#----------------#

const MAX_CHUNK_SIZE = 10

# Constrained to chunk <= MAX_CHUNK_SIZE, minimize (in order of priority):
#   1. the number of chunks that need to be computed
#   2. the number of "left over" perturbations in the final chunk
function pickchunksize(k)
    if k <= MAX_CHUNK_SIZE
        return k
    else
        nchunks = round(Int, k / MAX_CHUNK_SIZE, RoundUp)
        return round(Int, k / nchunks, RoundUp)
    end
end

# API function utilities #
#------------------------#

@inline extract_value!(out::DiffResult, ydual) = DiffBase.value!(value, out, ydual)
@inline extract_value!(out, ydual) = nothing

vector_mode_dual_eval(f, x, opts::Tuple) = vector_mode_dual_eval(f, x, first(opts))

function vector_mode_dual_eval(f, x, opts)
    xdual = opts.duals
    seed!(xdual, x, opts.seeds)
    return f(xdual)
end

function vector_mode_dual_eval(f!, y, x, opts)
    ydual, xdual = opts.duals
    seed!(xdual, x, opts.seeds)
    seed!(ydual, y)
    f!(ydual, xdual)
    return ydual
end

############
# includes #
############

include("partials.jl")
include("dual.jl")
include("options.jl")
include("derivative.jl")
include("gradient.jl")
include("jacobian.jl")
include("hessian.jl")

###########
# exports #
###########

export DiffBase, DiffResult, GradientResult, JacobianResult, HessianResult

end # module

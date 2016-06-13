isdefined(Base, :__precompile__) && __precompile__()

module ForwardDiff

import Calculus
import NaNMath

#############################
# types/functions/constants #
#############################

const IS_MULTITHREADED_JULIA = VERSION >= v"0.5.0-dev+923" && Base.Threads.nthreads() > 1

if IS_MULTITHREADED_JULIA
    const NTHREADS = Base.Threads.nthreads()
    @inline compat_threadid() = Base.Threads.threadid()
else
    const NTHREADS = 1
    @inline compat_threadid() = 1
end

const AUTO_DEFINED_UNARY_FUNCS = map(first, Calculus.symbolic_derivatives_1arg())
const NANMATH_FUNCS = (:sin, :cos, :tan, :asin, :acos, :acosh,
                       :atanh, :log, :log2, :log10, :lgamma, :log1p)

const MAX_CHUNK_SIZE = 20

abstract ForwardDiffResult

############
# includes #
############

include("partials.jl")
include("dual.jl")
include("utils.jl")
include("derivative.jl")
include("gradient.jl")
include("jacobian.jl")
include("hessian.jl")

###########
# exports #
###########

export Chunk, DerivativeResult, GradientResult, JacobianResult, HessianResult

end # module

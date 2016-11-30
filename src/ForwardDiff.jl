isdefined(Base, :__precompile__) && __precompile__()

module ForwardDiff

using Compat
using DiffBase
using DiffBase: DiffResult

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

const CHUNK_THRESHOLD = 10

############
# includes #
############

include("partials.jl")
include("dual.jl")
include("config.jl")
include("api_utils.jl")
include("derivative.jl")
include("gradient.jl")
include("jacobian.jl")
include("hessian.jl")
include("deprecated.jl")

export DiffBase

end # module

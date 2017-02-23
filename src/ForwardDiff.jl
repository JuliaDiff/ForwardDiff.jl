__precompile__()

module ForwardDiff

using Compat
using DiffBase
using DiffBase: DiffResult

import Calculus
import NaNMath
import SpecialFunctions

#############################
# types/functions/constants #
#############################

# NaN-safe mode switch #
#----------------------#

const NANSAFE_MODE_ENABLED = false

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

const SPECIAL_FUNCS = (:erf, :erfc, :erfinv, :erfcinv, :erfi, :erfcx,
                       :dawson, :digamma, :eta, :zeta, :airyai, :airyaiprime,
                       :airybi, :airybiprime, :airyaix, :besselj, :besselj0,
                       :besselj1, :besseljx, :bessely, :bessely0, :bessely1,
                       :besselyx, :besselh, :hankelh1, :hankelh1x, :hankelh2,
                       :hankelh2x, :besseli, :besselix, :besselk, :besselkx)

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

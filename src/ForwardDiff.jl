module ForwardDiff

using DiffRules, DiffResults
using DiffResults: DiffResult, MutableDiffResult, ImmutableDiffResult
using StaticArrays
if VERSION >= v"1.6"
    using Preferences
end
using Random
using LinearAlgebra
import SIMD: Vec

import Printf
import NaNMath
import SpecialFunctions
import LogExpFunctions
import CommonSubexpressions

const SIMDFloat = Union{Float64, Float32}
const SIMDInt = Union{
                       Int128, Int64, Int32, Int16, Int8,
                       UInt128, UInt64, UInt32, UInt16, UInt8,
                     }
const SIMDType = Union{SIMDFloat, SIMDInt}

include("prelude.jl")
include("partials.jl")
include("dual.jl")
include("config.jl")
include("apiutils.jl")
include("derivative.jl")
include("gradient.jl")
include("jacobian.jl")
include("hessian.jl")

export DiffResults

end # module

module ForwardDiff

using DiffRules, DiffResults
using DiffResults: DiffResult, MutableDiffResult
if VERSION >= v"1.6"
    using Preferences
end
using Random
using LinearAlgebra
if VERSION < v"1.2.0-DEV.125" # 1da48c2e4028c1514ed45688be727efbef1db884
    require_one_based_indexing(A...) = !Base.has_offset_axes(A...) || throw(ArgumentError(
        "offset arrays are not supported but got an array with index other than 1"))
else
    using Base: require_one_based_indexing
end
import Printf
import NaNMath
import SpecialFunctions
import LogExpFunctions
import CommonSubexpressions

include("prelude.jl")
include("partials.jl")
include("dual.jl")
include("config.jl")
include("apiutils.jl")
include("derivative.jl")
include("gradient.jl")
include("jacobian.jl")
include("hessian.jl")

if !isdefined(Base, :get_extension)
    include("../ext/ForwardDiffStaticArraysExt.jl")
end

export DiffResults

end # module

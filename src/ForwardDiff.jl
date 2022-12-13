module ForwardDiff

using DiffRules, DiffResults
using DiffResults: DiffResult, MutableDiffResult
using Preferences
using Random
using LinearAlgebra
using Base: require_one_based_indexing
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
    include("../ext/StaticArraysExt.jl")
end

export DiffResults

end # module

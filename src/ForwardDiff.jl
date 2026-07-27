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

export DiffResults

@static if VERSION >= v"1.11.0-DEV.469"
    eval(Expr(:public, :derivative, :derivative!, :gradient, :gradient!, :jacobian, :jacobian!, :hessian, :hessian!, :GradientConfig, :JacobianConfig, :HessianConfig, :DerivativeConfig, :Chunk, :Dual, :value, :can_dual))
end

end # module

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

function __init__()
    if NANSAFE_MODE_PREFERENCE !== nothing
        Base.depwarn(
            "The `nansafe_mode` preference of ForwardDiff is deprecated. " *
            "NaN-safe mode is enabled by default since ForwardDiff 1.5.0, and the preference to disable it is now called `unsafe_mode`. " *
            "For now, `nansafe_mode = false` still disables NaN-safe mode unless `unsafe_mode` is set. " *
            "Please remove the `nansafe_mode` preference with `Preferences.delete_preferences!(ForwardDiff, \"nansafe_mode\"; force = true)` and restart Julia.",
            :__init__;
            force = true,
        )
    end
    return nothing
end

end # module

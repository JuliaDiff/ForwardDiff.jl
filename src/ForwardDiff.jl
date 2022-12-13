module ForwardDiff

using DiffRules, DiffResults
using DiffResults: DiffResult, MutableDiffResult, ImmutableDiffResult
using StaticArrays
using Preferences
using Random
using LinearAlgebra

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

end # module

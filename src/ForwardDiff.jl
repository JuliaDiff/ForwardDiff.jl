module ForwardDiff

using DiffRules, DiffResults
using DiffResults: DiffResult, MutableDiffResult
if VERSION >= v"1.6"
    using Preferences
end
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

if !isdefined(Base, :get_extension)
    include("../ext/ForwardDiffStaticArraysExt.jl")
end

export DiffResults

end # module

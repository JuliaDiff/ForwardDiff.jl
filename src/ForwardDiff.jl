__precompile__()

module ForwardDiff

using DiffRules, DiffResults
using DiffResults: DiffResult, MutableDiffResult, ImmutableDiffResult
using StaticArrays
using SpecialFunctions
using Compat
using Compat.Random

import NaNMath
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
include("deprecated.jl")

# This is a deprecation binding and should be removed in the next minor release.
const DiffBase = DiffResults

export DiffBase, DiffResults

end # module

module DerivativeTest

import Calculus

using Base.Test
using ForwardDiff

include(joinpath(dirname(@__FILE__), "utils.jl"))

########################
# test vs. Calculus.jl #
########################

const x = 1

for f in DiffBase.NUMBER_TO_NUMBER_FUNCS
    println("  ...testing $f")
    v = f(x)
    d = ForwardDiff.derivative(f, x)
    @test_approx_eq_eps d Calculus.derivative(f, x) FINITEDIFF_ERROR

    out = DiffBase.DiffResult(zero(v))
    ForwardDiff.derivative!(out, f, x)
    @test_approx_eq DiffBase.value(out) v
    @test_approx_eq DiffBase.derivative(out) d
end

for f in DiffBase.NUMBER_TO_ARRAY_FUNCS
    println("  ...testing $f")
    v = f(x)
    d = ForwardDiff.derivative(f, x)
    @test_approx_eq_eps d Calculus.derivative(f, x) FINITEDIFF_ERROR

    out = similar(v)
    ForwardDiff.derivative!(out, f, x)
    @test_approx_eq out d

    out = DiffBase.DiffResult(similar(v))
    ForwardDiff.derivative!(out, f, x)
    @test_approx_eq DiffBase.value(out) v
    @test_approx_eq DiffBase.derivative(out) d
end


end # module

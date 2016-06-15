module DerivativeTest

import Calculus

using Base.Test
using ForwardDiff

include(joinpath(dirname(@__FILE__), "utils.jl"))

########################
# test vs. Calculus.jl #
########################

x = 1

for f in NUMBER_TO_NUMBER_FUNCS
    println("  ...testing $f")
    v = f(x)
    d = ForwardDiff.derivative(f, x)
    @test_approx_eq_eps d Calculus.derivative(f, x) FINITEDIFF_ERROR

    out = DerivativeResult(zero(v))
    ForwardDiff.derivative!(out, f, x)
    @test_approx_eq ForwardDiff.value(out) v
    @test_approx_eq ForwardDiff.derivative(out) d
end

for f in NUMBER_TO_ARRAY_FUNCS
    println("  ...testing $f")
    v = f(x)
    d = ForwardDiff.derivative(f, x)
    @test_approx_eq_eps d Calculus.derivative(f, x) FINITEDIFF_ERROR

    out = similar(v)
    ForwardDiff.derivative!(out, f, x)
    @test_approx_eq out d

    out = DerivativeResult(similar(v))
    ForwardDiff.derivative!(out, f, x)
    @test_approx_eq ForwardDiff.value(out) v
    @test_approx_eq ForwardDiff.derivative(out) d
end


end # module

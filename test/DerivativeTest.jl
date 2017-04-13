module DerivativeTest

import Calculus

using Base.Test
using ForwardDiff

include(joinpath(dirname(@__FILE__), "utils.jl"))

srand(1)

########################
# test vs. Calculus.jl #
########################

const x = 1

for f in DiffBase.NUMBER_TO_NUMBER_FUNCS
    println("  ...testing $f")
    v = f(x)
    d = ForwardDiff.derivative(f, x)
    @test isapprox(d, Calculus.derivative(f, x), atol=FINITEDIFF_ERROR)

    out = DiffBase.DiffResult(zero(v), zero(v))
    ForwardDiff.derivative!(out, f, x)
    @test isapprox(DiffBase.value(out), v)
    @test isapprox(DiffBase.derivative(out), d)
end

for f in DiffBase.NUMBER_TO_ARRAY_FUNCS
    println("  ...testing $f")
    v = f(x)
    d = ForwardDiff.derivative(f, x)

    @test !(eltype(d) <: ForwardDiff.Dual)
    @test isapprox(d, Calculus.derivative(f, x), atol=FINITEDIFF_ERROR)

    out = similar(v)
    ForwardDiff.derivative!(out, f, x)
    @test isapprox(out, d)

    out = DiffBase.DiffResult(similar(v), similar(d))
    ForwardDiff.derivative!(out, f, x)
    @test isapprox(DiffBase.value(out), v)
    @test isapprox(DiffBase.derivative(out), d)
end

end # module

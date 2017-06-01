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
    out = ForwardDiff.derivative!(out, f, x)
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

for f! in DiffBase.INPLACE_NUMBER_TO_ARRAY_FUNCS
    println("  ...testing $f!")
    m, n = 3, 2
    y = zeros(m, n)
    f = x -> (tmp = similar(y, promote_type(eltype(y), typeof(x)), m, n); f!(tmp, x); tmp)
    v = f(x)
    cfg = ForwardDiff.DerivativeConfig(f!, y, x)
    d = ForwardDiff.derivative(f, x)

    fill!(y, 0.0)
    @test isapprox(ForwardDiff.derivative(f!, y, x), d)
    @test isapprox(v, y)

    fill!(y, 0.0)
    @test isapprox(ForwardDiff.derivative(f!, y, x, cfg), d)
    @test isapprox(v, y)

    out = similar(v)
    fill!(y, 0.0)
    ForwardDiff.derivative!(out, f!, y, x)
    @test isapprox(out, d)
    @test isapprox(v, y)

    out = similar(v)
    fill!(y, 0.0)
    ForwardDiff.derivative!(out, f!, y, x, cfg)
    @test isapprox(out, d)
    @test isapprox(v, y)

    out = DiffBase.DiffResult(similar(v), similar(d))
    ForwardDiff.derivative!(out, f!, y, x)
    @test isapprox(v, y)
    @test isapprox(DiffBase.value(out), v)
    @test isapprox(DiffBase.derivative(out), d)

    out = DiffBase.DiffResult(similar(v), similar(d))
    ForwardDiff.derivative!(out, f!, y, x, cfg)
    @test isapprox(v, y)
    @test isapprox(DiffBase.value(out), v)
    @test isapprox(DiffBase.derivative(out), d)
end

end # module

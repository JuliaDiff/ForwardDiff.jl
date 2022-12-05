module DerivativeTest

import Calculus

using Test
using Random
using ForwardDiff
using DiffTests

include(joinpath(dirname(@__FILE__), "utils.jl"))

Random.seed!(1)

########################
# test vs. Calculus.jl #
########################

const x = 1

for f in DiffTests.NUMBER_TO_NUMBER_FUNCS
    println("  ...testing $f")
    v = f(x)
    d = ForwardDiff.derivative(f, x)
    @test isapprox(d, Calculus.derivative(f, x), atol=FINITEDIFF_ERROR)

    out = DiffResults.DiffResult(zero(v), zero(v))
    out = ForwardDiff.derivative!(out, f, x)
    @test isapprox(DiffResults.value(out), v)
    @test isapprox(DiffResults.derivative(out), d)
end

for f in DiffTests.NUMBER_TO_ARRAY_FUNCS
    println("  ...testing $f")
    v = f(x)
    d = ForwardDiff.derivative(f, x)

    @test !(eltype(d) <: ForwardDiff.Dual)
    @test isapprox(d, Calculus.derivative(f, x), atol=FINITEDIFF_ERROR)

    out = similar(v)
    out = ForwardDiff.derivative!(out, f, x)
    @test isapprox(out, d)

    out = DiffResults.DiffResult(similar(v), similar(d))
    out = ForwardDiff.derivative!(out, f, x)
    @test isapprox(DiffResults.value(out), v)
    @test isapprox(DiffResults.derivative(out), d)
end

for f! in DiffTests.INPLACE_NUMBER_TO_ARRAY_FUNCS
    println("  ...testing $f!")
    m, n = 3, 2
    y = fill(0.0, m, n)
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

    out = DiffResults.DiffResult(similar(v), similar(d))
    out = ForwardDiff.derivative!(out, f!, y, x)
    @test isapprox(v, y)
    @test isapprox(DiffResults.value(out), v)
    @test isapprox(DiffResults.derivative(out), d)

    out = DiffResults.DiffResult(similar(v), similar(d))
    out = ForwardDiff.derivative!(out, f!, y, x, cfg)
    @test isapprox(v, y)
    @test isapprox(DiffResults.value(out), v)
    @test isapprox(DiffResults.derivative(out), d)
end

@testset "exponential function at base zero" begin
    @test (x -> ForwardDiff.derivative(y -> x^y, -0.5))(0.0) === -Inf
    @test (x -> ForwardDiff.derivative(y -> x^y,  0.0))(0.0) === -Inf
    @test (x -> ForwardDiff.derivative(y -> x^y,  0.5))(0.0) === 0.0
    @test (x -> ForwardDiff.derivative(y -> x^y,  1.5))(0.0) === 0.0
end

@testset "dimension error for derivative" begin
    @test_throws DimensionMismatch ForwardDiff.derivative(sum, fill(2pi, 3))
end

@testset "complex output" begin
    @test ForwardDiff.derivative(x -> (1+im)*x, 0) == (1+im)
end

end # module

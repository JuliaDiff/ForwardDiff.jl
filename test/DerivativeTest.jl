module DerivativeTest

import Calculus
import LinearAlgebra
import NaNMath

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

@testset "$f" for f in DiffTests.NUMBER_TO_NUMBER_FUNCS
    v = f(x)
    d = ForwardDiff.derivative(f, x)
    @test isapprox(d, Calculus.derivative(f, x), atol=FINITEDIFF_ERROR)

    out = DiffResults.DiffResult(zero(v), zero(v))
    out = ForwardDiff.derivative!(out, f, x)
    @test isapprox(DiffResults.value(out), v)
    @test isapprox(DiffResults.derivative(out), d)
end

@testset "$f" for f in DiffTests.NUMBER_TO_ARRAY_FUNCS
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

@testset "$(f!)" for f! in DiffTests.INPLACE_NUMBER_TO_ARRAY_FUNCS
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

@testset "exponentiation with NaNMath" begin
    @test isnan(ForwardDiff.derivative(x -> NaNMath.pow(NaN, x), 1.0))
    @test isnan(ForwardDiff.derivative(x -> NaNMath.pow(x,NaN), 1.0))
    @test !isnan(ForwardDiff.derivative(x -> NaNMath.pow(1.0, x),1.0))
    @test isnan(ForwardDiff.derivative(x -> NaNMath.pow(x,0.5), -1.0))

    @test isnan(ForwardDiff.derivative(x -> x^NaN, 2.0))
    @test ForwardDiff.derivative(x -> x^2.0,2.0) == 4.0
    @test_throws DomainError ForwardDiff.derivative(x -> x^0.5, -1.0)
end

@testset "dimension error for derivative" begin
    @test_throws DimensionMismatch ForwardDiff.derivative(sum, fill(2pi, 3))
end

@testset "complex output" begin
    @test ForwardDiff.derivative(x -> (1+im)*x, 0) == (1+im)
end

@testset "NaN-safe mode" begin
    x = ForwardDiff.derivative(log ∘ zero, 1.0)
    if ForwardDiff.NANSAFE_MODE_ENABLED
        @test iszero(x)
    else
        @test isnan(x)
    end
end

@testset "Givens rotations: Derivatives" begin
    # Test different branches in `LinearAlgebra.givensAlgorithm`
    for f in [randexp(), -randexp()], g in [0.0, f / 2, 2f, -f / 2, -2f], i in 1:3
        @test ForwardDiff.derivative(x -> LinearAlgebra.givensAlgorithm(x, g)[i], f) ≈
            Calculus.derivative(x -> LinearAlgebra.givensAlgorithm(x, g)[i], f)
        @test ForwardDiff.derivative(x -> LinearAlgebra.givensAlgorithm(f, x)[i], g) ≈
            Calculus.derivative(x -> LinearAlgebra.givensAlgorithm(f, x)[i], g)
    end
end

end # module

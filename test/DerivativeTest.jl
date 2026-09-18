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

# `abs`/`conj`/`real`/`angle` are nowhere complex differentiable, and work only because the real and
# imaginary parts carry separate partials.
const COMPLEX_OUTPUT_FUNCS = (
    ("cis and exp",    (y, x) -> (y[1] = cis(x); y[2] = exp((1+2im)*x)),
                       x -> [cis(x), exp((1+2im)*x)]),
    ("abs and conj",   (y, x) -> (z = cis(x)*(2+x); y[1] = abs(z)+0im; y[2] = conj(z)),
                       x -> (z = cis(x)*(2+x); [abs(z)+0im, conj(z)])),
    ("real and angle", (y, x) -> (z = (1+2im)*x^2+3; y[1] = real(z)+0im; y[2] = angle(z)+0im),
                       x -> (z = (1+2im)*x^2+3; [real(z)+0im, angle(z)+0im])),
    ("sqrt and log",   (y, x) -> (z = 2cis(x)+3; y[1] = sqrt(z); y[2] = log(z)/(x+1im)),
                       x -> (z = 2cis(x)+3; [sqrt(z), log(z)/(x+1im)])),
)

@testset "complex output" begin
    @test ForwardDiff.derivative(x -> (1+im)*x, 0) == (1+im)

    # The in-place path must agree exactly with the non-mutating one: same `Complex{Dual}` arithmetic.
    # `y` only matches approximately, since recomputing `f(x)` in `Float64` reassociates.
    @testset "in-place, $name" for (name, f!, f) in COMPLEX_OUTPUT_FUNCS
        x = 0.7
        v, d = f(x), ForwardDiff.derivative(f, x)
        @test !(eltype(d) <: ForwardDiff.Dual)
        @test d ≈ Calculus.derivative(f, x) atol=FINITEDIFF_ERROR

        y = Vector{ComplexF64}(undef, 2)
        for cfg in ((), (ForwardDiff.DerivativeConfig(f!, y, x),))
            @test ForwardDiff.derivative(f!, y, x, cfg...) == d
            @test y ≈ v

            out = similar(d)
            @test ForwardDiff.derivative!(out, f!, y, x, cfg...) === out
            @test out == d

            out = DiffResults.DiffResult(similar(v), similar(d))
            @test ForwardDiff.derivative!(out, f!, y, x, cfg...) === out
            @test DiffResults.value(out) ≈ v
            @test DiffResults.derivative(out) == d
        end
    end

    @testset "in-place, entries f! leaves alone" begin
        y = ComplexF64[0, 9-4im]
        d = ForwardDiff.derivative((y, x) -> (y[1] = cis(x)), y, 0.3)
        @test d[1] ≈ im*cis(0.3)
        @test d[2] === 0.0+0.0im
        @test y[2] === 9.0-4.0im
    end

    @testset "in-place, nested" begin
        h!(y, x) = (y[1] = cis(2x); y[2] = x^2 + 3im*x)
        @test ForwardDiff.derivative(1.0) do x
            ForwardDiff.derivative(h!, Vector{Complex{typeof(x)}}(undef, 2), x)
        end ≈ [-4cis(2.0), 2.0+0im]
    end

    # `Complex{BigFloat}` is not `isbitstype`, covering the `isassigned` branch of
    # `_seed_zero_partials!`; the `Matrix` covers a non-vector shape.
    @testset "in-place, $(eltype(y))" for y in (Vector{Complex{BigFloat}}(undef, 3),
                                               Matrix{ComplexF64}(undef, 2, 2))
        f(x) = fill(cis(x)*(1+x), size(y))
        x = convert(real(eltype(y)), 4//10)
        @test ForwardDiff.derivative((y, x) -> (y .= cis(x)*(1+x)), y, x) ==
              ForwardDiff.derivative(f, x)
        @test y ≈ f(x)
    end
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

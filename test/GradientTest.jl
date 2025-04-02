module GradientTest

import Calculus
import NaNMath

using Test
using LinearAlgebra
using ForwardDiff
using ForwardDiff: Dual, Tag
using StaticArrays
using DiffTests

include(joinpath(dirname(@__FILE__), "utils.jl"))

##################
# hardcoded test #
##################

f = DiffTests.rosenbrock_1
x = [0.1, 0.2, 0.3]
v = f(x)
g = [-9.4, 15.6, 52.0]

@testset "Rosenbrock, chunk size = $c and tag = $(repr(tag))" for c in (1, 2, 3), tag in (nothing, Tag(f, eltype(x)))
    cfg = ForwardDiff.GradientConfig(f, x, ForwardDiff.Chunk{c}(), tag)

    @test eltype(cfg) == Dual{typeof(tag), eltype(x), c}

    @test isapprox(g, ForwardDiff.gradient(f, x, cfg))
    @test isapprox(g, ForwardDiff.gradient(f, x))

    out = similar(x)
    ForwardDiff.gradient!(out, f, x, cfg)
    @test isapprox(out, g)

    out = similar(x)
    ForwardDiff.gradient!(out, f, x)
    @test isapprox(out, g)

    out = DiffResults.GradientResult(x)
    ForwardDiff.gradient!(out, f, x, cfg)
    @test isapprox(DiffResults.value(out), v)
    @test isapprox(DiffResults.gradient(out), g)

    out = DiffResults.GradientResult(x)
    ForwardDiff.gradient!(out, f, x)
    @test isapprox(DiffResults.value(out), v)
end

cfgx = ForwardDiff.GradientConfig(sin, x)
@test_throws ForwardDiff.InvalidTagException ForwardDiff.gradient(f, x, cfgx)
@test ForwardDiff.gradient(f, x, cfgx, Val{false}()) == ForwardDiff.gradient(f,x)


########################
# test vs. Calculus.jl #
########################

@testset "$f" for f in DiffTests.VECTOR_TO_NUMBER_FUNCS
    v = f(X)
    g = ForwardDiff.gradient(f, X)
    @test isapprox(g, Calculus.gradient(f, X), atol=FINITEDIFF_ERROR)
    @testset "... with chunk size = $c and tag = $(repr(tag))" for c in CHUNK_SIZES, tag in (nothing, Tag(f, eltype(x)))
        cfg = ForwardDiff.GradientConfig(f, X, ForwardDiff.Chunk{c}(), tag)

        out = ForwardDiff.gradient(f, X, cfg)
        @test isapprox(out, g)

        out = similar(X)
        ForwardDiff.gradient!(out, f, X, cfg)
        @test isapprox(out, g)

        out = DiffResults.GradientResult(X)
        ForwardDiff.gradient!(out, f, X, cfg)
        @test isapprox(DiffResults.value(out), v)
        @test isapprox(DiffResults.gradient(out), g)
    end
end

##########################################
# test specialized StaticArray codepaths #
##########################################

@testset "Specialized StaticArray codepaths: $T" for T in (StaticArrays.SArray, StaticArrays.MArray)
    x = rand(3, 3)

    sx = T{Tuple{3,3}}(x)

    cfg = ForwardDiff.GradientConfig(nothing, x)
    scfg = ForwardDiff.GradientConfig(nothing, sx)

    actual = ForwardDiff.gradient(prod, x)
    @test ForwardDiff.gradient(prod, sx) == actual
    @test ForwardDiff.gradient(prod, sx, cfg) == actual
    @test ForwardDiff.gradient(prod, sx, scfg) == actual
    @test ForwardDiff.gradient(prod, sx, scfg) isa StaticArray
    @test ForwardDiff.gradient(prod, sx, scfg, Val{false}()) == actual
    @test ForwardDiff.gradient(prod, sx, scfg, Val{false}()) isa StaticArray

    out = similar(x)
    ForwardDiff.gradient!(out, prod, sx)
    @test out == actual

    out = similar(x)
    ForwardDiff.gradient!(out, prod, sx, cfg)
    @test out == actual

    out = similar(x)
    ForwardDiff.gradient!(out, prod, sx, scfg)
    @test out == actual

    result = DiffResults.GradientResult(x)
    result = ForwardDiff.gradient!(result, prod, x)

    result1 = DiffResults.GradientResult(x)
    result2 = DiffResults.GradientResult(x)
    result3 = DiffResults.GradientResult(x)
    result1 = ForwardDiff.gradient!(result1, prod, sx)
    result2 = ForwardDiff.gradient!(result2, prod, sx, cfg)
    result3 = ForwardDiff.gradient!(result3, prod, sx, scfg)
    @test DiffResults.value(result1) == DiffResults.value(result)
    @test DiffResults.value(result2) == DiffResults.value(result)
    @test DiffResults.value(result3) == DiffResults.value(result)
    @test DiffResults.gradient(result1) == DiffResults.gradient(result)
    @test DiffResults.gradient(result2) == DiffResults.gradient(result)
    @test DiffResults.gradient(result3) == DiffResults.gradient(result)

    sresult1 = DiffResults.GradientResult(sx)
    sresult2 = DiffResults.GradientResult(sx)
    sresult3 = DiffResults.GradientResult(sx)
    sresult1 = ForwardDiff.gradient!(sresult1, prod, sx)
    sresult2 = ForwardDiff.gradient!(sresult2, prod, sx, cfg)
    sresult3 = ForwardDiff.gradient!(sresult3, prod, sx, scfg)
    @test DiffResults.value(sresult1) == DiffResults.value(result)
    @test DiffResults.value(sresult2) == DiffResults.value(result)
    @test DiffResults.value(sresult3) == DiffResults.value(result)
    @test DiffResults.gradient(sresult1) == DiffResults.gradient(result)
    @test DiffResults.gradient(sresult2) == DiffResults.gradient(result)
    @test DiffResults.gradient(sresult3) == DiffResults.gradient(result)

    # make sure this is not a source of type instability
    @inferred ForwardDiff.GradientConfig(f, sx)
end

@testset "exponential function at base zero" begin
    @test isequal(ForwardDiff.gradient(t -> t[1]^t[2], [0.0, -0.5]), [NaN, NaN])
    @test isequal(ForwardDiff.gradient(t -> t[1]^t[2], [0.0,  0.0]), [NaN, NaN])
    @test isequal(ForwardDiff.gradient(t -> t[1]^t[2], [0.0,  0.5]), [Inf, NaN])
    @test isequal(ForwardDiff.gradient(t -> t[1]^t[2], [0.0,  1.5]), [0.0, 0.0])
end

#############
# bug fixes #
#############

# Issue 399
@testset "chunk size zero" begin
    f_const(x) = 1.0
    g_grad_const = x -> ForwardDiff.gradient(f_const, x)
    @test g_grad_const([1.0]) == [0.0]
    @test isempty(g_grad_const(zeros(Float64, 0)))
end

@testset "dimension errors for gradient" begin
    @test_throws DimensionMismatch ForwardDiff.gradient(identity, 2pi) # input
    @test_throws DimensionMismatch ForwardDiff.gradient(identity, fill(2pi, 2)) # vector_mode_gradient
    @test_throws DimensionMismatch ForwardDiff.gradient(identity, fill(2pi, 10^6)) # chunk_mode_gradient
end

# Issue 548
@testset "ArithmeticStyle" begin
    function f(p)
        sum(collect(0.0:p[1]:p[2]))
    end
    @test ForwardDiff.gradient(f, [0.3, 25.0]) == [3486.0, 0.0]
end

@testset "det with branches" begin
    # Issue 197
    det2(A) = return (
        A[1,1]*(A[2,2]*A[3,3]-A[2,3]*A[3,2]) -
        A[1,2]*(A[2,1]*A[3,3]-A[2,3]*A[3,1]) +
        A[1,3]*(A[2,1]*A[3,2]-A[2,2]*A[3,1])
    )

    A = [1 0 0; 0 2 0; 0 pi 3]
    @test det2(A) == det(A) == 6
    @test istril(A)

    ∇A = [6 0 0; 0 3 -pi; 0 0 2]
    @test ForwardDiff.gradient(det2, A) ≈ ∇A
    @test ForwardDiff.gradient(det, A) ≈ ∇A

    # And issue 407
    @test ForwardDiff.hessian(det, A) ≈ ForwardDiff.hessian(det2, A)

    # https://discourse.julialang.org/t/forwarddiff-and-zygote-return-wrong-jacobian-for-log-det-l/77961
    S = [1.0 0.8; 0.8 1.0]
    L = cholesky(S).L
    @test ForwardDiff.gradient(L -> log(det(L)), Matrix(L)) ≈ [1.0 -1.3333333333333337; 0.0 1.666666666666667]
    @test ForwardDiff.gradient(L -> logdet(L), Matrix(L)) ≈ [1.0 -1.3333333333333337; 0.0 1.666666666666667]
end

@testset "gradient for exponential with NaNMath" begin
    @test isnan(ForwardDiff.gradient(x -> NaNMath.pow(x[1],x[1]), [NaN, 1.0])[1])
    @test ForwardDiff.gradient(x -> NaNMath.pow(x[1], x[2]), [1.0, 1.0]) == [1.0, 0.0]
    @test isnan(ForwardDiff.gradient((x) -> NaNMath.pow(x[1], x[2]), [-1.0, 0.5])[1])

    @test isnan(ForwardDiff.gradient(x -> x[1]^x[2], [NaN, 1.0])[1])
    @test ForwardDiff.gradient(x -> x[1]^x[2], [1.0, 1.0]) == [1.0, 0.0]
    @test_throws DomainError ForwardDiff.gradient(x -> x[1]^x[2], [-1.0, 0.5])
end

@testset "branches in mul!" begin
    a, b = rand(3,3), rand(3,3)

    # Issue 536, version with 3-arg *, Julia 1.7:
    @test ForwardDiff.derivative(x -> sum(x*a*b), 0.0) ≈ sum(a * b)

    # version with just mul!
    dx = ForwardDiff.derivative(0.0) do x
        c = similar(a, typeof(x))
        mul!(c, a, b, x, false)
        sum(c)
    end
    @test dx ≈ sum(a * b)
end

# issue #738
@testset "LowerTriangular, UpperTriangular and Diagonal" begin
    for n in (3, 10, 20)
        M = rand(n, n)
        for T in (LowerTriangular, UpperTriangular, Diagonal)
            @test ForwardDiff.gradient(sum, T(randn(n, n))) == T(ones(n, n))
            @test ForwardDiff.gradient(x -> dot(M, x), T(randn(n, n))) == T(M)

            # Check number of function evaluations and chunk sizes
            fevals = Ref(0)
            npartials = Ref(0)
            y = ForwardDiff.gradient(T(randn(n, n))) do x
                fevals[] += 1
                npartials[] += ForwardDiff.npartials(eltype(x))
                return sum(x)
            end
            if npartials[] <= ForwardDiff.DEFAULT_CHUNK_THRESHOLD
                # Vector mode (single evaluation)
                @test fevals[] == 1
                @test npartials[] == sum(y)
            else
                # Chunk mode (multiple evaluations)
                @test fevals[] > 1
                @test sum(y) <= npartials[] < sum(y) + fevals[]
            end
        end
    end
end

end # module

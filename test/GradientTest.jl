module GradientTest

import Calculus
import LinearAlgebra
import NaNMath

using Test
using LinearAlgebra
using ForwardDiff
using ForwardDiff: Dual, Tag
using StaticArrays
using SparseArrays: sparse
using DiffTests

include(joinpath(dirname(@__FILE__), "utils.jl"))

struct TestTag end
struct OuterTestTag end
ForwardDiff.:≺(::Type{TestTag}, ::Type{OuterTestTag}) = true
ForwardDiff.:≺(::Type{OuterTestTag}, ::Type{<:Tag}) = true

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
    if ForwardDiff.NANSAFE_MODE_ENABLED
        @test isequal(ForwardDiff.gradient(t -> t[1]^t[2], [0.0, -0.5]), [-Inf, -Inf])
        @test isequal(ForwardDiff.gradient(t -> t[1]^t[2], [0.0,  0.0]), [NaN, -Inf])
        @test isequal(ForwardDiff.gradient(t -> t[1]^t[2], [0.0,  0.5]), [Inf, 0.0])
    else
        @test isequal(ForwardDiff.gradient(t -> t[1]^t[2], [0.0, -0.5]), [NaN, NaN])
        @test isequal(ForwardDiff.gradient(t -> t[1]^t[2], [0.0,  0.0]), [NaN, NaN])
        @test isequal(ForwardDiff.gradient(t -> t[1]^t[2], [0.0,  0.5]), [Inf, NaN])
    end
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
    if ForwardDiff.NANSAFE_MODE_ENABLED
        @test isequal(ForwardDiff.gradient(x -> NaNMath.pow(x[1],x[2]), [NaN, 1.0]), [1.0, NaN])
    else
        @test isequal(ForwardDiff.gradient(x -> NaNMath.pow(x[1],x[2]), [NaN, 1.0]), [NaN, NaN])
    end
    @test ForwardDiff.gradient(x -> NaNMath.pow(x[1], x[2]), [1.0, 1.0]) == [1.0, 0.0]
    @test isnan(ForwardDiff.gradient((x) -> NaNMath.pow(x[1], x[2]), [-1.0, 0.5])[1])

    if ForwardDiff.NANSAFE_MODE_ENABLED
        @test isequal(ForwardDiff.gradient(x -> x[1]^x[2], [NaN, 1.0]), [1.0, NaN])
    else
        @test isequal(ForwardDiff.gradient(x -> x[1]^x[2], [NaN, 1.0]), [NaN, NaN])
    end
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

# issue #769
@testset "functions with `Dual` output" begin
    x = [Dual{OuterTestTag}(Dual{TestTag}(1.3, 2.1), Dual{TestTag}(0.3, -2.4))]
    f(x) = sum(ForwardDiff.value, x)
    der = ForwardDiff.derivative(ForwardDiff.value, only(x))

    # Vector mode
    grad = ForwardDiff.gradient(f, x)
    @test grad isa Vector{typeof(der)}
    @test grad == [der]
    grad = ForwardDiff.gradient(f, SVector{1}(x))
    @test grad isa SVector{1,typeof(der)}
    @test grad == SVector{1}(der)

    # Chunk mode
    y = repeat(x, 3)
    cfg = ForwardDiff.GradientConfig(f, y, ForwardDiff.Chunk{2}())
    grad = ForwardDiff.gradient(f, y, cfg)
    @test grad isa Vector{typeof(der)}
    @test grad == [der, der, der]
    cfg = ForwardDiff.GradientConfig(f, SVector{3}(y), ForwardDiff.Chunk{2}())
    grad = ForwardDiff.gradient(f, SVector{3}(y), cfg)
    @test grad isa SVector{3,typeof(der)}
    @test grad == SVector{3}(der, der, der)
end

@testset "NaN-safe mode" begin
    # issue #774
    f = x -> log(zero(x[1]) + x[2])
    x = [1.0, 0.0]
    y1 = ForwardDiff.gradient(f, x)
    y2 = ForwardDiff.gradient(f, x, ForwardDiff.GradientConfig(f, x, ForwardDiff.Chunk{1}()))
    y3 = ForwardDiff.gradient(f, x, ForwardDiff.GradientConfig(f, x, ForwardDiff.Chunk{2}()))
    for y in (y1, y2, y3)
        if ForwardDiff.NANSAFE_MODE_ENABLED
            @test y == [0.0, Inf]
        else
            @test isequal(y, [NaN, Inf])
        end
    end

    # issue #745
    g = a -> a[1] * exp(-a[2])
    a = [1.0, -1e3]
    b1 = ForwardDiff.gradient(g, a)
    b2 = ForwardDiff.gradient(g, a, ForwardDiff.GradientConfig(g, a, ForwardDiff.Chunk{1}()))
    b3 = ForwardDiff.gradient(g, a, ForwardDiff.GradientConfig(g, a, ForwardDiff.Chunk{2}()))
    for b in (b1, b2, b3)
        if ForwardDiff.NANSAFE_MODE_ENABLED
            @test b == [Inf, -Inf]
        else
            @test isequal(b, [NaN, NaN])
        end
    end
end

@testset "Givens rotations: Gradients" begin
    # Test different branches in `LinearAlgebra.givensAlgorithm`
    for f in [randexp(), -randexp()], g in [0.0, f / 2, 2f, -f / 2, -2f], i in 1:3
        # Gradients wrt to a single input argument
        dydf = only(ForwardDiff.gradient(x -> LinearAlgebra.givensAlgorithm(only(x), g)[i], [f]))
        @test dydf == ForwardDiff.derivative(x -> LinearAlgebra.givensAlgorithm(x, g)[i], f)
        dydg = only(ForwardDiff.gradient(x -> LinearAlgebra.givensAlgorithm(f, only(x))[i], [g]))
        @test dydg == ForwardDiff.derivative(x -> LinearAlgebra.givensAlgorithm(f, x)[i], g)

        # Gradient with respect to both input arguments
        grad = ForwardDiff.gradient(x -> LinearAlgebra.givensAlgorithm(x[1], x[2])[i], [f, g])
        @test grad == [dydf, dydg]
        @test grad ≈ Calculus.gradient(x -> LinearAlgebra.givensAlgorithm(x[1], x[2])[i], [f, g])
    end
end

@testset "norm" begin
    # issues #243 and #794
    for x in ([0.0, 0.0], SVector(0.0, 0.0)), p in (2, 3, 1.5, 1, 0, Inf, -Inf)
        @test ForwardDiff.gradient(x -> norm(x, p), x) == [0.0, 0.0]
        @test ForwardDiff.gradient(x -> norm(x, p)^2, x) == [0.0, 0.0]
        @test ForwardDiff.gradient(x -> norm(Tuple(x), p), x) == [0.0, 0.0]
    end
    @test norm([Dual(0.0, 1.0), Dual(0.0, 1.0)]) === Dual(0.0, 0.0)
    @test norm(SVector(Dual(0.0, 1.0), Dual(0.0, 1.0), Dual(0.0, 1.0))) === Dual(0.0, 0.0)
    @test norm((Dual(0.0, 1.0), Dual(0.0, 1.0))) === Dual(0.0, 0.0)
    @test norm((Dual(0.0, 1.0), complex(Dual(0.0, 1.0), Dual(0.0, 1.0))), 3) === Dual(0.0, 0.0)
    @test norm(ForwardDiff.partials(Dual(Dual(1.0, 0.0), Dual(0.0, 1.0), Dual(0.0, 1.0)))) === Dual(0.0, 0.0)
    @test norm([Dual(1e-200, 1e-100), Dual(0.0, 0.0)]) === Dual(1e-200, 1e-100)
    for x in ([Dual(0.0, 1.0), Dual(1.0, 0.0)], SVector(Dual(0.0, 1.0), Dual(1.0, 0.0)), (Dual(0.0, 1.0), Dual(1.0, 0.0)))
        @test norm(x, 0) === Dual(1.0, 0.0)
    end

    # Infinite entries yield `NaN` partials, unless NaN-safe mode is enabled
    for x in ([Dual(Inf, 0.0), Dual(1.0, 1.0)], (Dual(Inf, 0.0), Dual(1.0, 1.0))), p in (2, 3, Inf)
        if ForwardDiff.NANSAFE_MODE_ENABLED
            @test norm(x, p) === Dual(Inf, 0.0)
        else
            @test isequal(norm(x, p), Dual(Inf, NaN))
        end
    end

    # Same precision as the primal norm
    for x in ([Dual(1f0, 1f0), Dual(-2f0, 0f0)], SVector(Dual(1f0, 1f0), Dual(-2f0, 0f0)), (Dual(1f0, 1f0), Dual(-2f0, 0f0))), p in (2, 3, 2.5, Inf)
        y = norm(map(ForwardDiff.value, x), p)
        @test norm(x, p) isa Dual{Nothing,typeof(y),1}
        if !(x isa StaticArray)
            @test @inferred(norm(x, p)) isa Dual{Nothing,Float32,1}
        end
    end
    for s in (1e-200, 1e200)
        @test ForwardDiff.gradient(norm, [s, s]) ≈ [√0.5, √0.5]
        @test ForwardDiff.gradient(norm, SVector(s, s)) ≈ [√0.5, √0.5]
    end
    @test ForwardDiff.gradient(x -> norm(x, 1.01), [5e-324, 1.0]) ≈ [5e-324^0.01, 1.0]

    # `p` is a `Dual`
    ref = (x, p) -> sum(t -> abs(t)^p, x)^(1 / p)
    x = [3.0, -4.0, 0.5]
    for p in (3.0, 1.5, 0.5)
        dref = ForwardDiff.derivative(p -> ref(x, p), p)
        @test ForwardDiff.derivative(p -> norm(x, p), p) ≈ dref
        @test ForwardDiff.derivative(p -> norm(SVector{3}(x), p), p) ≈ dref
        @test ForwardDiff.derivative(p -> norm(x', p), p) ≈ dref
        @test ForwardDiff.derivative(p -> norm(sparse(x), p), p) ≈ dref
        @test ForwardDiff.derivative(p -> norm(sparse(reshape(x, 3, 1)), p), p) ≈ dref
        @test ForwardDiff.derivative(p -> norm(Tuple(x), p), p) ≈ dref
        @test ForwardDiff.derivative(p -> norm(complex.(x, x), p), p) ≈ ForwardDiff.derivative(p -> ref(complex.(x, x), p), p)
        ∇ref = ForwardDiff.gradient(v -> ref(v[1:3], v[4]), [x; p])
        @test ForwardDiff.gradient(v -> norm(v[1:3], v[4]), [x; p]) ≈ ∇ref
        @test ForwardDiff.gradient(v -> norm(SVector{3}(v[1], v[2], v[3]), v[4]), [x; p]) ≈ ∇ref
        @test ForwardDiff.gradient(v -> norm(v[1:3]', v[4]), [x; p]) ≈ ∇ref
        @test ForwardDiff.gradient(v -> norm(sparse(v[1:3]), v[4]), [x; p]) ≈ ∇ref
        @test ForwardDiff.gradient(v -> norm(sparse(reshape(v[1:3], 3, 1)), v[4]), [x; p]) ≈ ∇ref
        @test ForwardDiff.gradient(v -> norm((v[1], v[2], v[3]), v[4]), [x; p]) ≈ ∇ref
        @test ForwardDiff.derivative(p -> ForwardDiff.gradient(x -> norm(x, p), x), p) ≈
            ForwardDiff.derivative(p -> ForwardDiff.gradient(x -> ref(x, p), x), p)
        @test ForwardDiff.gradient(x -> ForwardDiff.derivative(p -> norm(x, p), p), x) ≈
            ForwardDiff.gradient(x -> ForwardDiff.derivative(p -> ref(x, p), p), x)
    end
    for p in (1.0, 2.0, 3.0)
        Href = ForwardDiff.hessian(v -> ref(v[1:3], v[4]), [x; p])
        @test ForwardDiff.hessian(v -> norm(v[1:3], v[4]), [x; p]) ≈ Href
        @test ForwardDiff.hessian(v -> norm(v[SOneTo(3)], v[4]), SVector{4}([x; p])) ≈ Href
        @test ForwardDiff.hessian(v -> norm((v[1], v[2], v[3]), v[4]), [x; p]) ≈ Href
    end
    @test ForwardDiff.derivative(p -> ForwardDiff.gradient(x -> norm(x, p), x), Inf) == zeros(3)
    @test ForwardDiff.derivative(p -> norm([0.0, 0.0], p), 3.0) == 0
    @test ForwardDiff.derivative(p -> norm([1e200, 1e200], p), 3.0) ≈ 1e200 * ForwardDiff.derivative(p -> ref([1.0, 1.0], p), 3.0)

    x = [1.0, -2.0, 0.0, 3.0]
    z = v -> complex.(v[1:2], v[3:4])
    for p in (2, 3, 1.5, Inf)
        f = x -> norm(x, p)
        g = p == Inf ? (x -> maximum(abs, x)) : (x -> sum(t -> abs(t)^p, x)^(1 / p))
        ∇g = ForwardDiff.gradient(g, x)
        for c in (1, 4)
            @test ForwardDiff.gradient(f, x, ForwardDiff.GradientConfig(f, x, ForwardDiff.Chunk{c}())) ≈ ∇g
        end
        @test ForwardDiff.gradient(f, SVector{4}(x)) ≈ ∇g
        @test ForwardDiff.gradient(f ∘ Tuple, x) ≈ ∇g
        @test ForwardDiff.gradient(f ∘ z, x) ≈ ForwardDiff.gradient(g ∘ z, x)
    end
    # Subgradient zero for zero entries, as for other `p`
    @test ForwardDiff.gradient(x -> norm(x, 1), x) == sign.(x)
    @test ForwardDiff.gradient(x -> norm(x, 1), SVector{4}(x)) == sign.(x)
    @test ForwardDiff.gradient(x -> norm(z(x), 1), x) ≈ ForwardDiff.gradient(x -> sum(abs, z(x)), x)
    @test ForwardDiff.gradient(x -> norm(x, -Inf), [3.0, -1.0, 2.0]) == [0.0, -1.0, 0.0]

    u = x / norm(x)
    @test ForwardDiff.hessian(norm, x) ≈ (I - u * u') / norm(x)
    @test ForwardDiff.hessian(norm, SVector{4}(x)) ≈ (I - u * u') / norm(x)
    @test ForwardDiff.hessian(x -> norm(x, 3), x) ≈ ForwardDiff.hessian(x -> cbrt(sum(t -> abs(t)^3, x)), x)

    d = [Dual(3.0, 1.0), Dual(-4.0, 2.0)]
    for p in (2, 3, 1, Inf)
        y = norm(d, p)
        @test norm(d', p) === y
        @test norm(transpose(d), p) === y
        @test norm(sparse(d), p) === y
        @test norm(sparse(reshape(d, 2, 1)), p) === y
        @test norm(Tuple(d), p) === y
    end
end

end # module

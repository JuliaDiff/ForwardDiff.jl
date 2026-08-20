module HessianTest

import Calculus

using Test
using LinearAlgebra
using ForwardDiff
using ForwardDiff: Dual, Tag
using StaticArrays
using DiffTests

include(joinpath(dirname(@__FILE__), "utils.jl"))

#############################
# rosenbrock hardcoded test #
#############################

f = DiffTests.rosenbrock_1
x = [0.1, 0.2, 0.3]
v = f(x)
g = [-9.4, 15.6, 52.0]
h = [-66.0  -40.0    0.0;
     -40.0  130.0  -80.0;
       0.0  -80.0  200.0]

@testset "running hardcoded test with chunk size = $c and tag = $(repr(tag))" for c in HESSIAN_CHUNK_SIZES, tag in (nothing, Tag((f,ForwardDiff.gradient), eltype(x)))
    cfg = ForwardDiff.HessianConfig(f, x, ForwardDiff.Chunk{c}(), tag)
    resultcfg = ForwardDiff.HessianConfig(f, DiffResults.HessianResult(x), x, ForwardDiff.Chunk{c}(), tag)

    @test eltype(resultcfg) == eltype(cfg)

    @test isapprox(h, ForwardDiff.hessian(f, x))
    @test isapprox(h, ForwardDiff.hessian(f, x, cfg))

    out = similar(x, 3, 3)
    ForwardDiff.hessian!(out, f, x)
    @test isapprox(out, h)

    out = similar(x, 3, 3)
    ForwardDiff.hessian!(out, f, x, cfg)
    @test isapprox(out, h)

    out = DiffResults.HessianResult(x)
    ForwardDiff.hessian!(out, f, x)
    @test isapprox(DiffResults.value(out), v)
    @test isapprox(DiffResults.gradient(out), g)
    @test isapprox(DiffResults.hessian(out), h)

    out = DiffResults.HessianResult(x)
    ForwardDiff.hessian!(out, f, x, resultcfg)
    @test isapprox(DiffResults.value(out), v)
    @test isapprox(DiffResults.gradient(out), g)
    @test isapprox(DiffResults.hessian(out), h)
end

cfgx = ForwardDiff.HessianConfig(sin, x)
@test_throws ForwardDiff.InvalidTagException ForwardDiff.hessian(f, x, cfgx)
@test ForwardDiff.hessian(f, x, cfgx, Val{false}()) == ForwardDiff.hessian(f,x)


########################
# test vs. Calculus.jl #
########################

for f in DiffTests.VECTOR_TO_NUMBER_FUNCS
    v = f(X)
    g = ForwardDiff.gradient(f, X)
    h = ForwardDiff.hessian(f, X)
    # finite difference approximation error is really bad for Hessians...
    @test isapprox(h, Calculus.hessian(f, X), atol=0.02)
    @testset "$f with chunk size = $c and tag = $(repr(tag))" for c in HESSIAN_CHUNK_SIZES, tag in (nothing, Tag((f,ForwardDiff.gradient), eltype(x)))
        cfg = ForwardDiff.HessianConfig(f, X, ForwardDiff.Chunk{c}(), tag)
        resultcfg = ForwardDiff.HessianConfig(f, DiffResults.HessianResult(X), X, ForwardDiff.Chunk{c}(), tag)

        out = ForwardDiff.hessian(f, X, cfg)
        @test isapprox(out, h)

        out = similar(X, length(X), length(X))
        ForwardDiff.hessian!(out, f, X, cfg)
        @test isapprox(out, h)

        out = DiffResults.HessianResult(X)
        ForwardDiff.hessian!(out, f, X, resultcfg)
        @test isapprox(DiffResults.value(out), v)
        @test isapprox(DiffResults.gradient(out), g)
        @test isapprox(DiffResults.hessian(out), h)
    end
end

##########################################
# test specialized StaticArray codepaths #
##########################################

@info "testing specialized StaticArray codepaths"

x = rand(3, 3)
for T in (StaticArrays.SArray, StaticArrays.MArray)
    sx = T{Tuple{3,3}}(x)

    cfg = ForwardDiff.HessianConfig(nothing, x)
    scfg = ForwardDiff.HessianConfig(nothing, sx)

    actual = ForwardDiff.hessian(prod, x)
    @test ForwardDiff.hessian(prod, sx) == actual
    @test ForwardDiff.hessian(prod, sx, cfg) == actual
    @test ForwardDiff.hessian(prod, sx, scfg) == actual
    @test ForwardDiff.hessian(prod, sx, scfg) isa StaticArray
    @test ForwardDiff.hessian(prod, sx, scfg, Val{false}()) == actual
    @test ForwardDiff.hessian(prod, sx, scfg, Val{false}()) isa StaticArray

    out = similar(x, 9, 9)
    ForwardDiff.hessian!(out, prod, sx)
    @test out == actual

    out = similar(x, 9, 9)
    ForwardDiff.hessian!(out, prod, sx, cfg)
    @test out == actual

    out = similar(x, 9, 9)
    ForwardDiff.hessian!(out, prod, sx, scfg)
    @test out == actual

    result = DiffResults.HessianResult(x)
    result = ForwardDiff.hessian!(result, prod, x)

    result1 = DiffResults.HessianResult(x)
    result2 = DiffResults.HessianResult(x)
    result3 = DiffResults.HessianResult(x)
    result1 = ForwardDiff.hessian!(result1, prod, sx)
    result2 = ForwardDiff.hessian!(result2, prod, sx, ForwardDiff.HessianConfig(prod, result2, x, ForwardDiff.Chunk(x), nothing))
    result3 = ForwardDiff.hessian!(result3, prod, sx, ForwardDiff.HessianConfig(prod, result3, x, ForwardDiff.Chunk(x), nothing))
    @test DiffResults.value(result1) == DiffResults.value(result)
    @test DiffResults.value(result2) == DiffResults.value(result)
    @test DiffResults.value(result3) == DiffResults.value(result)
    @test DiffResults.gradient(result1) == DiffResults.gradient(result)
    @test DiffResults.gradient(result2) == DiffResults.gradient(result)
    @test DiffResults.gradient(result3) == DiffResults.gradient(result)
    @test DiffResults.hessian(result1) == DiffResults.hessian(result)
    @test DiffResults.hessian(result2) == DiffResults.hessian(result)
    @test DiffResults.hessian(result3) == DiffResults.hessian(result)

    sresult1 = DiffResults.HessianResult(sx)
    sresult2 = DiffResults.HessianResult(sx)
    sresult3 = DiffResults.HessianResult(sx)
    sresult1 = ForwardDiff.hessian!(sresult1, prod, sx)
    sresult2 = ForwardDiff.hessian!(sresult2, prod, sx, ForwardDiff.HessianConfig(prod, sresult2, x, ForwardDiff.Chunk(x), nothing))
    sresult3 = ForwardDiff.hessian!(sresult3, prod, sx, ForwardDiff.HessianConfig(prod, sresult3, x, ForwardDiff.Chunk(x), nothing))
    @test DiffResults.value(sresult1) == DiffResults.value(result)
    @test DiffResults.value(sresult2) == DiffResults.value(result)
    @test DiffResults.value(sresult3) == DiffResults.value(result)
    @test DiffResults.gradient(sresult1) == DiffResults.gradient(result)
    @test DiffResults.gradient(sresult2) == DiffResults.gradient(result)
    @test DiffResults.gradient(sresult3) == DiffResults.gradient(result)
    @test DiffResults.hessian(sresult1) == DiffResults.hessian(result)
    @test DiffResults.hessian(sresult2) == DiffResults.hessian(result)
    @test DiffResults.hessian(sresult3) == DiffResults.hessian(result)
end

# issues #838 and #839, which `hessian` inherits through `jacobian(gradient(f), x)`
@testset "structured inputs: $(nameof(W))" for (W, sidx) in (
        # both axes are indexed by the linear indices of `x`, hard zeros off the structure
        (LowerTriangular, [i + 3 * (j - 1) for j in 1:3 for i in j:3]),
        (UpperTriangular, [i + 3 * (j - 1) for j in 1:3 for i in 1:j]),
        (Diagonal,        1:4:9),
    )
    x = W(randn(3, 3))
    # d²f/dx[a]dx[b] is `1 + (a == b)` for structural `a`, `b`, and zero everywhere else
    f = z -> (sum(abs2, z) + sum(z)^2) / 2
    L = length(x)

    expected = zeros(L, L)
    expected[sidx, sidx] .= 1
    for k in sidx
        expected[k, k] += 1
    end
    val = f(x)
    grad = zeros(3, 3)
    grad[sidx] .= x[sidx] .+ sum(x)

    # one chunk size below the full length, so that the final chunk is a partial one
    @testset "chunk size = $c" for c in unique((1, 2, length(sidx) - 1, length(sidx)))
        cfg = ForwardDiff.HessianConfig(f, x, ForwardDiff.Chunk{c}())

        H = ForwardDiff.hessian(f, x, cfg)
        @test size(H) == (L, L)
        @test H == expected

        out = fill(NaN, L, L)
        @test ForwardDiff.hessian!(out, f, x, cfg) === out
        @test out == expected

        # `DiffResults.HessianResult` allocates a dense gradient buffer even for a structured `x`
        result = DiffResults.HessianResult(x)
        result = ForwardDiff.hessian!(result, f, x,
                                      ForwardDiff.HessianConfig(f, result, x, ForwardDiff.Chunk{c}()))
        @test DiffResults.value(result) ≈ val
        @test DiffResults.gradient(result) == grad
        @test DiffResults.hessian(result) == expected
    end
end

# issue #842, which `hessian` inherits through both of its sub-configs
@testset "config reused for a differently structured input" begin
    n = 3
    A = randn(n, n)
    inputs = (A, LowerTriangular(A), UpperTriangular(A), Diagonal(diag(A)))
    f = z -> (sum(abs2, z) + sum(z)^2) / 2

    @testset "chunk size = $c" for c in (2, n)
        @testset "$(nameof(typeof(xcfg))) config" for xcfg in inputs
            cfg = ForwardDiff.HessianConfig(f, xcfg, ForwardDiff.Chunk{c}())
            # each input has a different structure, so `x === xcfg` is the one the config fits
            for x in inputs
                x === xcfg && continue
                msg = "ArgumentError: the config was built for an array of type " *
                      "$(nameof(typeof(xcfg))) and cannot be used with an array of type " *
                      "$(nameof(typeof(x)))"
                @test_throws msg ForwardDiff.hessian(f, x, cfg)
                @test_throws msg ForwardDiff.hessian!(fill(NaN, n^2, n^2), f, x, cfg)
            end
        end
    end
end

@testset "branches in dot" begin
    # https://github.com/JuliaDiff/ForwardDiff.jl/issues/551
    H = [1 2 3; 4 5 6; 7 8 9];
    @test ForwardDiff.hessian(x->dot(x,H,x), fill(0.00001, 3)) ≈ [2 6 10; 6 10 14; 10 14 18]
    @test ForwardDiff.hessian(x->dot(x,H,x), zeros(3)) ≈ [2 6 10; 6 10 14; 10 14 18]
end

#https://github.com/JuliaDiff/ForwardDiff.jl/issues/720
@testset "allocation-free hessian with StaticArrays" begin
    function hessian_allocs()
        g = r -> (r[1]^2 - 3) * (r[2]^2 - 2)
        x = SVector(0.5, 2.8)
        hres = DiffResults.HessianResult(x)
        return @allocated(ForwardDiff.hessian!(hres, g, x))
    end
    @test iszero(hessian_allocs())
end

end # module

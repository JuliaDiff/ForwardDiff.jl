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

    # The result-aware and result-independent config constructors are interchangeable.
    out = DiffResults.HessianResult(x)
    ForwardDiff.hessian!(out, f, x, cfg)
    @test isapprox(DiffResults.value(out), v)
    @test isapprox(DiffResults.gradient(out), g)
    @test isapprox(DiffResults.hessian(out), h)
end

cfgx = ForwardDiff.HessianConfig(sin, x)
@test_throws ForwardDiff.InvalidTagException ForwardDiff.hessian(f, x, cfgx)
@test ForwardDiff.hessian(f, x, cfgx, Val{false}()) == ForwardDiff.hessian(f,x)
@test_throws ArgumentError ForwardDiff.hessian(f, x, ForwardDiff.HessianConfig(f, x, ForwardDiff.Chunk{length(x) + 1}()))
@test_throws DimensionMismatch ForwardDiff.hessian(identity, x)
@test_throws DimensionMismatch ForwardDiff.hessian!(similar(x, 3, 3), identity, x)


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

    symmetry_f(z) = sum(sin(z[i]) / (1 + z[mod1(i + 1, length(z))]^2) for i in eachindex(z))
    symmetric_static = ForwardDiff.hessian(symmetry_f, sx)
    @test symmetric_static == transpose(symmetric_static)
    @test symmetric_static == ForwardDiff.hessian(symmetry_f, x)
    @test all(iszero, ForwardDiff.hessian(Returns(2.0), sx))
    @test_throws DimensionMismatch ForwardDiff.hessian(identity, sx)

    out = similar(x, 9, 9)
    ForwardDiff.hessian!(out, prod, sx)
    @test out == actual

    out = similar(x, 9, 9)
    ForwardDiff.hessian!(out, symmetry_f, sx)
    @test out == symmetric_static
    @test out == transpose(out)

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

        # a result that is not a matrix is reshaped, so it only has to match in length
        flat = fill(NaN, L^2)
        @test ForwardDiff.hessian!(flat, f, x, cfg) === flat
        @test reshape(flat, L, L) == expected

        # a wrongly shaped result is rejected rather than partially filled, the structural shape
        # this used to have included
        @test_throws DimensionMismatch ForwardDiff.hessian!(fill(NaN, L + 1, L + 1), f, x, cfg)
        @test_throws DimensionMismatch ForwardDiff.hessian!(fill(NaN, length(sidx), length(sidx)), f, x, cfg)

        # `DiffResults.HessianResult` allocates a dense gradient buffer even for a structured `x`
        result = DiffResults.HessianResult(x)
        result = ForwardDiff.hessian!(result, f, x,
                                      ForwardDiff.HessianConfig(f, result, x, ForwardDiff.Chunk{c}()))
        @test DiffResults.value(result) ≈ val
        @test DiffResults.gradient(result) == grad
        @test DiffResults.hessian(result) == expected
    end
end

@testset "chunk size independence" begin
    # Every block of the sweep reads the nested partials at the same nesting order, so the result is
    # not merely close across chunk sizes but bitwise equal, and equal to the vector mode of the
    # `StaticArray` path, which reads them that way too.
    n = 16
    f = z -> sum(sin(z[i]) * exp(z[mod1(i + 1, n)]) / (1 + z[mod1(i + 2, n)]^2) for i in eachindex(z))
    x = collect(range(0.1, 0.9; length=n))
    reference = ForwardDiff.hessian(f, SVector{n}(x))
    @test reference == transpose(reference)
    @testset "chunk size = $c" for c in (1, 2, 3, 5, 7, 11, n)
        H = ForwardDiff.hessian(f, x, ForwardDiff.HessianConfig(f, x, ForwardDiff.Chunk{c}()))
        @test H == reference
    end

    # A structured `x` reads its positions block by block rather than in one pass, and the coupled
    # `f` of the structured testset above has an exactly representable Hessian, so its `==` says
    # nothing about floating point. There is no `StaticArray` wrapper to compare against, so the
    # largest chunk size is the reference.
    g = z -> sum(sin(z[i]) * exp(z[j]) / (1 + z[k]^2)
                 for i in eachindex(z), j in eachindex(z), k in eachindex(z) if i <= j <= k)
    @testset "$(nameof(typeof(w)))" for w in (LowerTriangular(rand(4, 4) .+ 1),
                                              UpperTriangular(rand(4, 4) .+ 1),
                                              Diagonal(rand(4) .+ 1))
        nstruct = ForwardDiff.structural_length(w)
        reference = ForwardDiff.hessian(g, w, ForwardDiff.HessianConfig(g, w, ForwardDiff.Chunk{nstruct}()))
        @test reference == transpose(reference)
        @testset "chunk size = $c" for c in 1:(nstruct - 1)
            @test ForwardDiff.hessian(g, w, ForwardDiff.HessianConfig(g, w, ForwardDiff.Chunk{c}())) == reference
        end
    end
end

@testset "wrongly shaped result" begin
    # A matrix result is written to as is, so it has to have the shape of the Hessian and not merely
    # as many entries. Results of other shapes are reshaped and only have to match in length.
    x = randn(3)
    f = z -> sum(abs2, z)
    sx = SVector{3}(x)
    @testset "chunk size = $c" for c in (2, 3)
        cfg = ForwardDiff.HessianConfig(f, x, ForwardDiff.Chunk{c}())
        @test_throws DimensionMismatch ForwardDiff.hessian!(fill(NaN, 4, 4), f, x, cfg)
        @test_throws DimensionMismatch ForwardDiff.hessian!(fill(NaN, 10), f, x, cfg)
        result = DiffResults.DiffResult(NaN, fill(NaN, 3), fill(NaN, 4, 4))
        @test_throws DimensionMismatch ForwardDiff.hessian!(result, f, x, cfg)
        # a `DiffResult` holding a Hessian that is not a matrix is reshaped like a bare one
        result = DiffResults.DiffResult(NaN, fill(NaN, 3), fill(NaN, 9))
        result = ForwardDiff.hessian!(result, f, x, cfg)
        @test reshape(DiffResults.hessian(result), 3, 3) == 2I(3)
    end
    # the `StaticArray` methods bypass the sweep and reshape the result themselves
    @test_throws DimensionMismatch ForwardDiff.hessian!(fill(NaN, 4, 4), f, sx)
    @test_throws DimensionMismatch ForwardDiff.hessian!(fill(NaN, 10), f, sx)
    # a result that is not a matrix goes through `reshape`
    flat = fill(NaN, 9)
    @test ForwardDiff.hessian!(flat, f, sx) === flat
    @test reshape(flat, 3, 3) == 2I(3)
end

@testset "empty input" begin
    f = z -> 1.0 + sum(z)
    @test ForwardDiff.hessian(f, Float64[]) == zeros(0, 0)
    @test ForwardDiff.hessian(f, SVector{0,Float64}()) == zeros(0, 0)
    # `DiffResults.HessianResult` cannot be built from an empty `x`, so assemble one by hand
    result = DiffResults.DiffResult(NaN, Float64[], Matrix{Float64}(undef, 0, 0))
    result = ForwardDiff.hessian!(result, f, Float64[])
    @test DiffResults.value(result) == 1.0
    @test isempty(DiffResults.hessian(result))
end

@testset "f(x) is not a real number" begin
    x = randn(3)
    sx = SVector{3}(x)
    @test_throws DimensionMismatch ForwardDiff.hessian!(fill(NaN, 3, 3), identity, sx)
    @test_throws DimensionMismatch ForwardDiff.hessian!(DiffResults.HessianResult(sx), identity, sx)
    # the sweep checks the first evaluation itself, before it has a result to write into
    @testset "chunk size = $c" for c in (2, 3)
        cfg = ForwardDiff.HessianConfig(identity, x, ForwardDiff.Chunk{c}())
        @test_throws DimensionMismatch ForwardDiff.hessian(identity, x, cfg)
        @test_throws DimensionMismatch ForwardDiff.hessian!(fill(NaN, 3, 3), identity, x, cfg)
        @test_throws DimensionMismatch ForwardDiff.hessian!(DiffResults.HessianResult(x), identity, x, cfg)
    end
end

@testset "f(x) ignores x" begin
    # `partials` is empty, the second case the `Partials{0}` `StaticArray` method covers
    f = z -> 1.0
    @test ForwardDiff.hessian(f, SVector{3}(randn(3))) == zeros(3, 3)
    @testset "chunk size = $c" for c in (2, 3)
        x = randn(3)
        @test ForwardDiff.hessian(f, x, ForwardDiff.HessianConfig(f, x, ForwardDiff.Chunk{c}())) == zeros(3, 3)
    end
end

@testset "BigFloat with an unassigned input entry" begin
    x = Vector{BigFloat}(undef, 10)
    hole = 5
    for i in eachindex(x)
        i == hole || (x[i] = BigFloat(i))
    end
    used = [i for i in eachindex(x) if i != hole]
    f(x) = sum(abs2(x[i]) for i in used)
    expected = zeros(BigFloat, 10, 10)
    for i in used
        expected[i, i] = 2
    end

    @test !isassigned(x, hole)
    for chunksize in (1, 2, 10)
        cfg = ForwardDiff.HessianConfig(f, x, ForwardDiff.Chunk{chunksize}())
        H = ForwardDiff.hessian(f, x, cfg)
        @test H isa Matrix{BigFloat}
        @test H == expected
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

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

@testset "empty input" begin
    f = z -> 1.0
    for x in (Float64[], SVector{0,Float64}())
        H = ForwardDiff.hessian(f, x)
        @test size(H) == (0, 0)
        @test ForwardDiff.hessian!(fill(NaN, 0, 0), f, x) == H
    end
end

@testset "a StaticArray result that is not a matrix" begin
    sx = SVector(1.0, 2.0, 3.0)
    flat = fill(NaN, 9)
    @test ForwardDiff.hessian!(flat, prod, sx) === flat
    @test reshape(flat, 3, 3) == ForwardDiff.hessian(prod, sx)
end

@testset "an array-valued f is not a Hessian" begin
    sx = SVector(1.0, 2.0, 3.0)
    @test_throws DimensionMismatch ForwardDiff.hessian(identity, sx)
    @test_throws DimensionMismatch ForwardDiff.hessian!(fill(NaN, 3, 3), identity, sx)
    @test_throws DimensionMismatch ForwardDiff.hessian!(DiffResults.HessianResult(sx), identity, sx)
    @test_throws DimensionMismatch ForwardDiff.hessian(identity, [1.0, 2.0, 3.0])
end

@testset "the result does not depend on the chunk size" begin
    n = 16
    v = randn(n)
    # the mixed partials of `log(sum(exp, ·))` round differently in the two orders; `sum(z)^3`,
    # `exp(sum(z))`, `prod(z)` and `sum(sin, z) * sum(cos, z)` do not, and would pass regardless
    f = z -> log(sum(exp, z))
    expected = Matrix(ForwardDiff.hessian(f, SVector{n}(v)))

    @testset "chunk size = $c" for c in (1, 2, 3, 5, 7, 11, n)
        H = ForwardDiff.hessian(f, v, ForwardDiff.HessianConfig(f, v, ForwardDiff.Chunk{c}()))
        @test H == expected
        @test H == transpose(H)
    end
end

@testset "LowerTriangular, UpperTriangular and Diagonal" begin
    for n in (3, 5), T in (LowerTriangular, UpperTriangular, Diagonal)
        x = T(randn(n, n))
        xlen = ForwardDiff.structural_length(x)
        weights = reshape(collect(1.0:n^2), n, n)
        objective = x -> dot(weights, abs2.(x))
        expected = diagm(2 .* [weights[idx] for idx in ForwardDiff.structural_eachindex(x)])

        H = ForwardDiff.hessian(objective, x)
        @test size(H) == (xlen, xlen)
        @test H == expected

        out = fill(NaN, xlen, xlen)
        ForwardDiff.hessian!(out, objective, x)
        @test out == expected

        flat = fill(NaN, xlen^2)
        ForwardDiff.hessian!(flat, objective, x)
        @test reshape(flat, xlen, xlen) == expected
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

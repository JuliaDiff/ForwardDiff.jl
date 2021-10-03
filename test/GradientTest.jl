module GradientTest

import Calculus

using Test
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

for c in (1, 2, 3), tag in (nothing, Tag(f, eltype(x)))
    println("  ...running hardcoded test with chunk size = $c and tag = $(repr(tag))")
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

for f in DiffTests.VECTOR_TO_NUMBER_FUNCS
    v = f(X)
    g = ForwardDiff.gradient(f, X)
    @test isapprox(g, Calculus.gradient(f, X), atol=FINITEDIFF_ERROR)
    for c in CHUNK_SIZES, tag in (nothing, Tag(f, eltype(x)))
        println("  ...testing $f with chunk size = $c and tag = $(repr(tag))")
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

println("  ...testing specialized StaticArray codepaths")

x = rand(3, 3)

for T in (StaticArrays.SArray, StaticArrays.MArray)
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
end

@testset "exponential function at base zero" begin
    @test isequal(ForwardDiff.gradient(t -> t[1]^t[2], [0.0, -0.5]), [NaN, NaN])
    @test isequal(ForwardDiff.gradient(t -> t[1]^t[2], [0.0,  0.0]), [NaN, NaN])
    @test isequal(ForwardDiff.gradient(t -> t[1]^t[2], [0.0,  0.5]), [Inf, NaN])
    @test isequal(ForwardDiff.gradient(t -> t[1]^t[2], [0.0,  1.5]), [0.0, 0.0])
end

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

@testset "ArithmeticStyle" begin
    function f(p)
        sum(collect(0.0:p[1]:p[2]))
    end
    @test ForwardDiff.gradient(f, [0.2,25.0]) == [7875.0, 0.0]
end

end # module

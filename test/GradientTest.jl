module GradientTest

import Calculus

using Base.Test
using ForwardDiff
using ForwardDiff: Dual, Tag
using StaticArrays

include(joinpath(dirname(@__FILE__), "utils.jl"))

##################
# hardcoded test #
##################

f = DiffBase.rosenbrock_1
x = [0.1, 0.2, 0.3]
v = f(x)
g = [-9.4, 15.6, 52.0]

for c in (1, 2, 3), tag in (nothing, f)
    println("  ...running hardcoded test with chunk size = $c and tag = $tag")
    cfg = ForwardDiff.GradientConfig(tag, x, ForwardDiff.Chunk{c}())

    @test eltype(cfg) == Dual{typeof(Tag(typeof(tag), eltype(x))), eltype(x), c}

    @test isapprox(g, ForwardDiff.gradient(f, x, cfg))
    @test isapprox(g, ForwardDiff.gradient(f, x))

    out = similar(x)
    ForwardDiff.gradient!(out, f, x, cfg)
    @test isapprox(out, g)

    out = similar(x)
    ForwardDiff.gradient!(out, f, x)
    @test isapprox(out, g)

    out = DiffBase.GradientResult(x)
    ForwardDiff.gradient!(out, f, x, cfg)
    @test isapprox(DiffBase.value(out), v)
    @test isapprox(DiffBase.gradient(out), g)

    out = DiffBase.GradientResult(x)
    ForwardDiff.gradient!(out, f, x)
    @test isapprox(DiffBase.value(out), v)
end

########################
# test vs. Calculus.jl #
########################

for f in DiffBase.VECTOR_TO_NUMBER_FUNCS
    v = f(X)
    g = ForwardDiff.gradient(f, X)
    @test isapprox(g, Calculus.gradient(f, X), atol=FINITEDIFF_ERROR)
    for c in CHUNK_SIZES, tag in (nothing, f)
        println("  ...testing $f with chunk size = $c and tag = $tag")
        cfg = ForwardDiff.GradientConfig(tag, X, ForwardDiff.Chunk{c}())

        out = ForwardDiff.gradient(f, X, cfg)
        @test isapprox(out, g)

        out = similar(X)
        ForwardDiff.gradient!(out, f, X, cfg)
        @test isapprox(out, g)

        out = DiffBase.GradientResult(X)
        ForwardDiff.gradient!(out, f, X, cfg)
        @test isapprox(DiffBase.value(out), v)
        @test isapprox(DiffBase.gradient(out), g)
    end
end

##########################################
# test specialized StaticArray codepaths #
##########################################

x = rand(3, 3)
sx = StaticArrays.SArray{Tuple{3,3}}(x)
out = similar(x)
actual = ForwardDiff.gradient(prod, x)
actual_val = prod(x)

@test ForwardDiff.gradient(prod, sx) == actual
@test ForwardDiff.gradient_val(prod, sx) == (actual, actual_val)

ForwardDiff.gradient!(out, prod, sx)

@test out == actual

result = DiffBase.GradientResult(x)
sresult = DiffBase.GradientResult(sx)

result = ForwardDiff.gradient!(result, prod, x)
sresult = ForwardDiff.gradient!(sresult, prod, sx)

@test DiffBase.value(sresult) == DiffBase.value(result)
@test DiffBase.gradient(sresult) == DiffBase.gradient(result)

end # module

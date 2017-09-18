module HessianTest

import Calculus

using Base.Test
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

for c in (1, 2, 3), tag in (nothing, f)
    println("  ...running hardcoded test with chunk size = $c and tag = $tag")
    cfg = ForwardDiff.HessianConfig(tag, x, ForwardDiff.Chunk{c}())
    resultcfg = ForwardDiff.HessianConfig(tag, DiffResults.HessianResult(x), x, ForwardDiff.Chunk{c}())

    D = Dual{typeof(Tag(Void, eltype(x))), eltype(x), c}
    @test eltype(cfg) == Dual{typeof(Tag(typeof(tag), Dual{Void,eltype(x),0})), D, c}
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

########################
# test vs. Calculus.jl #
########################

for f in DiffTests.VECTOR_TO_NUMBER_FUNCS
    v = f(X)
    g = ForwardDiff.gradient(f, X)
    h = ForwardDiff.hessian(f, X)
    # finite difference approximation error is really bad for Hessians...
    @test isapprox(h, Calculus.hessian(f, X), atol=0.02)
    for c in CHUNK_SIZES, tag in (nothing, f)
        println("  ...testing $f with chunk size = $c and tag = $tag")
        cfg = ForwardDiff.HessianConfig(tag, X, ForwardDiff.Chunk{c}())
        resultcfg = ForwardDiff.HessianConfig(tag, DiffResults.HessianResult(X), X, ForwardDiff.Chunk{c}())

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

println("  ...testing specialized StaticArray codepaths")

x = rand(3, 3)
sx = StaticArrays.SArray{Tuple{3,3}}(x)

cfg = ForwardDiff.HessianConfig(nothing, x)
scfg = ForwardDiff.HessianConfig(nothing, sx)

actual = ForwardDiff.hessian(prod, x)
@test ForwardDiff.hessian(prod, sx) == actual
@test ForwardDiff.hessian(prod, sx, cfg) == actual
@test ForwardDiff.hessian(prod, sx, scfg) == actual

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
result2 = ForwardDiff.hessian!(result2, prod, sx, ForwardDiff.HessianConfig(nothing, result2, x))
result3 = ForwardDiff.hessian!(result3, prod, sx, ForwardDiff.HessianConfig(nothing, result3, x))
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
sresult2 = ForwardDiff.hessian!(sresult2, prod, sx, ForwardDiff.HessianConfig(nothing, sresult2, x))
sresult3 = ForwardDiff.hessian!(sresult3, prod, sx, ForwardDiff.HessianConfig(nothing, sresult3, x))
@test DiffResults.value(sresult1) == DiffResults.value(result)
@test DiffResults.value(sresult2) == DiffResults.value(result)
@test DiffResults.value(sresult3) == DiffResults.value(result)
@test DiffResults.gradient(sresult1) == DiffResults.gradient(result)
@test DiffResults.gradient(sresult2) == DiffResults.gradient(result)
@test DiffResults.gradient(sresult3) == DiffResults.gradient(result)
@test DiffResults.hessian(sresult1) == DiffResults.hessian(result)
@test DiffResults.hessian(sresult2) == DiffResults.hessian(result)
@test DiffResults.hessian(sresult3) == DiffResults.hessian(result)

end # module

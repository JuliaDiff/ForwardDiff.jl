module HessianTest

import Calculus

using Compat
using Compat.Test
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

for c in (1, 2, 3), tag in (nothing, Tag((f,gradient), eltype(x)))
    println("  ...running hardcoded test with chunk size = $c and tag = $tag")
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
    for c in CHUNK_SIZES, tag in (nothing, Tag((f,gradient), eltype(x)))
        println("  ...testing $f with chunk size = $c and tag = $tag")
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


println("  ...testing specialized FieldVector codepaths")

struct Point3D{R<:Real} <: FieldVector{3,R}
    x::R
    y::R
    z::R
end
StaticArrays.similar_type(p::Type{P}, ::Type{R}, size::Size{(3,)}) where {P<:Point3D, R<:Real} = Point3D{R}

x = rand(3, 1)
fx = Point3D(x)

cfg = ForwardDiff.HessianConfig(nothing, x)
fcfg = ForwardDiff.HessianConfig(nothing, fx)

actual = ForwardDiff.hessian(prod, x)
@test ForwardDiff.hessian(prod, fx) == actual
@test ForwardDiff.hessian(prod, fx, cfg) == actual
@test ForwardDiff.hessian(prod, fx, fcfg) == actual

out = similar(x, length(x), length(x))
ForwardDiff.hessian!(out, prod, fx)
@test out == actual

out = similar(x, length(x), length(x))
ForwardDiff.hessian!(out, prod, fx, cfg)
@test out == actual

out = similar(x, length(x), length(x))
ForwardDiff.hessian!(out, prod, fx, fcfg)
@test out == actual

result = DiffResults.HessianResult(x)
result = ForwardDiff.hessian!(result, prod, x)

result1 = DiffResults.HessianResult(x)
result2 = DiffResults.HessianResult(x)
result3 = DiffResults.HessianResult(x)
result1 = ForwardDiff.hessian!(result1, prod, fx)
result2 = ForwardDiff.hessian!(result2, prod, fx, ForwardDiff.HessianConfig(prod, result2, x, ForwardDiff.Chunk(x), nothing))
result3 = ForwardDiff.hessian!(result3, prod, fx, ForwardDiff.HessianConfig(prod, result3, x, ForwardDiff.Chunk(x), nothing))
@test DiffResults.value(result1) == DiffResults.value(result)
@test DiffResults.value(result2) == DiffResults.value(result)
@test DiffResults.value(result3) == DiffResults.value(result)
@test DiffResults.gradient(result1) == DiffResults.gradient(result)
@test DiffResults.gradient(result2) == DiffResults.gradient(result)
@test DiffResults.gradient(result3) == DiffResults.gradient(result)
@test DiffResults.hessian(result1) == DiffResults.hessian(result)
@test DiffResults.hessian(result2) == DiffResults.hessian(result)
@test DiffResults.hessian(result3) == DiffResults.hessian(result)

fresult1 = DiffResults.HessianResult(fx)
fresult2 = DiffResults.HessianResult(fx)
fresult3 = DiffResults.HessianResult(fx)
fresult1 = ForwardDiff.hessian!(fresult1, prod, fx)
fresult2 = ForwardDiff.hessian!(fresult2, prod, fx, ForwardDiff.HessianConfig(prod, fresult2, x, ForwardDiff.Chunk(x), nothing))
fresult3 = ForwardDiff.hessian!(fresult3, prod, fx, ForwardDiff.HessianConfig(prod, fresult3, x, ForwardDiff.Chunk(x), nothing))
@test DiffResults.value(fresult1) == DiffResults.value(result)
@test DiffResults.value(fresult2) == DiffResults.value(result)
@test DiffResults.value(fresult3) == DiffResults.value(result)
@test DiffResults.gradient(fresult1) == DiffResults.gradient(result)[:]
@test DiffResults.gradient(fresult2) == DiffResults.gradient(result)[:]
@test DiffResults.gradient(fresult3) == DiffResults.gradient(result)[:]
@test DiffResults.hessian(fresult1) == DiffResults.hessian(result)
@test DiffResults.hessian(fresult2) == DiffResults.hessian(result)
@test DiffResults.hessian(fresult3) == DiffResults.hessian(result)

end # module

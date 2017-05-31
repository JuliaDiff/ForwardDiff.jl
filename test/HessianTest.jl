module HessianTest

import Calculus

using Base.Test
using ForwardDiff
using ForwardDiff: Dual, Tag
using StaticArrays

include(joinpath(dirname(@__FILE__), "utils.jl"))

#############################
# rosenbrock hardcoded test #
#############################

f = DiffBase.rosenbrock_1
x = [0.1, 0.2, 0.3]
v = f(x)
g = [-9.4, 15.6, 52.0]
h = [-66.0  -40.0    0.0;
     -40.0  130.0  -80.0;
       0.0  -80.0  200.0]

for c in (1, 2, 3), tag in (nothing, f)
    println("  ...running hardcoded test with chunk size = $c and tag = $tag")
    cfg = ForwardDiff.HessianConfig(tag, x, ForwardDiff.Chunk{c}())
    resultcfg = ForwardDiff.HessianConfig(tag, DiffBase.HessianResult(x), x, ForwardDiff.Chunk{c}())

    D = Dual{typeof(Tag(Void, eltype(x))), eltype(x), c}
    @test eltype(cfg) == Dual{typeof(Tag(typeof(tag), Dual{Void,eltype(x),0})), D, c}
    @test eltype(resultcfg) == eltype(cfg)

    @test isapprox(h, ForwardDiff.hessian(f, x))
    @test isapprox(h, ForwardDiff.hessian(f, x, cfg))

    out = similar(x, 3, 3)
    out = ForwardDiff.hessian!(out, f, x)
    @test isapprox(out, h)

    out = similar(x, 3, 3)
    out = ForwardDiff.hessian!(out, f, x, cfg)
    @test isapprox(out, h)

    out = DiffBase.HessianResult(x)
    out = ForwardDiff.hessian!(out, f, x)
    @test isapprox(DiffBase.value(out), v)
    @test isapprox(DiffBase.gradient(out), g)
    @test isapprox(DiffBase.hessian(out), h)

    out = DiffBase.HessianResult(x)
    out = ForwardDiff.hessian!(out, f, x, resultcfg)
    @test isapprox(DiffBase.value(out), v)
    @test isapprox(DiffBase.gradient(out), g)
    @test isapprox(DiffBase.hessian(out), h)
end

########################
# test vs. Calculus.jl #
########################

for f in DiffBase.VECTOR_TO_NUMBER_FUNCS
    v = f(X)
    g = ForwardDiff.gradient(f, X)
    h = ForwardDiff.hessian(f, X)
    # finite difference approximation error is really bad for Hessians...
    @test isapprox(h, Calculus.hessian(f, X), atol=0.02)
    for c in CHUNK_SIZES, tag in (nothing, f)
        println("  ...testing $f with chunk size = $c and tag = $tag")
        cfg = ForwardDiff.HessianConfig(tag, X, ForwardDiff.Chunk{c}())
        resultcfg = ForwardDiff.HessianConfig(tag, DiffBase.HessianResult(X), X, ForwardDiff.Chunk{c}())

        out = ForwardDiff.hessian(f, X, cfg)
        @test isapprox(out, h)

        out = similar(X, length(X), length(X))
        out = ForwardDiff.hessian!(out, f, X, cfg)
        @test isapprox(out, h)

        out = DiffBase.HessianResult(X)
        out = ForwardDiff.hessian!(out, f, X, resultcfg)
        @test isapprox(DiffBase.value(out), v)
        @test isapprox(DiffBase.gradient(out), g)
        @test isapprox(DiffBase.hessian(out), h)
    end
end

##########################################
# test specialized StaticArray codepaths #
##########################################

x = rand(3, 3)
sx = StaticArrays.SArray{Tuple{3,3}}(x)
out = similar(x, 9, 9)
actual = ForwardDiff.hessian(prod, x)

@test ForwardDiff.hessian(prod, sx) == actual

out = ForwardDiff.hessian!(out, prod, sx)

@test out == actual

result = DiffBase.HessianResult(x)
sresult = DiffBase.HessianResult(sx)

result = ForwardDiff.hessian!(result, prod, x)
sresult = ForwardDiff.hessian!(sresult, prod, sx)

@test DiffBase.value(sresult) == DiffBase.value(result)
@test DiffBase.gradient(sresult) == DiffBase.gradient(result)
@test DiffBase.hessian(sresult) == DiffBase.hessian(result)

end # module

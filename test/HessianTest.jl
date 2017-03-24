module HessianTest

import Calculus

using Base.Test
using ForwardDiff

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
    resultcfg = ForwardDiff.HessianConfig(DiffBase.HessianResult(x), tag, x, ForwardDiff.Chunk{c}())

    @test isapprox(h, ForwardDiff.hessian(f, x))
    @test isapprox(h, ForwardDiff.hessian(f, x, cfg))

    out = similar(x, 3, 3)
    ForwardDiff.hessian!(out, f, x)
    @test isapprox(out, h)

    out = similar(x, 3, 3)
    ForwardDiff.hessian!(out, f, x, cfg)
    @test isapprox(out, h)

    out = DiffBase.HessianResult(x)
    ForwardDiff.hessian!(out, f, x)
    @test isapprox(DiffBase.value(out), v)
    @test isapprox(DiffBase.gradient(out), g)
    @test isapprox(DiffBase.hessian(out), h)

    out = DiffBase.HessianResult(x)
    ForwardDiff.hessian!(out, f, x, resultcfg)
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
        resultcfg = ForwardDiff.HessianConfig(DiffBase.HessianResult(X), tag, X, ForwardDiff.Chunk{c}())

        out = ForwardDiff.hessian(f, X, cfg)
        @test isapprox(out, h)

        out = similar(X, length(X), length(X))
        ForwardDiff.hessian!(out, f, X, cfg)
        @test isapprox(out, h)

        out = DiffBase.HessianResult(X)
        ForwardDiff.hessian!(out, f, X, resultcfg)
        @test isapprox(DiffBase.value(out), v)
        @test isapprox(DiffBase.gradient(out), g)
        @test isapprox(DiffBase.hessian(out), h)
    end
end

end # module

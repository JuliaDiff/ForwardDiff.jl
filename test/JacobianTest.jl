module JacobianTest

import Calculus

using Base.Test
using ForwardDiff
using ForwardDiff: Dual, Tag, JacobianConfig
using StaticArrays

include(joinpath(dirname(@__FILE__), "utils.jl"))

##################
# hardcoded test #
##################

f! = (y, x) -> begin
    y[1] = x[1] * x[2]
    y[1] *= sin(x[3]^2)
    y[2] = y[1] + x[3]
    y[3] = y[1] / y[2]
    y[4] = x[3]
    return nothing
end
f = x -> (y = zeros(promote_type(eltype(x), Float64), 4); f!(y, x); return y)
x = [1, 2, 3]
v = f(x)
j = [0.8242369704835132  0.4121184852417566  -10.933563142616123
     0.8242369704835132  0.4121184852417566  -9.933563142616123
     0.169076696546684   0.084538348273342   -2.299173530851733
     0.0                 0.0                 1.0]

for c in (1, 2, 3), tags in ((nothing, nothing), (f, f!))
    println("  ...running hardcoded test with chunk size = $c and tag = $tags")
    cfg = JacobianConfig(tags[1], x, ForwardDiff.Chunk{c}())
    ycfg = JacobianConfig(tags[2], zeros(4), x, ForwardDiff.Chunk{c}())

    @test eltype(cfg)  == Dual{typeof(Tag(typeof(tags[1]), eltype(x))), eltype(x), c}
    @test eltype(ycfg) == Dual{typeof(Tag(typeof(tags[2]), eltype(x))), eltype(x), c}

    # testing f(x)
    @test isapprox(j, ForwardDiff.jacobian(f, x, cfg))
    @test isapprox(j, ForwardDiff.jacobian(f, x))

    out = zeros(4, 3)
    ForwardDiff.jacobian!(out, f, x, cfg)
    @test isapprox(out, j)

    out = zeros(4, 3)
    ForwardDiff.jacobian!(out, f, x)
    @test isapprox(out, j)

    out = DiffBase.JacobianResult(zeros(4), zeros(3))
    ForwardDiff.jacobian!(out, f, x, JacobianConfig(tags[1], x))
    @test isapprox(DiffBase.value(out), v)
    @test isapprox(DiffBase.jacobian(out), j)

    # testing f!(y, x)
    y = zeros(4)
    @test isapprox(j, ForwardDiff.jacobian(f!, y, x, ycfg))
    @test isapprox(v, y)

    y = zeros(4)
    @test isapprox(j, ForwardDiff.jacobian(f!, y, x))
    @test isapprox(v, y)

    out, y = zeros(4, 3), zeros(4)
    ForwardDiff.jacobian!(out, f!, y, x, ycfg)
    @test isapprox(out, j)
    @test isapprox(y, v)

    out, y = zeros(4, 3), zeros(4)
    ForwardDiff.jacobian!(out, f!, y, x)
    @test isapprox(out, j)
    @test isapprox(y, v)

    out = DiffBase.JacobianResult(zeros(4), zeros(3))
    y = zeros(4)
    ForwardDiff.jacobian!(out, f!, y, x, ycfg)
    @test DiffBase.value(out) == y
    @test isapprox(y, v)
    @test isapprox(DiffBase.jacobian(out), j)

    out = DiffBase.JacobianResult(zeros(4), zeros(3))
    y = zeros(4)
    ForwardDiff.jacobian!(out, f!, y, x)
    @test DiffBase.value(out) == y
    @test isapprox(y, v)
    @test isapprox(DiffBase.jacobian(out), j)
end

########################
# test vs. Calculus.jl #
########################

for f in DiffBase.ARRAY_TO_ARRAY_FUNCS
    v = f(X)
    j = ForwardDiff.jacobian(f, X)
    @test isapprox(j, Calculus.jacobian(x -> vec(f(x)), X, :forward), atol=FINITEDIFF_ERROR)
    for c in CHUNK_SIZES, tag in (nothing, f)
        println("  ...testing $f with chunk size = $c and tag = $tag")
        cfg = JacobianConfig(tag, X, ForwardDiff.Chunk{c}())

        out = ForwardDiff.jacobian(f, X, cfg)
        @test isapprox(out, j)

        out = similar(X, length(v), length(X))
        ForwardDiff.jacobian!(out, f, X, cfg)
        @test isapprox(out, j)

        out = DiffBase.DiffResult(similar(v, length(v)), similar(v, length(v), length(X)))
        ForwardDiff.jacobian!(out, f, X, cfg)
        @test isapprox(DiffBase.value(out), v)
        @test isapprox(DiffBase.jacobian(out), j)
    end
end

for f! in DiffBase.INPLACE_ARRAY_TO_ARRAY_FUNCS
    v = zeros(Y)
    f!(v, X)
    j = ForwardDiff.jacobian(f!, zeros(Y), X)
    @test isapprox(j, Calculus.jacobian(x -> (y = zeros(Y); f!(y, x); vec(y)), X, :forward), atol=FINITEDIFF_ERROR)
    for c in CHUNK_SIZES, tag in (nothing, f!)
        println("  ...testing $(f!) with chunk size = $c and tag = $tag")
        ycfg = JacobianConfig(tag, zeros(Y), X, ForwardDiff.Chunk{c}())

        y = zeros(Y)
        out = ForwardDiff.jacobian(f!, y, X, ycfg)
        @test isapprox(y, v)
        @test isapprox(out, j)

        y = zeros(Y)
        out = similar(Y, length(Y), length(X))
        ForwardDiff.jacobian!(out, f!, y, X)
        @test isapprox(y, v)
        @test isapprox(out, j)

        y = zeros(Y)
        out = DiffBase.JacobianResult(y, X)
        ForwardDiff.jacobian!(out, f!, y, X)
        @test DiffBase.value(out) == y
        @test isapprox(y, v)
        @test isapprox(DiffBase.jacobian(out), j)

        y = zeros(Y)
        out = DiffBase.JacobianResult(y, X)
        ForwardDiff.jacobian!(out, f!, y, X, ycfg)
        @test DiffBase.value(out) == y
        @test isapprox(y, v)
        @test isapprox(DiffBase.jacobian(out), j)
    end
end

##########################################
# test specialized StaticArray codepaths #
##########################################

x = rand(3, 3)
sx = StaticArrays.SArray{Tuple{3,3}}(x)
out = similar(x, 6, 9)
actual = ForwardDiff.jacobian(diff, x)

@test ForwardDiff.jacobian(diff, sx) == actual

out = ForwardDiff.jacobian!(out, diff, sx)

@test out == actual

#test MutableDiffResult
result = DiffBase.JacobianResult(similar(x, 6), x)
sresult = DiffBase.JacobianResult(similar(sx, 6), sx)

result = ForwardDiff.jacobian!(result, diff, x)
sresult = ForwardDiff.jacobian!(sresult, diff, sx)

@test DiffBase.value(sresult) == DiffBase.value(result)
@test DiffBase.jacobian(sresult) == DiffBase.jacobian(result)

#test ImmutableDiffResult
result = DiffBase.JacobianResult(similar(x, 6), x)
sresult = DiffBase.JacobianResult(SVector{6}(zeros(6)), sx)

result = ForwardDiff.jacobian!(result, diff, x)
sresult = ForwardDiff.jacobian!(sresult, diff, sx)

@test DiffBase.value(sresult) == DiffBase.value(result)
@test DiffBase.jacobian(sresult) == DiffBase.jacobian(result)

end # module

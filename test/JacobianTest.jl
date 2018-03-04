module JacobianTest

import Calculus

using Compat
using Compat.Test
using ForwardDiff
using ForwardDiff: Dual, Tag, JacobianConfig
using StaticArrays
using DiffTests

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
f = x -> (y = fill(zero(promote_type(eltype(x), Float64)), 4); f!(y, x); return y)
x = [1, 2, 3]
v = f(x)
j = [0.8242369704835132  0.4121184852417566  -10.933563142616123
     0.8242369704835132  0.4121184852417566  -9.933563142616123
     0.169076696546684   0.084538348273342   -2.299173530851733
     0.0                 0.0                 1.0]

for c in (1, 2, 3), tags in ((nothing, nothing),
                             (Tag(f, eltype(x)), Tag(f!, eltype(x))))
    println("  ...running hardcoded test with chunk size = $c and tag = $tags")
    cfg = JacobianConfig(f, x, ForwardDiff.Chunk{c}(), tags[1])
    ycfg = JacobianConfig(f!, fill(0.0, 4), x, ForwardDiff.Chunk{c}(), tags[2])

    @test eltype(cfg)  == Dual{typeof(tags[1]), eltype(x), c}
    @test eltype(ycfg) == Dual{typeof(tags[2]), eltype(x), c}

    # testing f(x)
    @test isapprox(j, ForwardDiff.jacobian(f, x, cfg))
    @test isapprox(j, ForwardDiff.jacobian(f, x))

    out = fill(0.0, 4, 3)
    ForwardDiff.jacobian!(out, f, x, cfg)
    @test isapprox(out, j)

    out = fill(0.0, 4, 3)
    ForwardDiff.jacobian!(out, f, x)
    @test isapprox(out, j)

    out = DiffResults.JacobianResult(fill(0.0, 4), fill(0.0, 3))
    ForwardDiff.jacobian!(out, f, x, cfg)
    @test isapprox(DiffResults.value(out), v)
    @test isapprox(DiffResults.jacobian(out), j)

    # testing f!(y, x)
    y = fill(0.0, 4)
    @test isapprox(j, ForwardDiff.jacobian(f!, y, x, ycfg))
    @test isapprox(v, y)

    y = fill(0.0, 4)
    @test isapprox(j, ForwardDiff.jacobian(f!, y, x))
    @test isapprox(v, y)

    out, y = fill(0.0, 4, 3), fill(0.0, 4)
    ForwardDiff.jacobian!(out, f!, y, x, ycfg)
    @test isapprox(out, j)
    @test isapprox(y, v)

    out, y = fill(0.0, 4, 3), fill(0.0, 4)
    ForwardDiff.jacobian!(out, f!, y, x)
    @test isapprox(out, j)
    @test isapprox(y, v)

    out = DiffResults.JacobianResult(fill(0.0, 4), fill(0.0, 3))
    y = fill(0.0, 4)
    ForwardDiff.jacobian!(out, f!, y, x, ycfg)
    @test DiffResults.value(out) == y
    @test isapprox(y, v)
    @test isapprox(DiffResults.jacobian(out), j)

    out = DiffResults.JacobianResult(fill(0.0, 4), fill(0.0, 3))
    y = fill(0.0, 4)
    ForwardDiff.jacobian!(out, f!, y, x)
    @test DiffResults.value(out) == y
    @test isapprox(y, v)
    @test isapprox(DiffResults.jacobian(out), j)
end

cfgx = ForwardDiff.JacobianConfig(sin, x)
@test_throws ForwardDiff.InvalidTagException ForwardDiff.jacobian(f, x, cfgx)
@test ForwardDiff.jacobian(f, x, cfgx, Val{false}()) == ForwardDiff.jacobian(f,x)


########################
# test vs. Calculus.jl #
########################

for f in DiffTests.ARRAY_TO_ARRAY_FUNCS
    v = f(X)
    j = ForwardDiff.jacobian(f, X)
    @test isapprox(j, Calculus.jacobian(x -> vec(f(x)), X, :forward), atol=FINITEDIFF_ERROR)
    for c in CHUNK_SIZES, tag in (nothing, Tag)
        if tag == Tag
            tag = Tag(f, eltype(X))
        end
        println("  ...testing $f with chunk size = $c and tag = $tag")
        cfg = JacobianConfig(f, X, ForwardDiff.Chunk{c}(), tag)

        out = ForwardDiff.jacobian(f, X, cfg)
        @test isapprox(out, j)

        out = similar(X, length(v), length(X))
        ForwardDiff.jacobian!(out, f, X, cfg)
        @test isapprox(out, j)

        out = DiffResults.DiffResult(similar(v, length(v)), similar(v, length(v), length(X)))
        ForwardDiff.jacobian!(out, f, X, cfg)
        @test isapprox(DiffResults.value(out), v)
        @test isapprox(DiffResults.jacobian(out), j)
    end
end

for f! in DiffTests.INPLACE_ARRAY_TO_ARRAY_FUNCS
    v = fill!(similar(Y), 0.0)
    f!(v, X)
    j = ForwardDiff.jacobian(f!, fill!(similar(Y), 0.0), X)
    @test isapprox(j, Calculus.jacobian(x -> (y = fill!(similar(Y), 0.0); f!(y, x); vec(y)), X, :forward), atol=FINITEDIFF_ERROR)
    for c in CHUNK_SIZES, tag in (nothing, Tag(f!, eltype(X)))
        println("  ...testing $(f!) with chunk size = $c and tag = $tag")
        ycfg = JacobianConfig(f!, fill!(similar(Y), 0.0), X, ForwardDiff.Chunk{c}(), tag)

        y = fill!(similar(Y), 0.0)
        out = ForwardDiff.jacobian(f!, y, X, ycfg)
        @test isapprox(y, v)
        @test isapprox(out, j)

        y = fill!(similar(Y), 0.0)
        out = similar(Y, length(Y), length(X))
        ForwardDiff.jacobian!(out, f!, y, X)
        @test isapprox(y, v)
        @test isapprox(out, j)

        y = fill!(similar(Y), 0.0)
        out = DiffResults.JacobianResult(y, X)
        ForwardDiff.jacobian!(out, f!, y, X)
        @test DiffResults.value(out) == y
        @test isapprox(y, v)
        @test isapprox(DiffResults.jacobian(out), j)

        y = fill!(similar(Y), 0.0)
        out = DiffResults.JacobianResult(y, X)
        ForwardDiff.jacobian!(out, f!, y, X, ycfg)
        @test DiffResults.value(out) == y
        @test isapprox(y, v)
        @test isapprox(DiffResults.jacobian(out), j)
    end
end

##########################################
# test specialized StaticArray codepaths #
##########################################

println("  ...testing specialized StaticArray codepaths")

x = rand(3, 3)
sx = StaticArrays.SArray{Tuple{3,3}}(x)

cfg = ForwardDiff.JacobianConfig(nothing, x)
scfg = ForwardDiff.JacobianConfig(nothing, sx)

actual = ForwardDiff.jacobian(diff, x)
@test ForwardDiff.jacobian(diff, sx) == actual
@test ForwardDiff.jacobian(diff, sx, cfg) == actual
@test ForwardDiff.jacobian(diff, sx, scfg) == actual

out = similar(x, 6, 9)
ForwardDiff.jacobian!(out, diff, sx)
@test out == actual

out = similar(x, 6, 9)
ForwardDiff.jacobian!(out, diff, sx, cfg)
@test out == actual

out = similar(x, 6, 9)
ForwardDiff.jacobian!(out, diff, sx, scfg)
@test out == actual

result = DiffResults.JacobianResult(similar(x, 6), x)
result = ForwardDiff.jacobian!(result, diff, x)

result1 = DiffResults.JacobianResult(similar(sx, 6), sx)
result2 = DiffResults.JacobianResult(similar(sx, 6), sx)
result3 = DiffResults.JacobianResult(similar(sx, 6), sx)
result1 = ForwardDiff.jacobian!(result1, diff, sx)
result2 = ForwardDiff.jacobian!(result2, diff, sx, cfg)
result3 = ForwardDiff.jacobian!(result3, diff, sx, scfg)
@test DiffResults.value(result1) == DiffResults.value(result)
@test DiffResults.value(result2) == DiffResults.value(result)
@test DiffResults.value(result3) == DiffResults.value(result)
@test DiffResults.jacobian(result1) == DiffResults.jacobian(result)
@test DiffResults.jacobian(result2) == DiffResults.jacobian(result)
@test DiffResults.jacobian(result3) == DiffResults.jacobian(result)

sy = @SVector fill(zero(eltype(sx)), 6)
sresult1 = DiffResults.JacobianResult(sy, sx)
sresult2 = DiffResults.JacobianResult(sy, sx)
sresult3 = DiffResults.JacobianResult(sy, sx)
sresult1 = ForwardDiff.jacobian!(sresult1, diff, sx)
sresult2 = ForwardDiff.jacobian!(sresult2, diff, sx, cfg)
sresult3 = ForwardDiff.jacobian!(sresult3, diff, sx, scfg)
@test DiffResults.value(sresult1) == DiffResults.value(result)
@test DiffResults.value(sresult2) == DiffResults.value(result)
@test DiffResults.value(sresult3) == DiffResults.value(result)
@test DiffResults.jacobian(sresult1) == DiffResults.jacobian(result)
@test DiffResults.jacobian(sresult2) == DiffResults.jacobian(result)
@test DiffResults.jacobian(sresult3) == DiffResults.jacobian(result)


println("  ...testing specialized FieldVector codepaths")

struct Point3D{R<:Real} <: FieldVector{3,R}
    x::R
    y::R
    z::R
end
StaticArrays.similar_type(p::Type{P}, ::Type{R}, size::Size{(3,)}) where {P<:Point3D, R<:Real} = Point3D{R}

x = rand(3, 1)
fx = Point3D(x)

cfg = ForwardDiff.JacobianConfig(nothing, x)
fcfg = ForwardDiff.JacobianConfig(nothing, fx)

actual = ForwardDiff.jacobian(diff, x)
@test ForwardDiff.jacobian(diff, fx) == actual
@test ForwardDiff.jacobian(diff, fx, cfg) == actual
@test ForwardDiff.jacobian(diff, fx, fcfg) == actual

out = similar(x, 2, 3)
ForwardDiff.jacobian!(out, diff, fx)
@test out == actual

out = similar(x, 2, 3)
ForwardDiff.jacobian!(out, diff, fx, cfg)
@test out == actual

out = similar(x, 2, 3)
ForwardDiff.jacobian!(out, diff, fx, fcfg)
@test out == actual

result = DiffResults.JacobianResult(similar(x, 2), x)
result = ForwardDiff.jacobian!(result, diff, x)

result1 = DiffResults.JacobianResult(similar(fx, 2), fx)
result2 = DiffResults.JacobianResult(similar(fx, 2), fx)
result3 = DiffResults.JacobianResult(similar(fx, 2), fx)
result1 = ForwardDiff.jacobian!(result1, diff, fx)
result2 = ForwardDiff.jacobian!(result2, diff, fx, cfg)
result3 = ForwardDiff.jacobian!(result3, diff, fx, fcfg)
@test DiffResults.value(result1) == DiffResults.value(result)
@test DiffResults.value(result2) == DiffResults.value(result)
@test DiffResults.value(result3) == DiffResults.value(result)
@test DiffResults.jacobian(result1) == DiffResults.jacobian(result)
@test DiffResults.jacobian(result2) == DiffResults.jacobian(result)
@test DiffResults.jacobian(result3) == DiffResults.jacobian(result)

sy = @SVector fill(zero(eltype(fx)), 2)
fresult1 = DiffResults.JacobianResult(sy, fx)
fresult2 = DiffResults.JacobianResult(sy, fx)
fresult3 = DiffResults.JacobianResult(sy, fx)
fresult1 = ForwardDiff.jacobian!(fresult1, diff, fx)
fresult2 = ForwardDiff.jacobian!(fresult2, diff, fx, cfg)
fresult3 = ForwardDiff.jacobian!(fresult3, diff, fx, fcfg)
@test DiffResults.value(fresult1) == DiffResults.value(result)
@test DiffResults.value(fresult2) == DiffResults.value(result)
@test DiffResults.value(fresult3) == DiffResults.value(result)
@test DiffResults.jacobian(fresult1) == DiffResults.jacobian(result)
@test DiffResults.jacobian(fresult2) == DiffResults.jacobian(result)
@test DiffResults.jacobian(fresult3) == DiffResults.jacobian(result)
end # module

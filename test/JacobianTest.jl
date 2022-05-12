module JacobianTest

import Calculus

using Test
using ForwardDiff
using ForwardDiff: Dual, Tag, JacobianConfig
using StaticArrays
using DiffTests
using LinearAlgebra

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
    println("  ...running hardcoded test with chunk size = $c and tag = $(repr(tags))")
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
    @test isapprox(j, Calculus.jacobian(x -> vec(f(x)), X, :forward), atol=1.3FINITEDIFF_ERROR)
    for c in CHUNK_SIZES, tag in (nothing, Tag)
        if tag == Tag
            tag = Tag(f, eltype(X))
        end
        println("  ...testing $f with chunk size = $c and tag = $(repr(tag))")
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
        println("  ...testing $(f!) with chunk size = $c and tag = $(repr(tag))")
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
for T in (StaticArrays.SArray, StaticArrays.MArray)
    sx = T{Tuple{3,3}}(x)

    cfg = ForwardDiff.JacobianConfig(nothing, x)
    scfg = ForwardDiff.JacobianConfig(nothing, sx)

    _diff(A) = diff(A; dims=1)

    actual = ForwardDiff.jacobian(_diff, x)
    @test ForwardDiff.jacobian(_diff, sx) == actual
    @test ForwardDiff.jacobian(_diff, sx, cfg) == actual
    @test ForwardDiff.jacobian(_diff, sx, scfg) == actual
    @test ForwardDiff.jacobian(_diff, sx, scfg) isa StaticArray
    @test ForwardDiff.jacobian(_diff, sx, scfg, Val{false}()) == actual
    @test ForwardDiff.jacobian(_diff, sx, scfg, Val{false}()) isa StaticArray

    out = similar(x, 6, 9)
    ForwardDiff.jacobian!(out, _diff, sx)
    @test out == actual

    out = similar(x, 6, 9)
    ForwardDiff.jacobian!(out, _diff, sx, cfg)
    @test out == actual

    out = similar(x, 6, 9)
    ForwardDiff.jacobian!(out, _diff, sx, scfg)
    @test out == actual

    result = DiffResults.JacobianResult(similar(x, 6), x)
    result = ForwardDiff.jacobian!(result, _diff, x)

    result1 = DiffResults.JacobianResult(similar(sx, 6), sx)
    result2 = DiffResults.JacobianResult(similar(sx, 6), sx)
    result3 = DiffResults.JacobianResult(similar(sx, 6), sx)
    result1 = ForwardDiff.jacobian!(result1, _diff, sx)
    result2 = ForwardDiff.jacobian!(result2, _diff, sx, cfg)
    result3 = ForwardDiff.jacobian!(result3, _diff, sx, scfg)
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
    sresult1 = ForwardDiff.jacobian!(sresult1, _diff, sx)
    sresult2 = ForwardDiff.jacobian!(sresult2, _diff, sx, cfg)
    sresult3 = ForwardDiff.jacobian!(sresult3, _diff, sx, scfg)
    @test DiffResults.value(sresult1) == DiffResults.value(result)
    @test DiffResults.value(sresult2) == DiffResults.value(result)
    @test DiffResults.value(sresult3) == DiffResults.value(result)
    @test DiffResults.jacobian(sresult1) == DiffResults.jacobian(result)
    @test DiffResults.jacobian(sresult2) == DiffResults.jacobian(result)
    @test DiffResults.jacobian(sresult3) == DiffResults.jacobian(result)
end

@testset "dimension errors for jacobian" begin
    @test_throws DimensionMismatch ForwardDiff.jacobian(identity, 2pi) # input
    @test_throws DimensionMismatch ForwardDiff.jacobian(sum, fill(2pi, 2)) # vector_mode_jacobian
    @test_throws DimensionMismatch ForwardDiff.jacobian(sum, fill(2pi, 10^6)) # chunk_mode_jacobian
end

@testset "eigen" begin
    @test ForwardDiff.jacobian(x -> eigvals(SymTridiagonal(x, x[1:end-1])), [1.,2.]) ≈ [(1 - 3/sqrt(5))/2 (1 - 1/sqrt(5))/2 ; (1 + 3/sqrt(5))/2 (1 + 1/sqrt(5))/2]
    @test ForwardDiff.jacobian(x -> eigvals(Symmetric(x*x')), [1.,2.]) ≈ [0 0; 2 4]
    
    x0 = [1.0, 2.0];
    ev1(x) = eigen(Symmetric(x*x')).vectors[:,1]
    @test ForwardDiff.jacobian(ev1, x0) ≈ Calculus.finite_difference_jacobian(ev1, x0)
    ev2(x) = eigen(SymTridiagonal(x, x[1:end-1])).vectors[:,1]
    @test ForwardDiff.jacobian(ev2, x0) ≈ Calculus.finite_difference_jacobian(ev2, x0)
    x0_static = SVector{2}(x0)
    @test ForwardDiff.jacobian(ev1, x0_static) ≈ Calculus.finite_difference_jacobian(ev1, x0)
end

@testset "type stability" begin
    g!(dy, y) = dy[1] = y[1]
    @inferred ForwardDiff.jacobian(g!, [1.0], [0.0])
end

end # module

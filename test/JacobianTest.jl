module JacobianTest

import Calculus

using Test
using ForwardDiff
using ForwardDiff: Dual, Tag, JacobianConfig
using StaticArrays
using DiffTests
using LinearAlgebra

include(joinpath(dirname(@__FILE__), "utils.jl"))

struct TestTag end
struct OuterTestTag end
ForwardDiff.:≺(::Type{TestTag}, ::Type{OuterTestTag}) = true
ForwardDiff.:≺(::Type{OuterTestTag}, ::Type{<:Tag}) = true

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
    @testset "$f with chunk size = $c and tag = $(repr(tag))" for c in CHUNK_SIZES, tag in (nothing, Tag)
        if tag == Tag
            tag = Tag(f, eltype(X))
        end
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
    @testset "$(f!) with chunk size = $c and tag = $(repr(tag))" for c in CHUNK_SIZES, tag in (nothing, Tag(f!, eltype(X)))
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

@info "testing specialized StaticArray codepaths"

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

    # make sure this is not a source of type instability
    @inferred ForwardDiff.JacobianConfig(f, sx)
end

#########
# misc. #
#########

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

    x0_svector = SVector{2}(x0)
    @test ForwardDiff.jacobian(ev1, x0_svector) isa SMatrix{2, 2}
    @test ForwardDiff.jacobian(ev1, x0_svector) ≈ Calculus.finite_difference_jacobian(ev1, x0)

    x0_mvector = MVector{2}(x0)
    @test ForwardDiff.jacobian(ev1, x0_mvector) isa MMatrix{2, 2}
    @test ForwardDiff.jacobian(ev1, x0_mvector) ≈ Calculus.finite_difference_jacobian(ev1, x0)

    # real eigenvalues
    f(x) = eigvals(reshape(x, 2, 2))
    x1 = [1.0, 2.0, 3.0, 4.0]
    @test ForwardDiff.jacobian(f, x1) ≈ Calculus.finite_difference_jacobian(f, x1)

    # complex eigenvalues
    g(x) = begin
        vals = eigvals(reshape(x, 2, 2))
        vcat(real(vals), imag(vals))
    end
    x2 = [0.0, -1.0, 1.0, 0.0]
    @test ForwardDiff.jacobian(g, x2) ≈ Calculus.finite_difference_jacobian(g, x2)

    # eigenvectors, deliberately without renormalizing: the derivatives have to belong to
    # the normalization `eigen` itself returns, i.e. unit 2-norm with largest entry real
    h(x) = vec(eigen(reshape(x, 2, 2)).vectors)
    x3 = [2.0, 1.0, 0.5, 3.0]
    @test ForwardDiff.jacobian(h, x3) ≈ Calculus.finite_difference_jacobian(h, x3)

    # complex eigenvectors
    hc(x) = begin
        V = eigen(reshape(x, 2, 2)).vectors
        vcat(real(vec(V)), imag(vec(V)))
    end
    x3c = vec([0.3 -1.2; 1.7 0.5])
    @test ForwardDiff.jacobian(hc, x3c) ≈ Calculus.finite_difference_jacobian(hc, x3c)

    # larger than 2x2, non-symmetric with real eigenvalues
    f3(x) = eigvals(reshape(x, 3, 3))
    x4 = vec([2.0 1.0 0.5; 0.5 3.0 1.5; 0.25 0.75 4.0])
    @test ForwardDiff.jacobian(f3, x4) ≈ Calculus.finite_difference_jacobian(f3, x4)
    h3(x) = vec(eigen(reshape(x, 3, 3)).vectors)
    @test ForwardDiff.jacobian(h3, x4) ≈ Calculus.finite_difference_jacobian(h3, x4)

    # 3x3 with one real and one complex conjugate pair of eigenvalues
    g3(x) = begin
        vals = eigvals(reshape(x, 3, 3))
        vcat(real(vals), imag(vals))
    end
    hc3(x) = begin
        V = eigen(reshape(x, 3, 3)).vectors
        vcat(real(vec(V)), imag(vec(V)))
    end
    x4c = vec([0.5 -1.3 0.2; 1.1 0.4 -0.6; 0.3 0.7 2.0])
    @test ForwardDiff.jacobian(g3, x4c) ≈ Calculus.finite_difference_jacobian(g3, x4c)
    @test ForwardDiff.jacobian(hc3, x4c) ≈ Calculus.finite_difference_jacobian(hc3, x4c)

    # the eigenvector derivatives used to belong to the normalization
    # `diag(inv(U) * U̇) == 0` instead, which differs for a non-normal matrix
    A_gauge = Dual{TestTag}.([1.0 1.0; 0.0 2.0], [1.0 0.0; 0.0 0.0])
    @test ForwardDiff.partials.(eigen(A_gauge).vectors, 1) ≈ [0.0 1/(2*sqrt(2)); 0.0 -1/(2*sqrt(2))]

    # keyword arguments are forwarded to the decomposition of the values
    A_kw = reshape(x4, 3, 3)
    A_kw_dual = Dual{TestTag}.(A_kw, Matrix(1.0I, 3, 3))
    for kwargs in ((), (sortby = nothing,), (permute = false,), (scale = false,),
                   (permute = false, scale = false), (sortby = λ -> -real(λ),))
        @test ForwardDiff.value.(eigvals(A_kw_dual; kwargs...)) ≈ eigvals(A_kw; kwargs...)
        @test eigvals(A_kw_dual; kwargs...) ≈ eigen(A_kw_dual; kwargs...).values
        f_kw(x) = eigvals(reshape(x, 3, 3); kwargs...)
        @test ForwardDiff.jacobian(f_kw, x4) ≈ Calculus.finite_difference_jacobian(f_kw, x4)
    end
    # and the derivatives follow the ordering they produce: `A_kw` has a real spectrum, so
    # sorting by `-real` reverses the order `LinearAlgebra.eigsortby` gives
    @test ForwardDiff.jacobian(x -> eigvals(reshape(x, 3, 3); sortby = λ -> -real(λ)), x4) ≈
          ForwardDiff.jacobian(x -> eigvals(reshape(x, 3, 3)), x4)[3:-1:1, :]

    # eltypes of the general path
    A_dual = Dual{TestTag}.([1.0 2.0; 3.0 4.0], [1.0 0.0; 0.0 0.0])
    @test eigvals(A_dual) isa Vector{Dual{TestTag,Float64,1}}
    @test eigen(A_dual).vectors isa Matrix{Dual{TestTag,Float64,1}}
    A_dual_complex = Dual{TestTag}.([0.0 -1.0; 1.0 0.0], [1.0 0.0; 0.0 0.0])
    @test eigvals(A_dual_complex) isa Vector{Complex{Dual{TestTag,Float64,1}}}
    @test eigen(A_dual_complex).vectors isa Matrix{Complex{Dual{TestTag,Float64,1}}}
end

@testset "type stability" begin
    g!(dy, y) = dy[1] = y[1]
    @inferred ForwardDiff.jacobian(g!, [1.0], [0.0])

    @testset "issue 639" begin
        f(x) = SA[x[1]^2+x[2]^2, x[2]^2+x[3]^2]
        x = SA[1.0, 2.0, 3.0]
        y = f(x)
        imdr = DiffResults.JacobianResult(y, x)
        @inferred ForwardDiff.jacobian!(imdr, f, x)
    end

    @testset "pr 735" begin
        f(x) = x .^ 2 ./ 2
        function withjacobian(x)
            res = DiffResults.JacobianResult(x)
            res = ForwardDiff.jacobian!(res, f, x)
            return DiffResults.value(res), DiffResults.jacobian(res)
        end
        @inferred withjacobian(SA[1.0, 2.0])
    end
end

# issues #436, #740
@testset "BigFloat" begin
    # Unassigned entries in the output
    x = BigFloat.(1:9)
    for chunksize in (1, 2, 9)
        y = similar(x)
        @test all(i -> !isassigned(y, i), eachindex(y))
        cfg = ForwardDiff.JacobianConfig(copyto!, y, x, ForwardDiff.Chunk{chunksize}())
        res = ForwardDiff.jacobian(copyto!, y, x, cfg)
        @test y == x
        @test res isa Matrix{BigFloat}
        @test res == I
    end

    # Unassigned (but unused) entry in the input and unassigned entries in the output. `hole` is
    # varied so the unassigned entry lands in a middle chunk as well as in the last one: only the
    # former reaches the `Base._unsetindex!` branch of the windowed seeding path, since the last
    # chunk is never cleared.
    @testset "unassigned input entry at $hole" for hole in (5, 10)
        x = Vector{BigFloat}(undef, 10)
        for i in eachindex(x)
            i == hole || (x[i] = BigFloat(i))
        end
        used = [i for i in eachindex(x) if i != hole]
        f = (y, x) -> (for (k, i) in enumerate(used); y[k] = x[i]; end; y)
        for chunksize in (1, 2, 10)
            y = similar(x, 9)
            @test all(i -> !isassigned(y, i), eachindex(y))
            cfg = ForwardDiff.JacobianConfig(f, y, x, ForwardDiff.Chunk{chunksize}())
            res = ForwardDiff.jacobian(f, y, x, cfg)
            @test y == x[used]
            @test res isa Matrix{BigFloat}
            @test res[:, used] == I
            @test all(iszero, res[:, hole])
        end
    end
end

# issue #769
@testset "functions with `Dual` output" begin
    x = [Dual{OuterTestTag}(Dual{TestTag}(1.3, 2.1), Dual{TestTag}(0.3, -2.4))]
    f(x) = map(ForwardDiff.value, x)
    der = ForwardDiff.derivative(ForwardDiff.value, only(x))

    # Vector mode
    jac = ForwardDiff.jacobian(f, x)
    @test jac isa Matrix{typeof(der)}
    @test jac == [der;;]
    jac = ForwardDiff.jacobian(f, SVector{1}(x))
    @test jac isa SMatrix{1,1,typeof(der)}
    @test jac == SMatrix{1,1}(der)

    # Chunk mode
    y = repeat(x, 3)
    cfg = ForwardDiff.JacobianConfig(f, y, ForwardDiff.Chunk{2}())
    jac = ForwardDiff.jacobian(f, y, cfg)
    @test jac isa Matrix{typeof(der)}
    @test jac == Diagonal([der, der, der])
    cfg = ForwardDiff.JacobianConfig(f, SVector{3}(y), ForwardDiff.Chunk{2}())
    jac = ForwardDiff.jacobian(f, SVector{3}(y), cfg)
    @test jac isa SMatrix{3,3,typeof(der)}
    @test jac == Diagonal([der, der, der])
end

end # module

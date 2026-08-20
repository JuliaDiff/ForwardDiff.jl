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

# issue #839
@testset "structured inputs: $(nameof(typeof(x)))" for (x, sidx) in (
        # The Jacobian is indexed by the linear indices of `x`: column `j` holds the derivatives with
        # respect to `x[j]`, and the columns of the structurally zero entries are zero. The nonzero
        # columns are written out by hand so that a bug in the position mapping cannot hide inside the
        # reference. Only the full-length chunk worked before, the others threw.
        (LowerTriangular(randn(4, 4)), [i + 4 * (j - 1) for j in 1:4 for i in j:4]),
        (UpperTriangular(randn(4, 4)), [i + 4 * (j - 1) for j in 1:4 for i in 1:j]),
        (Diagonal(randn(4, 4)),        collect(1:5:16)),
    )
    g = z -> [sum(z), sum(abs2, z)]
    g! = (y, z) -> (y[1] = sum(z); y[2] = sum(abs2, z); y)

    expected = zeros(2, length(x))
    expected[1, sidx] .= 1
    expected[2, sidx] .= 2 .* x[sidx]
    val = g(x)

    # `length(sidx)` is 10 or 4, so a chunk size of 3 leaves a partial final chunk
    @testset "chunk size = $c" for c in unique((1, 2, 3, length(sidx)))
        cfg = ForwardDiff.JacobianConfig(g, x, ForwardDiff.Chunk{c}())
        J = ForwardDiff.jacobian(g, x, cfg)
        @test size(J) == (2, length(x))
        @test J == expected

        out = fill(NaN, 2, length(x))
        @test ForwardDiff.jacobian!(out, g, x, cfg) === out
        @test out == expected

        # a result that is not a matrix is reshaped to one
        out = fill(NaN, 2 * length(x))
        @test ForwardDiff.jacobian!(out, g, x, cfg) === out
        @test reshape(out, 2, length(x)) == expected

        # `DiffResults.JacobianResult` allocates `length(x)` columns, which is what is needed
        result = DiffResults.JacobianResult(similar(val), x)
        result = ForwardDiff.jacobian!(result, g, x, cfg)
        @test DiffResults.jacobian(result) == expected
        @test DiffResults.value(result) ≈ val

        # in-place target function
        cfg! = ForwardDiff.JacobianConfig(g!, similar(val), x, ForwardDiff.Chunk{c}())
        y = fill(NaN, 2)
        @test ForwardDiff.jacobian(g!, y, x, cfg!) == expected
        @test y ≈ val
        out = fill(NaN, 2, length(x))
        y = fill(NaN, 2)
        ForwardDiff.jacobian!(out, g!, y, x, cfg!)
        @test out == expected
        @test y ≈ val
        result = DiffResults.JacobianResult(similar(val), x)
        y = fill(NaN, 2)
        result = ForwardDiff.jacobian!(result, g!, y, x, cfg!)
        @test DiffResults.jacobian(result) == expected
        @test DiffResults.value(result) ≈ val
    end
end

# issue #842
@testset "config reused for a differently structured input" begin
    # A config seeds the positions of its own work buffers, so an input of a different structure would
    # be seeded at one set of positions and extracted at another.
    n = 4
    A = randn(n, n)
    inputs = (A, LowerTriangular(A), UpperTriangular(A), Diagonal(diag(A)))
    g = z -> [sum(z), sum(abs2, z)]
    g! = (y, z) -> (y[1] = sum(z); y[2] = sum(abs2, z); y)

    @testset "chunk size = $c" for c in (2, n)
        @testset "$(nameof(typeof(xcfg))) config" for xcfg in inputs
            cfg = ForwardDiff.JacobianConfig(g, xcfg, ForwardDiff.Chunk{c}())
            cfg! = ForwardDiff.JacobianConfig(g!, zeros(2), xcfg, ForwardDiff.Chunk{c}())
            # each input has a different structure, so `x === xcfg` is the one the config fits
            for x in inputs
                x === xcfg && continue
                out = fill(NaN, 2, n^2)
                msg = "ArgumentError: the config was built for an array of type " *
                      "$(nameof(typeof(xcfg))) and cannot be used with an array of type " *
                      "$(nameof(typeof(x)))"
                @test_throws msg ForwardDiff.jacobian(g, x, cfg)
                @test_throws msg ForwardDiff.jacobian!(out, g, x, cfg)
                @test_throws msg ForwardDiff.jacobian(g!, fill(NaN, 2), x, cfg!)
                @test_throws msg ForwardDiff.jacobian!(out, g!, fill(NaN, 2), x, cfg!)
            end
        end

        # the buffer an `f!(y, x)` config holds for the output is checked too
        cfg! = ForwardDiff.JacobianConfig(nothing, Diagonal(zeros(2, 2)), A, ForwardDiff.Chunk{c}())
        @test_throws "ArgumentError: the config was built for an array of type Diagonal and " *
                     "cannot be used with an array of type Array" ForwardDiff.jacobian(
                         g!, zeros(2, 2), A, cfg!)
    end
end

@testset "wrongly shaped result" begin
    # A matrix result is used as is, so it has to have the shape of the Jacobian and not merely as
    # many entries. Results of other shapes are reshaped and only have to match in length.
    x = randn(4)
    g = z -> [sum(z), sum(abs2, z)]
    g! = (y, z) -> (y[1] = sum(z); y[2] = sum(abs2, z); y)
    @testset "chunk size = $c" for c in (2, 4)
        cfg = ForwardDiff.JacobianConfig(g, x, ForwardDiff.Chunk{c}())
        @test_throws DimensionMismatch ForwardDiff.jacobian!(fill(NaN, 4, 2), g, x, cfg)
        result = DiffResults.DiffResult(fill(NaN, 2), fill(NaN, 4, 2))
        @test_throws DimensionMismatch ForwardDiff.jacobian!(result, g, x, cfg)

        y = fill(NaN, 2)
        cfg! = ForwardDiff.JacobianConfig(g!, y, x, ForwardDiff.Chunk{c}())
        @test_throws DimensionMismatch ForwardDiff.jacobian!(fill(NaN, 4, 2), g!, y, x, cfg!)
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

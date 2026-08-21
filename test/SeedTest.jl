module SeedTest

import ForwardDiff
using ForwardDiff: Partials
using LinearAlgebra
using Test

include("utils.jl")

# The windowed `seed_zero_partials!` is only ever called to clear a chunk that was just seeded, so
# clearing *too much* is harmless and no test written against the public API can distinguish a
# correctly bounded implementation from an unbounded one. These tests pin the window down directly:
# they seed every structural position with a marker whose partials are all nonzero, clear a window,
# and check exactly which positions lost their marker.
#
# The expected structural index sets are written out by hand rather than obtained from
# `structural_indices`, so a bug there cannot hide inside the assertions depending on it; one test
# ties the two together. They are linear indices of `x` in seeding order, so `index` and `count` are
# positions along the sequence, not array indices.
const SEED_CASES = (
    (rand(10),                    collect(1:10)),
    (UpperTriangular(rand(5, 5)), [i + (j - 1) * 5 for j in 1:5 for i in 1:j]),
    (Diagonal(rand(6, 6)),        collect(1:7:36)),
)

# Positions within `sidx` whose partials are zero.
zeroed_positions(duals, sidx) =
    [i for (i, idx) in enumerate(sidx) if iszero(ForwardDiff.partials(duals[idx]))]

# Compares over *every* index of `x`, not just the structural ones, so a bug misplacing values
# outside the structural set is visible. Off-structure reads are safe: the wrapper types return
# `zero(Dual)` without touching the (uninitialized) parent storage.
values_match(duals, x) = all(idx -> ForwardDiff.value(duals[idx]) == x[idx], eachindex(x))

function fill_marker!(duals, x, sidx, marker)
    D = eltype(duals)
    for idx in sidx
        duals[idx] = D(x[idx], marker)
    end
    return duals
end

@testset "seed_zero_partials!: $(nameof(typeof(x)))" for (x, sidx) in SEED_CASES
    cfg = ForwardDiff.GradientConfig(nothing, x, ForwardDiff.Chunk{3}())
    duals, seeds, indices = cfg.duals, cfg.seeds, cfg.indices
    N = ForwardDiff.npartials(eltype(duals))
    marker = Partials(ntuple(i -> Float64(i), N))
    nstruct = length(sidx)

    # everything below counts positions along `sidx`, so pin it to the implementation once. The
    # config builds its positions from its own work buffer, which has the structure of `x`.
    @test ForwardDiff.structural_indices(x) == sidx
    @test indices == sidx
    @test ForwardDiff.structural_length(x) == nstruct

    # `count` defaults to N
    fill_marker!(duals, x, sidx, marker)
    ForwardDiff.seed_zero_partials!(duals, x, indices, 4)
    @test zeroed_positions(duals, sidx) == collect(4:(4 + N - 1))
    @test values_match(duals, x)

    # an explicit `count` narrows the window; a zero-width window is a no-op, which is what makes
    # `xlen - N` safe as the `count` of chunk mode's tail clear
    @testset "index=$index count=$count" for (index, count, expected) in ((4, 2, 4:5), (1, 0, 1:0))
        fill_marker!(duals, x, sidx, marker)
        ForwardDiff.seed_zero_partials!(duals, x, indices, index, count)
        @test zeroed_positions(duals, sidx) == collect(expected)
        @test values_match(duals, x)
    end

    # a window overrunning the end is an error rather than a silently truncated chunk
    @test_throws BoundsError ForwardDiff.seed_zero_partials!(duals, x, indices, nstruct - 1, N)

    # the form without a window clears every structural position
    fill_marker!(duals, x, sidx, marker)
    ForwardDiff.seed_zero_partials!(duals, x, indices)
    @test zeroed_positions(duals, sidx) == collect(1:nstruct)
    @test values_match(duals, x)

    # `seed!` and `seed_zero_partials!` must agree on what "the chunk at `index`" is, or chunk mode
    # would leave stale seeds behind. `duals` enters each iteration fully cleared.
    @testset "round-trips seed! at index=$index" for index in unique((1, 4, nstruct - N + 1))
        ForwardDiff.seed!(duals, x, indices, index, seeds)
        @test zeroed_positions(duals, sidx) ==
              [i for i in 1:nstruct if !(index <= i <= index + N - 1)]
        ForwardDiff.seed_zero_partials!(duals, x, indices, index)
        @test zeroed_positions(duals, sidx) == collect(1:nstruct)
        @test values_match(duals, x)
    end

    # the form without an `index` seeds the first chunk, as vector mode needs
    ForwardDiff.seed_zero_partials!(duals, x, indices)
    ForwardDiff.seed!(duals, x, indices, seeds)
    @test zeroed_positions(duals, sidx) == collect((N + 1):nstruct)
    @test values_match(duals, x)
end

# An unassigned entry of `x` is mirrored into the work buffer rather than read, which only an `Array`
# buffer can do. See #842.
@testset "unassigned entries" begin
    # `x` is only read below, so every case shares these. Their first entry stays unassigned.
    M = Matrix{BigFloat}(undef, 2, 2)
    for i in 2:4
        M[i] = big(i)
    end
    v = Vector{BigFloat}(undef, 2)
    v[2] = big(2)

    @testset "$(nameof(typeof(x)))" for x in
            (M, adjoint(M), PermutedDimsArray(M, (2, 1)), view(M, :, :))
        cfg = ForwardDiff.GradientConfig(nothing, x, ForwardDiff.Chunk{2}())
        duals, indices = cfg.duals, cfg.indices

        ForwardDiff.seed_zero_partials!(duals, x, indices)
        @test !isassigned(duals, 1)
        @test all(i -> ForwardDiff.value(duals[i]) == x[i], 2:4)

        ForwardDiff.seed!(duals, x, indices, cfg.seeds)
        @test !isassigned(duals, 1)
    end

    @testset "$(nameof(typeof(x)))" for x in
            (UpperTriangular(M), LowerTriangular(M), Diagonal(v))
        cfg = ForwardDiff.GradientConfig(nothing, x, ForwardDiff.Chunk{2}())
        duals, indices = cfg.duals, cfg.indices

        @test_throws ArgumentError ForwardDiff.seed_zero_partials!(duals, x, indices)
        @test_throws ArgumentError ForwardDiff.seed!(duals, x, indices, cfg.seeds)
        @test_throws "ArgumentError: cannot differentiate at an input with an unassigned entry " *
                     "at index 1: that would leave an entry of the $(nameof(typeof(x))) work " *
                     "buffer unassigned" ForwardDiff.seed!(duals, x, indices, cfg.seeds)
    end
end

@testset "seed_hessian_chunk!: $(nameof(typeof(x)))" for (x, sidx) in SEED_CASES
    cfg = ForwardDiff.HessianConfig(nothing, x, ForwardDiff.Chunk{3}())
    (; duals, indices, iseeds, oseeds) = cfg
    nstruct = length(sidx)

    ForwardDiff.seed_hessian_chunk!(duals, x, indices, 1, nothing, nothing, nstruct)
    ForwardDiff.seed_hessian_chunk!(duals, x, indices, 4, iseeds, oseeds)
    @test [i for (i, idx) in enumerate(sidx) if !iszero(ForwardDiff.partials(ForwardDiff.value(duals[idx])))] == collect(4:6)
    @test [i for (i, idx) in enumerate(sidx) if !iszero(ForwardDiff.partials(duals[idx]))] == collect(4:6)
    @test all(idx -> ForwardDiff.value(ForwardDiff.value(duals[idx])) == x[idx], eachindex(x))

    ForwardDiff.seed_hessian_chunk!(duals, x, indices, 4, nothing, nothing)
    @test all(idx -> iszero(ForwardDiff.partials(ForwardDiff.value(duals[idx]))), sidx)
    @test all(idx -> iszero(ForwardDiff.partials(duals[idx])), sidx)
end

end # module

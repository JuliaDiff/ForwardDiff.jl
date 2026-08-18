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
# `structural_eachindex`, so a bug in that iterator cannot hide inside the assertions depending on
# it; one test ties the two together. Order is significant: `index` and `count` are positions along
# the sequence, not array indices. The sets are heterogeneous by design — `Vector` and `Diagonal`
# enumerate linear indices (the latter via `diagind`), `UpperTriangular` enumerates `CartesianIndex`
# in column-major order.
const SEED_CASES = (
    (rand(10),                    collect(1:10)),
    (UpperTriangular(rand(5, 5)), [CartesianIndex(i, j) for j in 1:5 for i in 1:j]),
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
    duals, seeds = cfg.duals, cfg.seeds
    N = ForwardDiff.npartials(eltype(duals))
    marker = Partials(ntuple(i -> Float64(i), N))
    nstruct = length(sidx)

    # everything below counts positions along `sidx`, so pin it to the implementation once
    @test collect(ForwardDiff.structural_eachindex(duals, x)) == sidx
    @test ForwardDiff.structural_length(x) == nstruct

    # the columns a Jacobian receives derivatives in, in the order the seeds are laid out in
    @test collect(ForwardDiff.structural_columns(zeros(2, length(x)), x)) == LinearIndices(x)[sidx]
    @test_throws DimensionMismatch ForwardDiff.structural_columns(zeros(2, length(x) + 1), x)

    # `count` defaults to N
    fill_marker!(duals, x, sidx, marker)
    ForwardDiff.seed_zero_partials!(duals, x, 4)
    @test zeroed_positions(duals, sidx) == collect(4:(4 + N - 1))
    @test values_match(duals, x)

    # an explicit `count` narrows the window; a `count` overrunning the end is clamped by
    # `Iterators.take` rather than throwing; a zero-width window is a no-op, which is what makes
    # `xlen - N` safe as the `count` of chunk mode's tail clear
    @testset "index=$index count=$count" for (index, count, expected) in
                                             ((4, 2, 4:5),
                                              (nstruct - 1, N, (nstruct - 1):nstruct),
                                              (1, 0, 1:0))
        fill_marker!(duals, x, sidx, marker)
        ForwardDiff.seed_zero_partials!(duals, x, index, count)
        @test zeroed_positions(duals, sidx) == collect(expected)
        @test values_match(duals, x)
    end

    # the 2-arg form clears every structural position
    fill_marker!(duals, x, sidx, marker)
    ForwardDiff.seed_zero_partials!(duals, x)
    @test zeroed_positions(duals, sidx) == collect(1:nstruct)
    @test values_match(duals, x)

    # `seed!` and `seed_zero_partials!` must agree on what "the chunk at `index`" is, or chunk mode
    # would leave stale seeds behind. `duals` enters each iteration fully cleared.
    @testset "round-trips seed! at index=$index" for index in unique((1, 4, nstruct - N + 1))
        ForwardDiff.seed!(duals, x, index, seeds)
        @test zeroed_positions(duals, sidx) ==
              [i for i in 1:nstruct if !(index <= i <= index + N - 1)]
        ForwardDiff.seed_zero_partials!(duals, x, index)
        @test zeroed_positions(duals, sidx) == collect(1:nstruct)
        @test values_match(duals, x)
    end
end

@testset "seed_hessian_chunk!: $(nameof(typeof(x)))" for (x, sidx) in SEED_CASES
    cfg = ForwardDiff.HessianConfig(nothing, x, ForwardDiff.Chunk{3}())
    duals = cfg.gradient_config.duals
    iseeds = cfg.jacobian_config.seeds
    oseeds = cfg.gradient_config.seeds
    nstruct = length(sidx)

    ForwardDiff.seed_hessian_chunk!(duals, x, 1, nothing, nothing, nstruct)
    ForwardDiff.seed_hessian_chunk!(duals, x, 4, iseeds, oseeds)
    @test [i for (i, idx) in enumerate(sidx) if !iszero(ForwardDiff.partials(ForwardDiff.value(duals[idx])))] == collect(4:6)
    @test [i for (i, idx) in enumerate(sidx) if !iszero(ForwardDiff.partials(duals[idx]))] == collect(4:6)
    @test all(idx -> ForwardDiff.value(ForwardDiff.value(duals[idx])) == x[idx], eachindex(x))

    ForwardDiff.seed_hessian_chunk!(duals, x, 4, nothing, nothing)
    @test all(idx -> iszero(ForwardDiff.partials(ForwardDiff.value(duals[idx]))), sidx)
    @test all(idx -> iszero(ForwardDiff.partials(duals[idx])), sidx)

    # The off-diagonal blocks of the sweep seed one layer at a time, so check that each seed lands
    # in the layer it was asked for and that the other one is left cleared, rather than only that
    # something was written.
    izero = zero(eltype(iseeds))
    ozero = zero(eltype(oseeds))
    @testset "$(iseeds === nothing ? "outer" : "inner") layer only" for (is, os) in
            ((iseeds, nothing), (nothing, oseeds))
        ForwardDiff.seed_hessian_chunk!(duals, x, 4, is, os)
        for (i, idx) in enumerate(sidx)
            chunkpos = i - 3
            inner = ForwardDiff.partials(ForwardDiff.value(duals[idx]))
            outer = ForwardDiff.partials(duals[idx])
            wanted = 1 <= chunkpos <= 3
            @test inner == (wanted && is !== nothing ? is[chunkpos] : izero)
            @test outer == (wanted && os !== nothing ? os[chunkpos] : ozero)
        end
        ForwardDiff.seed_hessian_chunk!(duals, x, 4, nothing, nothing)
    end
end

end # module

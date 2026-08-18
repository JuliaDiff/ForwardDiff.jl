module AllocationsTest

using ForwardDiff
using LinearAlgebra
using StaticArrays

include(joinpath(dirname(@__FILE__), "utils.jl"))

convert_test_574() = convert(ForwardDiff.Dual{Nothing,ForwardDiff.Dual{Nothing,ForwardDiff.Dual{Nothing,Float64,8},4},2}, 1.3)

@testset "Test seed!/seed_zero_partials! allocations" begin
    x = rand(1000)
    cfg = ForwardDiff.GradientConfig(nothing, x)
    duals = cfg.duals
    seeds = cfg.seeds

    allocs_seed!(args...) = @allocated ForwardDiff.seed!(args...)
    allocs_seed!(duals, x, seeds)
    @test iszero(allocs_seed!(duals, x, seeds))
    allocs_seed!(duals, x, 1, seeds)
    @test iszero(allocs_seed!(duals, x, 1, seeds))

    # the 4-arg form passes `count` as a runtime value, so it catches an inference regression at the
    # `_seed_zero_partials!` boundary that the forms defaulting `count` to `N` could hide
    allocs_szp!(args...) = @allocated ForwardDiff.seed_zero_partials!(args...)
    allocs_szp!(duals, x)
    @test iszero(allocs_szp!(duals, x))
    allocs_szp!(duals, x, 1)
    @test iszero(allocs_szp!(duals, x, 1))
    allocs_szp!(duals, x, 1, 4)
    @test iszero(allocs_szp!(duals, x, 1, 4))

    allocs_convert_test_574() = @allocated convert_test_574()
    allocs_convert_test_574()
    @test iszero(allocs_convert_test_574())
end

@testset "Test jacobian! allocations" begin
    # jacobian! should not allocate when called with a pre-allocated result Matrix.
    # Previously, reshape() inside extract_jacobian! allocated a wrapper
    # object that could not be elided under --check-bounds=yes.
    function allocs_jacobian!()
        f!(y, x) = (y .= x .^ 2)
        x = [1.0, 2.0, 3.0]
        y = similar(x)
        result = zeros(3, 3)
        cfg = ForwardDiff.JacobianConfig(f!, y, x)
        ForwardDiff.jacobian!(result, f!, y, x, cfg)  # warmup
        return @allocated ForwardDiff.jacobian!(result, f!, y, x, cfg)
    end
    @test iszero(allocs_jacobian!())
end

# `extract_gradient!`/`extract_jacobian!` take their positions from `x`, so mapping a structural
# position to an index of `x` must not allocate, whether or not `x` has structurally zero entries.
function allocs_structured_gradient!(result, x, chunk)
    f(z) = sum(abs2, z)
    fill!(result, false)
    cfg = ForwardDiff.GradientConfig(f, x, chunk)
    ForwardDiff.gradient!(result, f, x, cfg)  # warmup
    return @allocated ForwardDiff.gradient!(result, f, x, cfg)
end

function allocs_structured_jacobian!(x, chunk)
    f!(y, z) = (y[1] = sum(abs2, z); y[2] = sqrt(sum(abs2, z)); y)
    y = zeros(2)
    result = zeros(2, length(x))
    cfg = ForwardDiff.JacobianConfig(f!, y, x, chunk)
    ForwardDiff.jacobian!(result, f!, y, x, cfg)  # warmup
    return @allocated ForwardDiff.jacobian!(result, f!, y, x, cfg)
end

@testset "Test gradient!/jacobian! allocations for $(nameof(typeof(x)))" for (x, nstruct) in (
        (rand(6, 6),                  36),
        (LowerTriangular(rand(6, 6)), 21),
        (UpperTriangular(rand(6, 6)), 21),
        (Diagonal(rand(6, 6)),         6),
    )
    # A result shaped like `x` receives a derivative in every entry it stores, a dense one has the
    # entries off the structure of `x` zeroed as well. The chunk sizes cover chunk and vector mode.
    for result in (similar(x), zeros(size(x))), chunk_size in (2, nstruct)
        chunk = ForwardDiff.Chunk{chunk_size}()
        @test iszero(allocs_structured_gradient!(result, x, chunk))
        @test iszero(allocs_structured_jacobian!(x, chunk))
    end
end

@testset "allocation-free nested StaticArray jacobian" begin
    # test that nested jacobians of StaticArrays do not allocate. 
    # This is a regression test for issue #798, where the inner jacobian was allocating
    toy_f(x) = SVector(x[1]^2 * x[2], sin(x[1]) + x[2]^3)

    function toy_J_flat(x::SVector{2,T}) where {T}
        y = ForwardDiff.jacobian(toy_f, x)
        return SVector{2,T}(y[1], y[2])
    end

    function toy_nested_jacobian(x0::SVector{2,T}) where {T}
        return ForwardDiff.jacobian(toy_J_flat, x0)
    end

    function allocs_jacobian()
        x0 = SVector(1.0, 2.0)
        return @allocated toy_nested_jacobian(x0)
    end
    @test iszero(allocs_jacobian())
end

end

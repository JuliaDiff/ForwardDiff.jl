module AllocationsTest

using ForwardDiff
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

    hcfg = ForwardDiff.HessianConfig(nothing, x)
    hduals = hcfg.gradient_config.duals
    iseeds = hcfg.jacobian_config.seeds
    oseeds = hcfg.gradient_config.seeds
    allocs_hseed!(args...) = @allocated ForwardDiff.seed_hessian_chunk!(args...)
    allocs_hseed!(hduals, x, 1, iseeds, oseeds)
    @test iszero(allocs_hseed!(hduals, x, 1, iseeds, oseeds))
    allocs_hseed!(hduals, x, 1, nothing, nothing, 4)
    @test iszero(allocs_hseed!(hduals, x, 1, nothing, nothing, 4))

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

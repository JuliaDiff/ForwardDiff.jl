module AllocationsTest

using ForwardDiff

include(joinpath(dirname(@__FILE__), "utils.jl"))

convert_test_574() = convert(ForwardDiff.Dual{Nothing,ForwardDiff.Dual{Nothing,ForwardDiff.Dual{Nothing,Float64,8},4},2}, 1.3)

@testset "Test seed! allocations" begin
    x = rand(1000)
    cfg = ForwardDiff.GradientConfig(nothing, x)
    duals = cfg.duals
    seeds = cfg.seeds
    seed = cfg.seeds[1]

    allocs_seed!(args...) = @allocated ForwardDiff.seed!(args...)
    allocs_seed!(duals, x, seeds)
    @test iszero(allocs_seed!(duals, x, seeds))
    allocs_seed!(duals, x, seed)
    @test iszero(allocs_seed!(duals, x, seed))
    allocs_seed!(duals, x, 1, seeds)
    @test iszero(allocs_seed!(duals, x, 1, seeds))
    allocs_seed!(duals, x, 1, seed)
    @test iszero(allocs_seed!(duals, x, 1, seed))

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

end

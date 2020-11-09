module AllocationsTest

using ForwardDiff

include(joinpath(dirname(@__FILE__), "utils.jl"))

@testset "Test seed! allocations" begin
    x = rand(1000)
    cfg = ForwardDiff.GradientConfig(nothing, x)
    duals = cfg.duals
    seeds = cfg.seeds
    seed = cfg.seeds[1]

    alloc = @allocated ForwardDiff.seed!(duals, x, seeds)
    alloc = @allocated ForwardDiff.seed!(duals, x, seeds)
    @test alloc == 0

    alloc = @allocated ForwardDiff.seed!(duals, x, seed)
    alloc = @allocated ForwardDiff.seed!(duals, x, seed)
    @test alloc == 0

    index = 1
    alloc = @allocated ForwardDiff.seed!(duals, x, index, seeds)
    alloc = @allocated ForwardDiff.seed!(duals, x, index, seeds)
    @test alloc == 0

    index = 1
    alloc = @allocated ForwardDiff.seed!(duals, x, index, seed)
    alloc = @allocated ForwardDiff.seed!(duals, x, index, seed)
    @test alloc == 0
end

end
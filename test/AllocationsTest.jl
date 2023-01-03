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
    
    alloc = @allocated convert_test_574()
    alloc = @allocated convert_test_574()
    @test alloc == 0
  
end

@testset "Test extract_gradient! allocations" begin
    T = Float64
    @testset "vector-mode size(result)=$size" for size in [(4,), (2,2)]
        dual = ForwardDiff.Dual(0, (rand(T, size...)...,))
        y = Array{T}(undef, size)
        alloc = @allocated ForwardDiff.extract_gradient!(Nothing, y, dual)
        alloc = @allocated ForwardDiff.extract_gradient!(Nothing, y, dual)
        @test alloc == 0
    end
    @testset "chunk-mode size(result)=$size" for size in [(DEFAULT_CHUNK_THRESHOLD+1,), (DEFAULT_CHUNK_THRESHOLD+1, DEFAULT_CHUNK_THRESHOLD+1)]
        Npartials = DEFAULT_CHUNK_THRESHOLD÷2
        dual = ForwardDiff.Dual(0, (rand(T, Npartials...)...,))
        y = Array{T}(undef, size)
        alloc = @allocated ForwardDiff.extract_gradient_chunk!(Nothing, y, dual, 2, Npartials)
        alloc = @allocated ForwardDiff.extract_gradient_chunk!(Nothing, y, dual, 2, Npartials)
        @test alloc == 0
        alloc = @allocated ForwardDiff.extract_gradient_chunk!(Nothing, y, dual, 2, Npartials-1)
        alloc = @allocated ForwardDiff.extract_gradient_chunk!(Nothing, y, dual, 2, Npartials-1)
        @test alloc == 0
    end
end

end

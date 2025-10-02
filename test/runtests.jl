using ForwardDiff, Test, Random

SEED = trunc(Int, time())
println("##### Random.seed!($SEED), on VERSION == $VERSION")
Random.seed!(SEED)

@testset "ForwardDiff.jl" begin
    t0 = time()
    @testset "Partials" begin
        println("##### Testing Partials...")
        t = @elapsed include("PartialsTest.jl")
        println("##### done (took $t seconds).")
    end
    @testset "Dual" begin
        println("##### Testing Dual...")
        t = @elapsed include("DualTest.jl")
        println("##### done (took $t seconds).")
    end
    @testset "Derivatives" begin
        println("##### Testing derivative functionality...")
        t = @elapsed include("DerivativeTest.jl")
        println("##### done (took $t seconds).")
    end
    @testset "Gradients" begin
        println("##### Testing gradient functionality...")
        t = @elapsed include("GradientTest.jl")
        println("##### done (took $t seconds).")
    end
    @testset "Jacobians" begin
        println("##### Testing jacobian functionality...")
        t = @elapsed include("JacobianTest.jl")
        println("##### done (took $t seconds).")
    end
    @testset "Hessians" begin
        println("##### Testing hessian functionality...")
        t = @elapsed include("HessianTest.jl")
        println("##### done (took $t seconds).")
    end
    @testset "Perturbation Confusion" begin
        println("##### Testing perturbation confusion functionality...")
        t = @elapsed include("ConfusionTest.jl")
        println("##### done (took $t seconds).")
    end
    @testset "Miscellaneous" begin
        println("##### Testing miscellaneous functionality...")
        t = @elapsed include("MiscTest.jl")
        println("##### done (took $t seconds).")
    end
    @testset "Allocations" begin
        println("##### Testing allocations...")
        t = @elapsed include("AllocationsTest.jl")
        println("##### done (took $t seconds).")
    end
    @testset "QA" begin
        println("##### QA testing...")
        t = @elapsed include("QATest.jl")
        println("##### done (took ", t, " seconds).")
    end
    println("##### Running all ForwardDiff tests took $(time() - t0) seconds.")
end

using ForwardDiff, Test

@testset "ForwardDiff" begin
    @testset "Partials" begin
        println("Testing Partials...")
        t = @elapsed include("PartialsTest.jl")
        println("done (took $t seconds).")
    end
    @testset "Dual" begin
        println("Testing Dual...")
        t = @elapsed include("DualTest.jl")
        println("done (took $t seconds).")
    end
    @testset "Derivative" begin
        println("Testing derivative functionality...")
        t = @elapsed include("DerivativeTest.jl")
        println("done (took $t seconds).")
    end
    @testset "Gradient" begin
        println("Testing gradient functionality...")
        t = @elapsed include("GradientTest.jl")
        println("done (took $t seconds).")
    end
    @testset "Jacobian" begin
        println("Testing jacobian functionality...")
        t = @elapsed include("JacobianTest.jl")
        println("done (took $t seconds).")
    end
    @testset "Hessian" begin
        println("Testing hessian functionality...")
        t = @elapsed include("HessianTest.jl")
        println("done (took $t seconds).")
    end
    @testset "Perturbation confusion" begin
        println("Testing perturbation confusion functionality...")
        t = @elapsed include("ConfusionTest.jl")
        println("done (took $t seconds).")
    end
    @testset "Miscellaneous" begin
        println("Testing miscellaneous functionality...")
        t = @elapsed include("MiscTest.jl")
        println("done (took $t seconds).")
    end
    if VERSION >= v"1.5-"
        @testset "Allocations" begin
            println("Testing allocations...")
            t = @elapsed include("AllocationsTest.jl")
            println("done (took $t seconds).")
        end
    end
end
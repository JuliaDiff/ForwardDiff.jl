using ForwardDiff, Test, Random
using Pkg

const GROUP = get(ENV, "GROUP", "All")

function activate_gpu_env()
    Pkg.activate("gpu")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
end

SEED = trunc(Int, time())
println("##### Random.seed!($SEED), on VERSION == $VERSION")
Random.seed!(SEED)

if GROUP == "All" || GROUP == "Core"
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
        println("##### Running all ForwardDiff tests took $(time() - t0) seconds.")
    end
elseif GROUP == "CUDA"
    @testset "ForwardDiff.jl CUDA" begin
        activate_gpu_env()
        @testset "CUDA Gradients" begin
            println("##### Testing CUDA gradients...")
            t = @elapsed include("CUDAGradientTest.jl")
            println("##### done (took $t seconds).")
        end
        @testset "CUDA Jacobians" begin
            println("##### Testing CUDA Jacobians...")
            t = @elapsed include("CUDAJacobianTest.jl")
            println("##### done (took $t seconds).")
        end
    end
end
module QATest

using ForwardDiff
using LinearAlgebra
using Test

using JET: @test_opt

@testset "JET" begin
    # issue #778
    @test_opt ForwardDiff.derivative(identity, 1.0)
    @test_opt ForwardDiff.gradient(only, [1.0], ForwardDiff.GradientConfig(only, [1.0], ForwardDiff.Chunk{1}()))
    @test_opt ForwardDiff.jacobian(identity, [1.0], ForwardDiff.JacobianConfig(identity, [1.0], ForwardDiff.Chunk{1}()))
    @test_opt ForwardDiff.hessian(only, [1.0], ForwardDiff.HessianConfig(only, [1.0], ForwardDiff.Chunk{1}()))

    # extraction iterates the structural positions of `x` for these
    @testset "$(nameof(typeof(x)))" for x in (LowerTriangular(rand(3, 3)),
                                              UpperTriangular(rand(3, 3)),
                                              Diagonal(rand(3, 3)))
        @test_opt ForwardDiff.gradient(first, x, ForwardDiff.GradientConfig(first, x, ForwardDiff.Chunk{2}()))
        @test_opt ForwardDiff.jacobian(vec, x, ForwardDiff.JacobianConfig(vec, x, ForwardDiff.Chunk{2}()))
    end
end

end # module

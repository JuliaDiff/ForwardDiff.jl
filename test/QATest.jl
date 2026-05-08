module QATest

using ForwardDiff
using Test

using JET: @test_opt

@testset "JET" begin
    # issue #778
    @test_opt ForwardDiff.derivative(identity, 1.0)
    @test_opt ForwardDiff.gradient(only, [1.0], ForwardDiff.GradientConfig(only, [1.0], ForwardDiff.Chunk{1}()))
    @test_opt ForwardDiff.jacobian(identity, [1.0], ForwardDiff.JacobianConfig(identity, [1.0], ForwardDiff.Chunk{1}()))
    @test_opt ForwardDiff.hessian(only, [1.0], ForwardDiff.HessianConfig(only, [1.0], ForwardDiff.Chunk{1}()))
end

end # module

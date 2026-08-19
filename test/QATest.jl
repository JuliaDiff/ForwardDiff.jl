module QATest

using ForwardDiff
using Test

import JET

if isdefined(JET, :JET_AVAILABLE) || JET.JET_AVAILABLE
    @testset "JET" begin
        # issue #778
        JET.@test_opt ForwardDiff.derivative(identity, 1.0)
        JET.@test_opt ForwardDiff.gradient(only, [1.0], ForwardDiff.GradientConfig(only, [1.0], ForwardDiff.Chunk{1}()))
        JET.@test_opt ForwardDiff.jacobian(identity, [1.0], ForwardDiff.JacobianConfig(identity, [1.0], ForwardDiff.Chunk{1}()))
        JET.@test_opt ForwardDiff.hessian(only, [1.0], ForwardDiff.HessianConfig(only, [1.0], ForwardDiff.Chunk{1}()))
    end
end

end # module

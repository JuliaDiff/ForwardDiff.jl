module QATest

using ForwardDiff
using LinearAlgebra
using Test

import JET

if !isdefined(JET, :JET_AVAILABLE) || JET.JET_AVAILABLE
    @testset "JET" begin
        # issue #778
        JET.@test_opt ForwardDiff.derivative(identity, 1.0)
        JET.@test_opt ForwardDiff.gradient(only, [1.0], ForwardDiff.GradientConfig(only, [1.0], ForwardDiff.Chunk{1}()))
        JET.@test_opt ForwardDiff.jacobian(identity, [1.0], ForwardDiff.JacobianConfig(identity, [1.0], ForwardDiff.Chunk{1}()))
        JET.@test_opt ForwardDiff.hessian(only, [1.0], ForwardDiff.HessianConfig(only, [1.0], ForwardDiff.Chunk{1}()))

        # seeding and extraction index the structural positions held by the config for these
        @testset "$(nameof(typeof(x)))" for x in (LowerTriangular(rand(3, 3)),
                                                  UpperTriangular(rand(3, 3)),
                                                  Diagonal(rand(3, 3)))
            JET.@test_opt ForwardDiff.gradient(first, x, ForwardDiff.GradientConfig(first, x, ForwardDiff.Chunk{2}()))
            JET.@test_opt ForwardDiff.jacobian(vec, x, ForwardDiff.JacobianConfig(vec, x, ForwardDiff.Chunk{2}()))
        end
    end
end

end # module

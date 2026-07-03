module QATest

using ForwardDiff
using Test

using JET: @test_opt

# On Julia prereleases JET's analysis lags the compiler and produces
# false-positive runtime-dispatch reports (e.g. on 1.13.0-rc1 for concrete
# Base broadcast/view/CartesianIndex code reached by these calls: native
# inference of every flagged frame is concrete and the kernels are
# allocation-free, so nothing is actually mis-inferred in ForwardDiff;
# https://github.com/aviatesk/JET.jl/issues/839). Only run the JET tests on
# released Julia versions.
if isempty(VERSION.prerelease)
    @testset "JET" begin
        # issue #778
        @test_opt ForwardDiff.derivative(identity, 1.0)
        @test_opt ForwardDiff.gradient(only, [1.0], ForwardDiff.GradientConfig(only, [1.0], ForwardDiff.Chunk{1}()))
        @test_opt ForwardDiff.jacobian(identity, [1.0], ForwardDiff.JacobianConfig(identity, [1.0], ForwardDiff.Chunk{1}()))
        @test_opt ForwardDiff.hessian(only, [1.0], ForwardDiff.HessianConfig(only, [1.0], ForwardDiff.Chunk{1}()))
    end
end

end # module

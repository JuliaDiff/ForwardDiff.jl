module QATest

using ForwardDiff
using Test

using JET: @test_opt

# On Julia 1.13, JET produces false-positive runtime-dispatch reports for
# concrete Base broadcast/view/CartesianIndex code reached by these calls:
# native inference of every flagged frame is concrete and the kernels are
# allocation-free, so nothing is actually mis-inferred in ForwardDiff.
# Re-enable once JET supports Julia 1.13; tracked in
# https://github.com/aviatesk/JET.jl/issues/839
if VERSION < v"1.13.0-"
    @testset "JET" begin
        # issue #778
        @test_opt ForwardDiff.derivative(identity, 1.0)
        @test_opt ForwardDiff.gradient(only, [1.0], ForwardDiff.GradientConfig(only, [1.0], ForwardDiff.Chunk{1}()))
        @test_opt ForwardDiff.jacobian(identity, [1.0], ForwardDiff.JacobianConfig(identity, [1.0], ForwardDiff.Chunk{1}()))
        @test_opt ForwardDiff.hessian(only, [1.0], ForwardDiff.HessianConfig(only, [1.0], ForwardDiff.Chunk{1}()))
    end
end

end # module

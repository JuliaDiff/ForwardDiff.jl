module DeprecatedTest

using Test
using ForwardDiff, DiffResults

using ForwardDiff: AbstractConfig, GradientConfig,
                   JacobianConfig, HessianConfig,
                   MultithreadConfig

include(joinpath(dirname(@__FILE__), "utils.jl"))

function similar_duals(a::AbstractArray, b::AbstractArray)
    return typeof(a) == typeof(b) && size(a) == size(b)
end

similar_duals(a::Tuple, b::Tuple) = all(similar_duals.(a, b))

function similar_config(a::AbstractConfig, b::AbstractConfig)
    return a.seeds == b.seeds && similar_duals(a.duals, b.duals)
end

function similar_config(a::HessianConfig, b::HessianConfig)
    return (similar_config(a.gradient_config, b.gradient_config) &&
            similar_config(a.jacobian_config, b.jacobian_config))
end

x = rand(3)
y = rand(3)
out = DiffResults.HessianResult(x)
N = 1
chunk = ForwardDiff.Chunk{N}()

@info("The following tests print lots of deprecation warnings on purpose.")

@test similar_config(GradientConfig{N}(x), GradientConfig(nothing, x, chunk))
@test similar_config(JacobianConfig{N}(x), JacobianConfig(nothing, x, chunk))
@test similar_config(JacobianConfig{N}(y, x), JacobianConfig(nothing, y, x, chunk))
@test similar_config(HessianConfig{N}(x), HessianConfig(nothing, x, chunk))
@test similar_config(HessianConfig{N}(out, x), HessianConfig(nothing, out, x, chunk))
@test similar_config(MultithreadConfig(GradientConfig(nothing, x, chunk)), GradientConfig(nothing, x, chunk))

@info("Deprecation testing is now complete, so any further deprecation warnings are real.")

end # module

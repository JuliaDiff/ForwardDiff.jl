###############
# API methods #
###############

@compat const AllowedHessianConfig{F,M} = Union{HessianConfig{Tag{F,M}}, HessianConfig{Tag{Void,M}}}

hessian(f, x, cfg::HessianConfig) = throw(ConfigMismatchError(f, cfg))
hessian!(out, f, x, cfg::HessianConfig) = throw(ConfigMismatchError(f, cfg))
hessian!(out::DiffResult, f, x, cfg::HessianConfig) = throw(ConfigMismatchError(f, cfg))

function hessian{F,M}(f::F, x, cfg::AllowedHessianConfig{F,M} = HessianConfig(f, x))
    ∇f = y -> gradient(f, y, cfg.gradient_config)
    return jacobian(∇f, x, cfg.jacobian_config)
end

function hessian!{F,M}(out, f::F, x, cfg::AllowedHessianConfig{F,M} = HessianConfig(f, x))
    ∇f = y -> gradient(f, y, cfg.gradient_config)
    jacobian!(out, ∇f, x, cfg.jacobian_config)
    return out
end

function hessian!{F,M}(out::DiffResult, f::F, x, cfg::AllowedHessianConfig{F,M} = HessianConfig(out, f, x))
    ∇f! = (y, z) -> begin
        result = DiffResult(zero(eltype(y)), y)
        gradient!(result, f, z, cfg.gradient_config)
        DiffBase.value!(out, value(DiffBase.value(result)))
        return y
    end
    jacobian!(DiffBase.hessian(out), ∇f!, DiffBase.gradient(out), x, cfg.jacobian_config)
    return out
end

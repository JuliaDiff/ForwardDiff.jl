###############
# API methods #
###############

const AllowedHessianConfig{F,H} = Union{HessianConfig{Tag{F,H}}, HessianConfig{Tag{Void,H}}}

hessian(f, x, cfg::HessianConfig) = throw(ConfigMismatchError(f, cfg))
hessian!(out, f, x, cfg::HessianConfig) = throw(ConfigMismatchError(f, cfg))
hessian!(out::DiffResult, f, x, cfg::HessianConfig) = throw(ConfigMismatchError(f, cfg))

function hessian(f::F, x, cfg::AllowedHessianConfig{F,H} = HessianConfig(f, x)) where {F,H}
    ∇f = y -> gradient(f, y, cfg.gradient_config)
    return jacobian(∇f, x, cfg.jacobian_config)
end

function hessian!(out, f::F, x, cfg::AllowedHessianConfig{F,H} = HessianConfig(f, x)) where {F,H}
    ∇f = y -> gradient(f, y, cfg.gradient_config)
    jacobian!(out, ∇f, x, cfg.jacobian_config)
    return out
end

function hessian!(out::DiffResult, f::F, x, cfg::AllowedHessianConfig{F,H} = HessianConfig(f, out, x)) where {F,H}
    ∇f! = (y, z) -> begin
        result = DiffResult(zero(eltype(y)), y)
        gradient!(result, f, z, cfg.gradient_config)
        DiffBase.value!(out, value(DiffBase.value(result)))
        return y
    end
    jacobian!(DiffBase.hessian(out), ∇f!, DiffBase.gradient(out), x, cfg.jacobian_config)
    return out
end

hessian(f::F, x::SArray) where {F} = jacobian(y -> gradient(f, y), x)
hessian_vals(f::F, x::SArray) where {F} = jacobian_vals(y -> gradient(f, y), x)

hessian!(out, f::F, x::SArray) where {F} = jacobian!(out, y -> gradient(f, y), x)

hessian!(out::DiffResult, f::F, x::SArray) where {F} = hessian!(out, f, x, HessianConfig(f, out, x))

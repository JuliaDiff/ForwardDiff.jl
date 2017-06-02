###############
# API methods #
###############

const AllowedHessianConfig{F,H} = Union{HessianConfig{Tag{F,H}}, HessianConfig{Tag{Void,H}}}

hessian(f, x::AbstractArray, cfg::HessianConfig) = throw(ConfigMismatchError(f, cfg))
hessian!(result::AbstractArray, f, x::AbstractArray, cfg::HessianConfig) = throw(ConfigMismatchError(f, cfg))
hessian!(result::DiffResult, f, x::AbstractArray, cfg::HessianConfig) = throw(ConfigMismatchError(f, cfg))

"""
    ForwardDiff.hessian(f, x::AbstractArray, cfg::HessianConfig = HessianConfig(f, x))

Return `H(f)` (i.e. `J(∇(f))`) evaluated at `x`, assuming `f` is called as `f(x)`.

This method assumes that `isa(f(x), Real)`.
"""
function hessian(f::F, x::AbstractArray, cfg::AllowedHessianConfig{F,H} = HessianConfig(f, x)) where {F,H}
    ∇f = y -> gradient(f, y, cfg.gradient_config)
    return jacobian(∇f, x, cfg.jacobian_config)
end

"""
    ForwardDiff.hessian!(result::AbstractArray, f, x::AbstractArray, cfg::HessianConfig = HessianConfig(f, x))

Compute `H(f)` (i.e. `J(∇(f))`) evaluated at `x` and store the result(s) in `result`,
assuming `f` is called as `f(x)`.

This method assumes that `isa(f(x), Real)`.
"""
function hessian!(result::AbstractArray, f::F, x::AbstractArray, cfg::AllowedHessianConfig{F,H} = HessianConfig(f, x)) where {F,H}
    ∇f = y -> gradient(f, y, cfg.gradient_config)
    jacobian!(result, ∇f, x, cfg.jacobian_config)
    return result
end

"""
    ForwardDiff.hessian!(result::DiffResult, f, x::AbstractArray, cfg::HessianConfig = HessianConfig(f, result, x))

Exactly like `ForwardDiff.hessian!(result::AbstractArray, f, x::AbstractArray, cfg::HessianConfig)`, but
because `isa(result, DiffResult)`, `cfg` is constructed as `HessianConfig(f, result, x)` instead of
`HessianConfig(f, x)`.
"""
function hessian!(result::DiffResult, f::F, x::AbstractArray, cfg::AllowedHessianConfig{F,H} = HessianConfig(f, result, x)) where {F,H}
    ∇f! = (y, z) -> begin
        inner_result = DiffResult(zero(eltype(y)), y)
        gradient!(inner_result, f, z, cfg.gradient_config)
        DiffBase.value!(result, value(DiffBase.value(inner_result)))
        return y
    end
    jacobian!(DiffBase.hessian(result), ∇f!, DiffBase.gradient(result), x, cfg.jacobian_config)
    return result
end

hessian(f::F, x::SArray) where {F} = jacobian(y -> gradient(f, y), x)
hessian(f::F, x::SArray, cfg::AllowedHessianConfig{F,H}) where {F,H} = hessian(f, x)

hessian!(result::AbstractArray, f::F, x::SArray) where {F} = jacobian!(result, y -> gradient(f, y), x)
hessian!(result::AbstractArray, f::F, x::SArray, cfg::AllowedHessianConfig{F,H}) where {F,H} = hessian!(result, f, x)

hessian!(result::DiffResult, f::F, x::SArray) where {F} = hessian!(result, f, x, HessianConfig(f, result, x))
hessian!(result::DiffResult, f::F, x::SArray, cfg::AllowedHessianConfig{F,H}) where {F,H} = hessian!(result, f, x)

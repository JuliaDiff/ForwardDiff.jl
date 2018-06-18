###############
# API methods #
###############

"""
    ForwardDiff.hessian(f, x::AbstractArray, cfg::HessianConfig = HessianConfig(f, x), check=Val{true}())

Return `H(f)` (i.e. `J(∇(f))`) evaluated at `x`, assuming `f` is called as `f(x)`.

This method assumes that `isa(f(x), Real)`.

Set `check` to `Val{false}()` to disable tag checking. This can lead to perturbation confusion, so should be used with care.
"""
function hessian(f, x::AbstractArray, cfg::HessianConfig{T} = HessianConfig(f, x), ::Val{CHK}=Val{true}()) where {T,CHK}
    CHK && checktag(T, f, x)
    ∇f = y -> gradient(f, y, cfg.gradient_config, Val{false}())
    return jacobian(∇f, x, cfg.jacobian_config, Val{false}())
end

"""
    ForwardDiff.hessian!(result::AbstractArray, f, x::AbstractArray, cfg::HessianConfig = HessianConfig(f, x), check=Val{true}())

Compute `H(f)` (i.e. `J(∇(f))`) evaluated at `x` and store the result(s) in `result`,
assuming `f` is called as `f(x)`.

This method assumes that `isa(f(x), Real)`.

Set `check` to `Val{false}()` to disable tag checking. This can lead to perturbation confusion, so should be used with care.
"""
function hessian!(result::AbstractArray, f, x::AbstractArray, cfg::HessianConfig{T} = HessianConfig(f, x), ::Val{CHK}=Val{true}()) where {T,CHK}
    CHK && checktag(T, f, x)
    ∇f = y -> gradient(f, y, cfg.gradient_config, Val{false}())
    jacobian!(result, ∇f, x, cfg.jacobian_config, Val{false}())
    return result
end


# We use this struct below instead of an
# equivalent closure in order to avoid
# JuliaLang/julia#15276-related performance
# issues. See #316.
mutable struct InnerGradientForHess{R,C,F}
    result::R
    cfg::C
    f::F
end

function (g::InnerGradientForHess)(y, z)
    inner_result = DiffResult(zero(eltype(y)), y)
    gradient!(inner_result, g.f, z, g.cfg.gradient_config, Val{false}())
    g.result = DiffResults.value!(g.result, value(DiffResults.value(inner_result)))
    return y
end

"""
    ForwardDiff.hessian!(result::DiffResult, f, x::AbstractArray, cfg::HessianConfig = HessianConfig(f, result, x), check=Val{true}())

Exactly like `ForwardDiff.hessian!(result::AbstractArray, f, x::AbstractArray, cfg::HessianConfig)`, but
because `isa(result, DiffResult)`, `cfg` is constructed as `HessianConfig(f, result, x)` instead of
`HessianConfig(f, x)`.

Set `check` to `Val{false}()` to disable tag checking. This can lead to perturbation confusion, so should be used with care.
"""
function hessian!(result::DiffResult, f, x::AbstractArray, cfg::HessianConfig{T} = HessianConfig(f, result, x), ::Val{CHK}=Val{true}()) where {T,CHK}
    CHK && checktag(T, f, x)
    ∇f! = InnerGradientForHess(result, cfg, f)
    jacobian!(DiffResults.hessian(result), ∇f!, DiffResults.gradient(result), x, cfg.jacobian_config, Val{false}())
    return ∇f!.result
end

hessian(f, x::SArray) = jacobian(y -> gradient(f, y), x)
hessian(f, x::SArray, cfg::HessianConfig) = hessian(f, x)
hessian(f, x::SArray, cfg::HessianConfig, ::Val) = hessian(f, x)

hessian!(result::AbstractArray, f, x::SArray) = jacobian!(result, y -> gradient(f, y), x)

hessian!(result::MutableDiffResult, f, x::SArray) = hessian!(result, f, x, HessianConfig(f, result, x))

hessian!(result::ImmutableDiffResult, f, x::SArray, cfg::HessianConfig) = hessian!(result, f, x)
hessian!(result::ImmutableDiffResult, f, x::SArray, cfg::HessianConfig, ::Val) = hessian!(result, f, x)

function hessian!(result::ImmutableDiffResult, f::F, x::SArray{S,V}) where {F,S,V}
    T = typeof(Tag(f,V))
    d1 = dualize(T, x)
    d2 = dualize(T, d1)
    fd2 = f(d2)
    val = value(T,value(T,fd2))
    grad = extract_gradient(T,value(T,fd2), x)
    hess = extract_jacobian(T,partials(T,fd2), x)
    result = DiffResults.hessian!(result, hess)
    result = DiffResults.gradient!(result, grad)
    result = DiffResults.value!(result, val)
    return result
end

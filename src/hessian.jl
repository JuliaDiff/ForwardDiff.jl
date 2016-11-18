###############
# API methods #
###############

function hessian{F}(f::F, x, cfg::AbstractConfig = HessianConfig(x))
    ∇f = y -> gradient(f, y, gradient_config(cfg))
    return jacobian(∇f, x, jacobian_config(cfg))
end

function hessian!{F}(out, f::F, x, cfg::AbstractConfig = HessianConfig(x))
    ∇f = y -> gradient(f, y, gradient_config(cfg))
    jacobian!(out, ∇f, x, jacobian_config(cfg))
    return out
end

function hessian!{F}(out::DiffResult, f::F, x, cfg::AbstractConfig = HessianConfig(out, x))
    ∇f! = (y, z) -> begin
        result = DiffResult(zero(eltype(y)), y)
        gradient!(result, f, z, gradient_config(cfg))
        DiffBase.value!(out, value(DiffBase.value(result)))
        return y
    end
    jacobian!(DiffBase.hessian(out), ∇f!, DiffBase.gradient(out), x, jacobian_config(cfg))
    return out
end

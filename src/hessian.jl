###############
# API methods #
###############

function hessian{F}(f::F, x, cfg::AbstractConfig = HessianConfig(x))
    ∇f = y -> gradient(f, y, gradient_options(cfg))
    return jacobian(∇f, x, jacobian_options(cfg))
end

function hessian!{F}(out, f::F, x, cfg::AbstractConfig = HessianConfig(x))
    ∇f = y -> gradient(f, y, gradient_options(cfg))
    jacobian!(out, ∇f, x, jacobian_options(cfg))
    return out
end

function hessian!{F}(out::DiffResult, f::F, x, cfg::AbstractConfig = HessianConfig(out, x))
    ∇f! = (y, z) -> begin
        result = DiffResult(zero(eltype(y)), y)
        gradient!(result, f, z, gradient_options(cfg))
        DiffBase.value!(out, value(DiffBase.value(result)))
        return y
    end
    jacobian!(DiffBase.hessian(out), ∇f!, DiffBase.gradient(out), x, jacobian_options(cfg))
    return out
end

######################
# Hessian API Errors #
######################

const HESS_OPTIONS_ERR_MSG = "To use `hessian`/`hessian!` with options, use `HessianConfig` or `Multithread(::HessianConfig)` instead of `Config`."

hessian{F}(f::F, x, ::Config) = error(HESS_OPTIONS_ERR_MSG)
hessian!{F}(out, f::F, x, ::Config) = error(HESS_OPTIONS_ERR_MSG)
hessian!{F}(::DiffResult, f::F, x, ::Config) = error(HESS_OPTIONS_ERR_MSG)

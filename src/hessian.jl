###############
# API methods #
###############

function hessian(f, x, opts::AbstractOptions = HessianOptions(x))
    ∇f = y -> gradient(f, y, gradient_options(opts))
    return jacobian(∇f, x, jacobian_options(opts))
end

function hessian!(out, f, x, opts::AbstractOptions = HessianOptions(x))
    ∇f = y -> gradient(f, y, gradient_options(opts))
    jacobian!(out, ∇f, x, jacobian_options(opts))
    return out
end

function hessian!(out::DiffResult, f, x, opts::AbstractOptions = HessianOptions(out, x))
    ∇f! = (y, z) -> begin
        result = DiffResult(zero(eltype(y)), y)
        gradient!(result, f, z, gradient_options(opts))
        DiffBase.value!(out, value(DiffBase.value(result)))
        return y
    end
    jacobian!(DiffBase.hessian(out), ∇f!, DiffBase.gradient(out), x, jacobian_options(opts))
    return out
end

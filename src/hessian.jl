###############
# API methods #
###############

function hessian(f, x, outer_opts = Options(x), inner_opts = Options(outer_opts))
    ∇f = y -> gradient(f, y, inner_opts)
    return jacobian(∇f, x, outer_opts)
end

function hessian!(out, f, x,
                  outer_opts = Options(x),
                  inner_opts = Options(outer_opts))
    ∇f = y -> gradient(f, y, inner_opts)
    jacobian!(out, ∇f, x, outer_opts)
    return out
end

function hessian!(out::DiffResult, f, x,
                  outer_opts = Options(DiffBase.gradient(out), x),
                  inner_opts = Options(outer_opts))
    ∇f! = (y, z) -> begin
        result = DiffResult(zero(eltype(y)), y)
        gradient!(result, f, z, inner_opts)
        DiffBase.value!(out, value(DiffBase.value(result)))
        return y
    end
    jacobian!(DiffBase.hessian(out), ∇f!, DiffBase.gradient(out), x, outer_opts)
    return out
end

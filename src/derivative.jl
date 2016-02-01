############################
# @derivative!/@derivative #
############################

const DERIVATIVE_KWARG_ORDER = (:all,)
const DERIVATIVE_F_KWARG_ORDER = (:all, :output_mutates)

macro derivative!(args...)
    args, kwargs = separate_kwargs(args)
    arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS, DERIVATIVE_KWARG_ORDER)
    return esc(:(ForwardDiff.derivative!($(args...), $(arranged_kwargs...))))
end

macro derivative(args...)
    args, kwargs = separate_kwargs(args)
    if length(args) == 1
        arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS, DERIVATIVE_F_KWARG_ORDER)
    else
        arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS, DERIVATIVE_KWARG_ORDER)
    end
    return esc(:(ForwardDiff.derivative($(args...), $(arranged_kwargs...))))
end

##########################
# derivative!/derivative #
##########################

function derivative!(f, output::Array, x::Real, A::DataType)
    return handle_deriv_result!(output, f(DiffNumber(x, one(x))), A)
end

function derivative(f, x::Real, A::DataType)
    return handle_deriv_result(f(DiffNumber(x, one(x))), A)
end

@generated function derivative{output_mutates}(f, A::DataType, ::Type{Val{output_mutates}})
    if output_mutates
        return quote
            d!(output::Array, x::Real) = derivative!(f, output, x, A)
            return d!
        end
    else
        return quote
            d(x::Real) = derivative(f, x, A)
            return d
        end
    end
end

###############################
# handling derivative results #
###############################

handle_deriv_result(result::DiffNumber, ::Type{Val{false}}) = partials(result, 1)

function handle_deriv_result(result::DiffNumber, ::Type{Val{true}})
    return value(result), partials(result, 1)
end

function handle_deriv_result{T}(result::Array{T}, A::DataType)
    return handle_deriv_result!(similar(result, numtype(T)), result, A)
end

function handle_deriv_result!(output::Array, result::Array, ::Type{Val{true}})
    valoutput = similar(output)
    for i in eachindex(result)
        valoutput[i] = value(result[i])
        output[i] = partials(result[i], 1)
    end
    return valoutput, output
end

function handle_deriv_result!(output::Array, result::Array, ::Type{Val{false}})
    for i in eachindex(result)
        output[i] = partials(result[i], 1)
    end
    return output
end

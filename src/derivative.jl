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

function derivative!{ALL}(f, out, x, ::Type{Val{ALL}})
    return handle_deriv_result!(out, f(DiffNumber(x, one(x))), Val{ALL})
end

function derivative{ALL}(f, x, ::Type{Val{ALL}})
    return handle_deriv_result(f(DiffNumber(x, one(x))), Val{ALL})
end

@generated function derivative{ALL, MUTATES}(f, ::Type{Val{ALL}}, ::Type{Val{MUTATES}})
    if MUTATES
        return quote
            d!(out, x) = derivative!(f, out, x, Val{ALL})
            return d!
        end
    else
        return quote
            d(x) = derivative(f, x, Val{ALL})
            return d
        end
    end
end

###############################
# handling derivative results #
###############################

handle_deriv_result(result::DiffNumber, ::Type{Val{false}}) = partials(result, 1)
handle_deriv_result(result::DiffNumber, ::Type{Val{true}}) = value(result), partials(result, 1)

function handle_deriv_result{ALL}(result, ::Type{Val{ALL}})
    output = similar(result, numtype(eltype(result)))
    return handle_deriv_result!(output, result, Val{ALL})
end

function handle_deriv_result!(output, result, ::Type{Val{true}})
    valoutput = similar(output)
    for i in eachindex(result)
        valoutput[i] = value(result[i])
        output[i] = partials(result[i], 1)
    end
    return valoutput, output
end

function handle_deriv_result!(output, result, ::Type{Val{false}})
    for i in eachindex(result)
        output[i] = partials(result[i], 1)
    end
    return output
end

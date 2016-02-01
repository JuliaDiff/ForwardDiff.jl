########################
# @jacobian!/@jacobian #
########################

const JACOBIAN_KWARG_ORDER = (:all, :chunk, :input_length, :output_length)
const JACOBIAN_F_KWARG_ORDER = (:all, :chunk, :input_length, :input_mutates, :output_length, :output_mutates)

macro jacobian!(args...)
    args, kwargs = separate_kwargs(args)
    arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS, JACOBIAN_KWARG_ORDER)
    return esc(:(ForwardDiff.jacobian!($(args...), $(arranged_kwargs...))))
end

macro jacobian(args...)
    args, kwargs = separate_kwargs(args)
    if length(args) == 1
        arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS, JACOBIAN_F_KWARG_ORDER)
    else
        arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS, JACOBIAN_KWARG_ORDER)
    end

    return esc(:(ForwardDiff.jacobian($(args...), $(arranged_kwargs...))))
end

######################
# jacobian!/jacobian #
######################

@generated function jacobian!(f, output::Matrix, x::Vector, A::DataType,
                              N::DataType, IL::DataType, OL::DataType)
    # TODO
end

@generated function jacobian(f, x::Vector, A::DataType, N::DataType)
    # TODO
end

@generated function jacobian{allresults, chunk, input_mutates, output_mutates}(f,
                                                                               ::Type{Val{allresults}},
                                                                               ::Type{Val{chunk}},
                                                                               ::Type{Val{input_mutates}},
                                                                               ::Type{Val{output_mutates}})
    # TODO
end

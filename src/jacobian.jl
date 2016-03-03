########################
# @jacobian!/@jacobian #
########################

const JACOBIAN_KWARG_ORDER = (:allresults, :chunk, :input_length, :multithread)
const JACOBIAN_F_KWARG_ORDER = (:allresults, :chunk, :input_length, :output_length, :multithread, :mutates)

macro jacobian!(args...)
    args, kwargs = separate_kwargs(args)
    arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS, JACOBIAN_KWARG_ORDER)
    return esc(:(ForwardDiff._jacobian!($(args...), $(arranged_kwargs...))))
end

macro jacobian(args...)
    args, kwargs = separate_kwargs(args)
    if length(args) == 1
        arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS, JACOBIAN_F_KWARG_ORDER)
    else
        arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS, JACOBIAN_KWARG_ORDER)
    end
    return esc(:(ForwardDiff._jacobian($(args...), $(arranged_kwargs...))))
end

##################
# JacobianResult #
##################

immutable JacobianResult{V, J} <: ForwardDiffResult
    value::V
    jacobian::J
end

jacobian(result::JacobianResult) = copy(result.jacobian)
jacobian!(arr, result::JacobianResult) = copy!(arr, result.jacobian)

value(result::JacobianResult) = map(value, result.value)
value!(arr, result::JacobianResult) = map!(value, arr, result.value)

########################
# _jacobian!/_jacobian #
########################

@generated function _jacobian!(f, output, x, allresults::DataType, chunk::DataType,
                              input_length::DataType, multithread::DataType)
    input_length_value = value(input_length) == nothing ? :(length(x)) : value(input_length)
    return_statement = value(allresults) ? :(result) : :(output)
    return quote
        result = _call_jacobian!(f, output, x, chunk, Val{$(input_length_value)}, multithread)
        return $(return_statement)
    end
end

@generated function _jacobian(f, x, allresults::DataType, chunk::DataType, input_length::DataType, multithread::DataType)
    return_statement = value(allresults) ? :(result) : :(result.jacobian)
    return quote
        result = _jacobian!(f, DummyOutput(), x, Val{true}, chunk, input_length, multithread)
        return $(return_statement)
    end
end

@generated function _jacobian(f, allresults::DataType, chunk::DataType, input_length::DataType,
                              output_length::DataType, multithread::DataType, mutates::DataType)
    if value(output_length) != nothing
        targetdef = quote
            targetf = x -> begin
                output = cachefetch!(compat_threadid(), eltype(x), Val{$(value(output_length))})
                f(output, x)
                return output
            end
        end
    else
        targetdef = :(targetf = f)
    end
    if value(mutates)
        return quote
            $(targetdef)
            j!(output, x) = _jacobian!(targetf, output, x, allresults, chunk, input_length, multithread)
            return j!
        end
    else
        return quote
            $(targetdef)
            j(x) = _jacobian(targetf, x, allresults, chunk, input_length, multithread)
            return j
        end
    end
end

#######################
# workhorse functions #
#######################

@generated function _call_jacobian!(f, output, x, chunk, input_length, multithread)
    input_length_value = value(input_length)
    chunk_value = value(chunk) == nothing ? pick_chunk(input_length_value) : value(chunk)
    @assert chunk_value <= input_length_value
    use_chunk_mode = chunk_value != input_length_value
    if use_chunk_mode
        return :(_jacobian_chunk_mode!(f, output, x, Val{$(chunk_value)}, Val{$(input_length_value)}))
    else
        return :(_jacobian_vector_mode!(f, output, x, Val{$(input_length_value)}))
    end
end

@generated function _jacobian_vector_mode!{input_length}(f, outarg, x, ::Type{Val{input_length}})
    if outarg <: DummyOutput
        outputdef = :(output = Matrix{S}(output_length, input_length))
    else
        outputdef = quote
            @assert size(outarg) == (output_length, input_length)
            output = outarg
        end
    end
    return quote
        @assert input_length == length(x)
        T = eltype(x)
        cache = get_cache(cachefetch!(Val{input_length}, Val{input_length}, T))
        workvec = cache.workvec
        seed_partials = cache.partials
        @simd for i in 1:input_length
            @inbounds workvec[i] = DiffNumber{input_length,T}(x[i], seed_partials[i])
        end
        result = f(workvec)
        S, output_length = numtype(eltype(result)), length(result)
        $(outdef)
        for j in 1:input_length, i in 1:output_length
            @inbounds output[i, j] = partials(result[i], j)
        end
        return JacobianResult(result, output)
    end
end

@generated function _jacobian_chunk_mode!{chunk, input_length}(f, outarg, x, ::Type{chunk}, ::Type{Val{input_length}})
    if outarg <: DummyOutput
        outputdef = :(output = Matrix{S}(output_length, input_length))
    else
        outputdef = quote
            @assert size(outarg) == (output_length, input_length)
            output = outarg
        end
    end
    remainder = compute_remainder(input_length, chunk)
    fill_length = input_length - remainder
    return quote
        @assert input_length == length(x)
        T = eltype(x)
        cache = get_cache(cachefetch!(Val{input_length}, Val{chunk}, T))
        workvec = cache.workvec
        seed_partials = cache.partials
        seed_partials_remainder = cache.partials_remainder
        zero_partials  = zero(Partials{chunk,T})

        # do the first chunk manually, so that we can infer the dimensions
        # of the output matrix if necessary
        @simd for i in 1:input_length
            @inbounds workvec[i] = DiffNumber{chunk,T}(x[i], zero_partials)
        end
        @simd for i in 1:chunk
            @inbounds workvec[i] = DiffNumber{chunk,T}(x[i], seed_partials[i])
        end
        chunk_result = f(workvec)
        S, output_length = numtype(eltype(chunk_result)), length(result)
        $(outputdef)
        for i in 1:chunk
            @simd for r in 1:nrows
                @inbounds output[r, i] = partials(chunk_result[r], i)
            end
            @inbounds workvec[i] = DiffNumber{chunk,T}(x[i], zero_partials)
        end

        # now do the rest of the chunks until we hit the fill_length
        for c in $(chunk + 1):$(chunk):$(fill_length)
            offset = c - 1
            @simd for i in 1:chunk
                j = i + offset
                @inbounds workvec[j] = DiffNumber{chunk,T}(x[j], seed_partials[i])
            end
            chunk_result = f(workvec)
            for i in 1:chunk
                j = i + offset
                @simd for r in 1:nrows
                    @inbounds output[r, j] = partials(chunk_result[r], i)
                end
                @inbounds workvec[j] = DiffNumber{chunk,T}(x[j], zero_partials)
            end
        end

        # do the final remaining chunk manually
        @simd for i in 1:$(remainder)
            j = $(fill_length) + i
            @inbounds workvec[j] = DiffNumber{chunk,T}(x[j], seed_partials_remainder[i])
        end
        chunk_result = f(workvec)
        @simd for i in 1:$(remainder)
            j = $(fill_length) + i
            @simd for r in 1:nrows
                @inbounds output[r, j] = partials(chunk_result[r], i)
            end
        end

        return JacobianResult(result, output)
    end
end

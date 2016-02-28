########################
# @gradient!/@gradient #
########################

const GRADIENT_KWARG_ORDER = (:allresults, :chunk, :input_length, :multithread)
const GRADIENT_F_KWARG_ORDER = (:allresults, :chunk, :input_length, :multithread, :mutates)

macro gradient!(args...)
    args, kwargs = separate_kwargs(args)
    arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS, GRADIENT_KWARG_ORDER)
    return esc(:(ForwardDiff._gradient!($(args...), $(arranged_kwargs...))))
end

macro gradient(args...)
    args, kwargs = separate_kwargs(args)
    if length(args) == 1
        arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS, GRADIENT_F_KWARG_ORDER)
    else
        arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS, GRADIENT_KWARG_ORDER)
    end
    return esc(:(ForwardDiff._gradient($(args...), $(arranged_kwargs...))))
end

##################
# GradientResult #
##################

immutable GradientResult{V, G} <: ForwardDiffResult
    value::V
    gradient::G
end

gradient(result::GradientResult) = copy(result.gradient)
gradient!(arr, result::GradientResult) = copy!(arr, result.gradient)

value(result::GradientResult) = copy(result.value)

######################
# gradient!/gradient #
######################

@generated function _gradient!(f, output, x, allresults::DataType, chunk::DataType,
                              input_length::DataType, multithread::DataType)
    input_length_value = value(input_length) == nothing ? :(length(x)) : value(input_length)
    return_statement = value(allresults) ? :(result) : :(output)
    return quote
        result = _call_gradient!(f, output, x, chunk, Val{$(input_length_value)}, multithread)
        return $(return_statement)
    end
end

@generated function _gradient(f, x, allresults::DataType, chunk::DataType, input_length::DataType, multithread::DataType)
    return_statement = value(allresults) ? :(result) : :(result.gradient)
    return quote
        result = _gradient!(f, DummyOutput(), x, Val{true}, chunk, input_length, multithread)
        return $(return_statement)
    end
end

@generated function _gradient(f, allresults::DataType, chunk::DataType, input_length::DataType,
                              multithread::DataType, mutates::DataType)
    if value(mutates)
        return quote
            g!(output, x) = _gradient!(f, output, x, allresults, chunk, input_length, multithread)
            return g!
        end
    else
        return quote
            g(x) = _gradient(f, x, allresults, chunk, input_length, multithread)
            return g
        end
    end
end

#######################
# workhorse functions #
#######################

# `_call_gradient!` is the entry point that is called by the API functions. It decides which
# workhorse function to call based on the provided parameters. Note that if a chunk size
# isn't given by an upstream caller, `_call_gradient!` picks one based on the input length.
@generated function _call_gradient!(f, output, x, chunk, input_length, multithread)
    input_length_value = value(input_length)
    chunk_value = value(chunk) == nothing ? pick_chunk(input_length_value) : value(chunk)
    @assert chunk_value <= input_length_value
    use_chunk_mode = chunk_value != input_length_value
    if use_chunk_mode
        if value(multithread) && IS_MULTITHREADED_JULIA
            gradfunc! = :_multi_gradient_chunk_mode!
        else
            gradfunc! = :_gradient_chunk_mode!
        end
        return :($(gradfunc!)(f, output, x, Val{$(chunk_value)}, Val{$(input_length_value)}))
    else
        return :(_gradient_vector_mode!(f, output, x, Val{$(input_length_value)}))
    end
end

@generated function _gradient_vector_mode!{input_length}(f, outarg, x, ::Type{Val{input_length}})
    if outarg <: DummyOutput
        outputdef = :(output = Vector{S}(input_length))
    else
        outputdef = quote
            @assert length(outarg) == input_length
            output = outarg
        end
    end
    return quote
        @assert input_length == length(x)
        T = eltype(x)
        tid = compat_threadid()
        seed_partials = cachefetch!(tid, Partials{input_length,T})
        workvec = cachefetch!(tid, DiffNumber{input_length,T}, Val{input_length})
        @simd for i in 1:input_length
            @inbounds workvec[i] = DiffNumber{input_length,T}(x[i], seed_partials[i])
        end
        result = f(workvec)
        S = numtype(result)
        $(outputdef)
        @simd for i in 1:input_length
            @inbounds output[i] = partials(result, i)
        end
        return GradientResult(value(result), output)
    end
end

@generated function _gradient_chunk_mode!{chunk,input_length}(f, outarg, x, ::Type{Val{chunk}}, ::Type{Val{input_length}})
    if outarg <: DummyOutput
        outputdef = :(output = Vector{S}(input_length))
    else
        outputdef = quote
            @assert length(outarg) == input_length
            output = outarg
        end
    end
    remainder = input_length % chunk == 0 ? chunk : input_length % chunk
    fill_length = input_length - remainder
    reseed_partials = remainder == chunk ? :() : :(seed_partials = cachefetch!(tid, Partials{chunk,T}, Val{$(remainder)}))
    return quote
        @assert input_length == length(x)
        T = eltype(x)
        tid = compat_threadid()
        zero_partials = zero(Partials{chunk,T})
        seed_partials = cachefetch!(tid, Partials{chunk,T})
        workvec = cachefetch!(tid, DiffNumber{chunk,T}, Val{input_length})

        # do first chunk manually so that we can infer the output eltype, if necessary
        @simd for i in 1:input_length
            @inbounds workvec[i] = DiffNumber{chunk,T}(x[i], zero_partials)
        end
        @simd for i in 1:chunk
            @inbounds workvec[i] = DiffNumber{chunk,T}(x[i], seed_partials[i])
        end
        chunk_result = f(workvec)
        S = numtype(chunk_result)
        $(outputdef)
        @simd for i in 1:chunk
            @inbounds output[i] = partials(chunk_result, i)
            @inbounds workvec[i] = DiffNumber{chunk,T}(x[i], zero_partials)
        end

        # do the rest of the chunks
        for c in $(chunk + 1):$(chunk):$(fill_length)
            offset = c - 1
            @simd for i in 1:chunk
                j = i + offset
                @inbounds workvec[j] = DiffNumber{chunk,T}(x[j], seed_partials[i])
            end
            chunk_result = f(workvec)
            @simd for i in 1:chunk
                j = i + offset
                @inbounds output[j] = partials(chunk_result, i)
                @inbounds workvec[j] = DiffNumber{chunk,T}(x[j], zero_partials)
            end
        end

        # Performing the final chunk manually seems to triggers some additional
        # optimization heuristics, which results in more efficient memory allocation
        $(reseed_partials)
        @simd for i in 1:$(remainder)
            j = $(fill_length) + i
            @inbounds workvec[j] = DiffNumber{chunk,T}(x[j], seed_partials[i])
        end
        chunk_result = f(workvec)
        @simd for i in 1:$(remainder)
            j = $(fill_length) + i
            @inbounds output[j] = partials(chunk_result, i)
            @inbounds workvec[j] = DiffNumber{chunk,T}(x[j], zero_partials)
        end

        return GradientResult(value(chunk_result), output)
    end
end

if IS_MULTITHREADED_JULIA
    @generated function _multi_gradient_chunk_mode!{chunk,input_length}(f, outarg, x, ::Type{Val{chunk}}, ::Type{Val{input_length}})
        if outarg <: DummyOutput
            outputdef = :(output = Vector{S}(input_length))
        else
            outputdef = quote
                @assert length(outarg) == input_length
                output = outarg
            end
        end
        remainder = input_length % chunk == 0 ? chunk : input_length % chunk
        fill_length = input_length - remainder
        reseed_partials = remainder == chunk ? :() : :(seed_partials = cachefetch!(tid, Partials{chunk,T}, Val{$(remainder)}))
        return quote
            @assert input_length == length(x)
            T = eltype(x)
            tid = compat_threadid()
            zero_partials = zero(Partials{chunk,T})
            seed_partials = cachefetch!(tid, Partials{chunk,T})
            workvecs = cachefetch!(DiffNumber{chunk,T}, Val{input_length})

            Base.Threads.@threads for t in 1:NTHREADS
                # see https://github.com/JuliaLang/julia/issues/14948
                local workvec = workvecs[t]
                @simd for i in 1:input_length
                    @inbounds workvec[i] = DiffNumber{chunk,T}(x[i], zero_partials)
                end
            end

            # do first chunk manually so that we can infer the output eltype, if necessary
            workvec = workvecs[tid]
            @simd for i in 1:chunk
                @inbounds workvec[i] = DiffNumber{chunk,T}(x[i], seed_partials[i])
            end
            chunk_result = f(workvec)
            S = numtype(chunk_result)
            $(outputdef)
            @simd for i in 1:chunk
                @inbounds output[i] = partials(chunk_result, i)
                @inbounds workvec[i] = DiffNumber{chunk,T}(x[i], zero_partials)
            end

            # do the rest of the chunks
            Base.Threads.@threads for c in $(chunk + 1):$(chunk):$(fill_length)
                local workvec = workvecs[compat_threadid()]
                local offset = c - 1
                @simd for i in 1:chunk
                    local j = i + offset
                    @inbounds workvec[j] = DiffNumber{chunk,T}(x[j], seed_partials[i])
                end
                local chunk_result = f(workvec)
                @simd for i in 1:chunk
                    local j = i + offset
                    @inbounds output[j] = partials(chunk_result, i)
                    @inbounds workvec[j] = DiffNumber{chunk,T}(x[j], zero_partials)
                end
            end

            # Performing the final chunk manually seems to triggers some additional
            # optimization heuristics, which results in more efficient memory allocation
            $(reseed_partials)
            workvec = workvecs[tid]
            @simd for i in 1:$(remainder)
                j = $(fill_length) + i
                @inbounds workvec[j] = DiffNumber{chunk,T}(x[j], seed_partials[i])
            end
            chunk_result = f(workvec)
            @simd for i in 1:$(remainder)
                j = $(fill_length) + i
                @inbounds output[j] = partials(chunk_result, i)
                @inbounds workvec[j] = DiffNumber{chunk,T}(x[j], zero_partials)
            end

            return GradientResult(value(chunk_result), output)
        end
    end
end

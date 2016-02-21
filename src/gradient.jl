########################
# @gradient!/@gradient #
########################

const GRADIENT_KWARG_ORDER = (:all, :chunk, :input_length, :multithread)
const GRADIENT_F_KWARG_ORDER =  (:all, :chunk, :input_length, :multithread, :output_mutates)

macro gradient!(args...)
    args, kwargs = separate_kwargs(args)
    arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS, GRADIENT_KWARG_ORDER)
    return esc(:(ForwardDiff.gradient!($(args...), $(arranged_kwargs...))))
end

macro gradient(args...)
    args, kwargs = separate_kwargs(args)
    if length(args) == 1
        arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS, GRADIENT_F_KWARG_ORDER)
    else
        arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS, GRADIENT_KWARG_ORDER)
    end
    return esc(:(ForwardDiff.gradient($(args...), $(arranged_kwargs...))))
end

######################
# gradient!/gradient #
######################

@generated function gradient!(f, out, x, all::DataType, chunk::DataType,
                              input_length::DataType, multithread::DataType)
    input_length_value = value(input_length) == nothing ? :(length(x)) : value(input_length)
    return_statement = value(all) ? :(val::$(eltype(out)), out::$(out)) : :(out::$(out))
    return quote
        val, _ = call_gradient!(f, out, x, chunk, Val{$(input_length_value)}, multithread)
        return $(return_statement)
    end
end

function gradient(f, x, all::DataType, chunk::DataType, input_length::DataType, multithread::DataType)
    return gradient!(f, similar(x), x, all, chunk, input_length, multithread)
end

@generated function gradient(f, all::DataType, chunk::DataType, input_length::DataType,
                             multithread::DataType, output_mutates::DataType)
    if value(output_mutates)
        R = value(all) ? :(Tuple{eltype(out),typeof(out)}) : :(typeof(out))
        return quote
            g!(out, x) = gradient!(f, out, x, all, chunk, input_length, multithread)::$(R)
            return g!
        end
    else
        R = value(all) ? :(Tuple{eltype(x),typeof(x)}) : :(typeof(x))
        return quote
            g(x) = gradient(f, x, all, chunk, input_length, multithread)::$(R)
            return g
        end
    end
end

######################################
# call_gradient!/workhorse functions #
######################################

# `call_gradient!` is the entry point that is called by the API functions. It decides which
# workhorse function to call based on the provided parameters. Note that if a chunk size
# isn't given by an upstream caller, `call_gradient!` picks one based on the input length.
@generated function call_gradient!(f, out, x, chunk, input_length, multithread)
    input_length_value = value(input_length)
    chunk_value = value(chunk) == nothing ? pick_chunk(input_length_value) : value(chunk)
    use_chunk_mode = chunk_value != input_length_value
    if use_chunk_mode
        gradfunc! = value(multithread) && IS_MULTITHREADED_JULIA ? :multi_gradient_chunk_mode! : :gradient_chunk_mode!
        return :($(gradfunc!)(f, out, x, Val{$(chunk_value)}, Val{$(input_length_value)}))
    else
        return :(gradient_vector_mode!(f, out, x, Val{$(input_length_value)}))
    end
end

function gradient_vector_mode!{input_length}(f, out, x, ::Type{Val{input_length}})
    @assert input_length == length(x) == length(out)
    S, T = eltype(out), eltype(x)
    tid = compat_threadid()
    seed_partials::Vector{Partials{input_length,T}} = cachefetch!(tid, Partials{input_length,T})
    workvec::Vector{DiffNumber{input_length,T}} = cachefetch!(tid, DiffNumber{input_length,T}, Val{input_length})

    @simd for i in 1:input_length
        @inbounds workvec[i] = DiffNumber{input_length,T}(x[i], seed_partials[i])
    end

    result::DiffNumber{input_length,S} = f(workvec)

    @simd for i in 1:input_length
        @inbounds out[i] = partials(result, i)
    end

    return value(result)::S, out
end

@generated function gradient_chunk_mode!{chunk,input_length}(f, out, x, ::Type{Val{chunk}}, ::Type{Val{input_length}})
    remainder = input_length % chunk == 0 ? chunk : input_length % chunk
    fill_length = input_length - remainder
    reseed_partials = remainder == chunk ? :() : :(seed_partials = cachefetch!(tid, Partials{chunk,T}, Val{$(remainder)}))
    return quote
        @assert input_length == length(x) == length(out)
        S, T = eltype(out), eltype(x)
        tid = compat_threadid()
        seed_partials::Vector{Partials{chunk,T}} = cachefetch!(tid, Partials{chunk,T})
        workvec::Vector{DiffNumber{chunk,T}} = cachefetch!(tid, DiffNumber{chunk,T}, Val{input_length})
        pzeros = zero(Partials{chunk,T})

        @simd for i in 1:input_length
            @inbounds workvec[i] = DiffNumber{chunk,T}(x[i], pzeros)
        end

        for c in 1:$(chunk):$(fill_length)
            @simd for i in 1:chunk
                j = i + c - 1
                @inbounds workvec[j] = DiffNumber{chunk,T}(x[j], seed_partials[i])
            end
            local result::DiffNumber{chunk,S} = f(workvec)
            @simd for i in 1:chunk
                j = i + c - 1
                @inbounds out[j] = partials(result, i)
                @inbounds workvec[j] = DiffNumber{chunk,T}(x[j], pzeros)
            end
        end

        # Performing the final chunk manually seems to triggers some additional
        # optimization heuristics, which results in more efficient memory allocation
        $(reseed_partials)

        @simd for i in 1:$(remainder)
            j = $(fill_length) + i
            @inbounds workvec[j] = DiffNumber{chunk,T}(x[j], seed_partials[i])
        end
        result::DiffNumber{chunk,S} = f(workvec)
        @simd for i in 1:$(remainder)
            j = $(fill_length) + i
            @inbounds out[j] = partials(result, i)
            @inbounds workvec[j] = DiffNumber{chunk,T}(x[j], pzeros)
        end

        return value(result)::S, out
    end
end

if IS_MULTITHREADED_JULIA
    @generated function multi_gradient_chunk_mode!{chunk,input_length}(f, out, x, ::Type{Val{chunk}}, ::Type{Val{input_length}})
        remainder = input_length % chunk == 0 ? chunk : input_length % chunk
        fill_length = input_length - remainder
        reseed_partials = remainder == chunk ? :() : :(seed_partials = cachefetch!(tid, Partials{chunk,T}, Val{$(remainder)}))
        return quote
            @assert input_length == length(x) == length(out)
            S, T = eltype(out), eltype(x)
            tid = compat_threadid()
            seed_partials::Vector{Partials{chunk,T}} = cachefetch!(tid, Partials{chunk,T})
            workvecs::NTuple{NTHREADS, Vector{DiffNumber{chunk,T}}} = cachefetch!(DiffNumber{chunk,T}, Val{input_length})
            pzeros = zero(Partials{chunk,T})

            Base.Threads.@threads for t in 1:NTHREADS
                # see https://github.com/JuliaLang/julia/issues/14948
                local workvec = workvecs[t]
                @simd for i in 1:input_length
                    @inbounds workvec[i] = DiffNumber{chunk,T}(x[i], pzeros)
                end
            end

            Base.Threads.@threads for c in 1:$(chunk):$(fill_length)
                local workvec = workvecs[compat_threadid()]
                @simd for i in 1:chunk
                    j = i + c - 1
                    @inbounds workvec[j] = DiffNumber{chunk,T}(x[j], seed_partials[i])
                end
                local result::DiffNumber{chunk,S} = f(workvec)
                @simd for i in 1:chunk
                    j = i + c - 1
                    @inbounds out[j] = partials(result, i)
                    @inbounds workvec[j] = DiffNumber{chunk,T}(x[j], pzeros)
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

            result::DiffNumber{chunk,S} = f(workvec)

            @simd for i in 1:$(remainder)
                j = $(fill_length) + i
                @inbounds out[j] = partials(result, i)
                @inbounds workvec[j] = DiffNumber{chunk,T}(x[j], pzeros)
            end

            return value(result)::S, out
        end
    end
end

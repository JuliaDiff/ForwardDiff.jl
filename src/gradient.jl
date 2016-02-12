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

@generated function gradient!{S,ALL,CHUNK,LEN,MULTITHREAD}(f, out::AbstractVector{S}, x::AbstractVector,
                                                           ::Type{Val{ALL}}, ::Type{Val{CHUNK}},
                                                           ::Type{Val{LEN}}, ::Type{Val{MULTITHREAD}})
    return quote
        val, grad = call_gradient!(f, out, x, Val{CHUNK}, Val{$(LEN == nothing ? :(length(x)) : LEN)}, Val{MULTITHREAD})
        return $(ALL ? :(val::S, out::$(out)) : :(out::$(out)))
    end
end

function gradient{ALL,CHUNK,LEN,MULTITHREAD}(f, x::AbstractVector, ::Type{Val{ALL}}, ::Type{Val{CHUNK}},
                                             ::Type{Val{LEN}}, ::Type{Val{MULTITHREAD}})
    return gradient!(f, similar(x), x, Val{ALL}, Val{CHUNK}, Val{LEN}, Val{MULTITHREAD})
end

@generated function gradient{ALL,CHUNK,LEN,MULTITHREAD,MUTATES}(f, ::Type{Val{ALL}}, ::Type{Val{CHUNK}},
                                                                ::Type{Val{LEN}}, ::Type{Val{MULTITHREAD}},
                                                                ::Type{Val{MUTATES}})
    if MUTATES
        R = ALL ? :(Tuple{S,typeof(out)}) : :(typeof(out))
        return quote
            g!{S}(out::AbstractVector{S}, x::AbstractVector) = gradient!(f, out, x, Val{ALL}, Val{CHUNK}, Val{LEN}, Val{MULTITHREAD})::$(R)
            return g!
        end
    else
        R = ALL ? :(Tuple{S,typeof(x)}) : :(typeof(x))
        return quote
            g{S}(x::AbstractVector{S}) = gradient(f, x, Val{ALL}, Val{CHUNK}, Val{LEN}, Val{MULTITHREAD})::$(R)
            return g
        end
    end
end

##################
# calc_gradient! #
##################
# The below code is pretty ugly, so here's an overview:
#
# `call_gradient!` is the entry point that is called by the API functions. If a chunk size
# isn't given by an upstream caller, `call_gradient!` picks one based on the input length.
#
# `calc_gradient!` is the workhorse function - it generates code for calculating the
# gradient in chunk-mode or vector-mode, depending on the input length and chunk size.
#
# `multi_calc_gradient!` is just like `_calc_gradient!`, but uses Julia's multithreading
# capabilities when performing calculations in chunk-mode.
#
# `VEC_MODE_EXPR` is a constant expression for vector-mode that provides the function body
# for `calc_gradient!` and `multi_calc_gradient!` when chunk size equals input length.
#
# `calc_gradient_expr` takes in a vector-mode expression body or chunk-mode expression body
# and returns a completed function body with the input body injected in the correct place.

@generated function call_gradient!{S,CHUNK,LEN,MULTITHREAD}(f, out::AbstractVector{S}, x::AbstractVector,
                                                            ::Type{Val{CHUNK}}, ::Type{Val{LEN}},
                                                            ::Type{Val{MULTITHREAD}})
    gradf! = MULTITHREAD && IS_MULTITHREADED_JULIA ? :multi_calc_gradient! : :calc_gradient!
    return :($(gradf!)(f, out, x, Val{$(CHUNK == nothing ? pick_chunk(LEN) : CHUNK)}, Val{LEN})::Tuple{S, $(out)})
end

@generated function calc_gradient!{S,T,CHUNK,LEN}(f, out::AbstractVector{S}, x::AbstractVector{T},
                                                  ::Type{Val{CHUNK}}, ::Type{Val{LEN}})
    if CHUNK == LEN
        body = VEC_MODE_EXPR
    else
        remainder = LEN % CHUNK == 0 ? CHUNK : LEN % CHUNK
        fill_length = LEN - remainder
        reseed_partials = remainder == CHUNK ? :() : :(seed_partials = cachefetch!(tid, Partials{CHUNK,T}, Val{$(remainder)}))
        body = quote
            workvec::Vector{DiffNumber{CHUNK,T}} = cachefetch!(tid, DiffNumber{CHUNK,T}, Val{LEN})
            pzeros = zero(Partials{CHUNK,T})

            @simd for i in 1:LEN
                @inbounds workvec[i] = DiffNumber{CHUNK,T}(x[i], pzeros)
            end

            for c in 1:$(CHUNK):$(fill_length)
                @simd for i in 1:CHUNK
                    j = i + c - 1
                    @inbounds workvec[j] = DiffNumber{CHUNK,T}(x[j], seed_partials[i])
                end
                local result::DiffNumber{CHUNK,S} = f(workvec)
                @simd for i in 1:CHUNK
                    j = i + c - 1
                    @inbounds out[j] = partials(result, i)
                    @inbounds workvec[j] = DiffNumber{CHUNK,T}(x[j], pzeros)
                end
            end

            # Performing the final chunk manually seems to triggers some additional
            # optimization heuristics, which results in more efficient memory allocation
            $(reseed_partials)
            @simd for i in 1:$(remainder)
                j = $(fill_length) + i
                @inbounds workvec[j] = DiffNumber{CHUNK,T}(x[j], seed_partials[i])
            end
            result::DiffNumber{CHUNK,S} = f(workvec)
            @simd for i in 1:$(remainder)
                j = $(fill_length) + i
                @inbounds out[j] = partials(result, i)
                @inbounds workvec[j] = DiffNumber{CHUNK,T}(x[j], pzeros)
            end
        end
    end
    return calc_gradient_expr(body)
end

if IS_MULTITHREADED_JULIA
    @generated function multi_calc_gradient!{S,T,CHUNK,LEN}(f, out::AbstractVector{S}, x::AbstractVector{T},
                                                            ::Type{Val{CHUNK}}, ::Type{Val{LEN}})
        if CHUNK == LEN
            body = VEC_MODE_EXPR
        else
            remainder = LEN % CHUNK == 0 ? CHUNK : LEN % CHUNK
            fill_length = LEN - remainder
            reseed_partials = remainder == CHUNK ? :() : :(seed_partials = cachefetch!(tid, Partials{CHUNK,T}, Val{$(remainder)}))
            body = quote
                workvecs::NTuple{NTHREADS, Vector{DiffNumber{CHUNK,T}}} = cachefetch!(DiffNumber{CHUNK,T}, Val{LEN})
                pzeros = zero(Partials{CHUNK,T})

                Base.Threads.@threads for t in 1:NTHREADS
                    # must be local, see https://github.com/JuliaLang/julia/issues/14948
                    local workvec = workvecs[t]
                    @simd for i in 1:LEN
                        @inbounds workvec[i] = DiffNumber{CHUNK,T}(x[i], pzeros)
                    end
                end

                Base.Threads.@threads for c in 1:$(CHUNK):$(fill_length)
                    local workvec = workvecs[Base.Threads.threadid()]
                    @simd for i in 1:CHUNK
                        j = i + c - 1
                        @inbounds workvec[j] = DiffNumber{CHUNK,T}(x[j], seed_partials[i])
                    end
                    local result::DiffNumber{CHUNK,S} = f(workvec)
                    @simd for i in 1:CHUNK
                        j = i + c - 1
                        @inbounds out[j] = partials(result, i)
                        @inbounds workvec[j] = DiffNumber{CHUNK,T}(x[j], pzeros)
                    end
                end

                # Performing the final chunk manually seems to triggers some additional
                # optimization heuristics, which results in more efficient memory allocation
                $(reseed_partials)
                workvec = workvecs[tid]
                @simd for i in 1:$(remainder)
                    j = $(fill_length) + i
                    @inbounds workvec[j] = DiffNumber{CHUNK,T}(x[j], seed_partials[i])
                end
                result::DiffNumber{CHUNK,S} = f(workvec)
                @simd for i in 1:$(remainder)
                    j = $(fill_length) + i
                    @inbounds out[j] = partials(result, i)
                    @inbounds workvec[j] = DiffNumber{CHUNK,T}(x[j], pzeros)
                end
            end
        end
        return calc_gradient_expr(body)
    end
end

const VEC_MODE_EXPR = quote
    workvec::Vector{DiffNumber{CHUNK,T}} = cachefetch!(tid, DiffNumber{CHUNK,T}, Val{LEN})
    @simd for i in 1:LEN
        @inbounds workvec[i] = DiffNumber{CHUNK,T}(x[i], seed_partials[i])
    end
    result::DiffNumber{CHUNK,S} = f(workvec)
    @simd for i in 1:LEN
        @inbounds out[i] = partials(result, i)
    end
end

function calc_gradient_expr(body)
    return quote
        @assert LEN == length(x) == length(out)
        tid = $(IS_MULTITHREADED_JULIA ? Base.Threads.threadid() : 1)
        seed_partials::Vector{Partials{CHUNK,T}} = cachefetch!(tid, Partials{CHUNK,T})
        $(body)
        return (value(result)::S, out)
    end
end

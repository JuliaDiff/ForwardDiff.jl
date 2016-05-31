########################
# @gradient!/@gradient #
########################

macro gradient!(args...)
    args, kwargs = separate_kwargs(args)
    arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS)
    return esc(:(ForwardDiff.gradient_entry_point!($(arranged_kwargs...), $(last(args)), $(args[1:end-1]...))))
end

macro gradient(args...)
    args, kwargs = separate_kwargs(args)
    arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS)
    return esc(:(ForwardDiff.gradient_entry_point($(arranged_kwargs...), $(last(args)), $(args[1:end-1]...))))
end

##################
# GradientResult #
##################

abstract GradientResult <: ForwardDiffResult

# vector mode #
#-------------#

immutable GradientVectorResult{D} <: GradientResult
    len::Int
    dual::D
end

function gradient(result::GradientVectorResult)
    out = Vector{numtype(eltype(result.dual))}(result.len)
    return gradient!(out, result)
end

function gradient!(out, result::GradientVectorResult)
    @assert length(out) == result.len
    @simd for i in 1:result.len
        @inbounds out[i] = partials(result.dual, i)
    end
    return out
end

value(result::GradientVectorResult) = value(result.dual)

# chunk mode #
#------------#

immutable GradientChunkResult{V,G} <: GradientResult
    value::V
    grad::G
end

gradient(result::GradientChunkResult) = result.grad

gradient!(out, result::GradientChunkResult) = copy!(out, result.grad)

value(result::GradientChunkResult) = result.value

###############
# API methods #
###############

function gradient_entry_point!(chunk, len, allresults, multithread, x, args...)
    return dispatch_gradient!(pickchunk(chunk, len, x), allresults, multithread, x, args...)
end

function gradient_entry_point(chunk, len, allresults, multithread, x, args...)
    return dispatch_gradient(pickchunk(chunk, len, x), allresults, multithread, x, args...)
end

# vector mode #
#-------------#

@inline function dispatch_gradient!{N}(::Tuple{Val{N}, Val{N}}, allresults, multithread, x, out, f)
    result = vector_mode_gradient!(Val{N}(), f, x)
    gradient!(out, result)
    return pickresult(allresults, result, out)
end

@inline function dispatch_gradient{N}(::Tuple{Val{N}, Val{N}}, allresults, multithread, x, f)
    result = vector_mode_gradient!(Val{N}(), f, x)
    out = gradient(result)
    return pickresult(allresults, result, out)
end

# chunk mode #
#------------#

@inline function dispatch_gradient!{C,L}(::Tuple{Val{C}, Val{L}}, allresults, multithread, x, out, f)
    result = chunk_mode_gradient!(multithread, Val{C}(), Val{L}(), out, f, x)
    return pickresult(allresults, result, out)
end

@inline function dispatch_gradient{C,L}(::Tuple{Val{C}, Val{L}}, allresults, multithread, x, f)
    result = chunk_mode_gradient!(multithread, Val{C}(), Val{L}(), DummyVar(), f, x)
    return pickresult(allresults, result, gradient(result))
end

#######################
# workhorse functions #
#######################

# vector mode #
#-------------#

function vector_mode_gradient!{L}(len::Val{L}, f, x)
    @assert length(x) == L
    xdual = fetchxdual(x, len, len)
    seeds = fetchseeds(eltype(xdual))
    seed!(xdual, x, 1, seeds)
    return GradientVectorResult(L, f(xdual))
end

# chunk mode #
#------------#

@generated function chunk_mode_gradient!{C,L}(multithread::Val{false}, chunk::Val{C}, len::Val{L}, outvar, f, x)
    if outvar <: DummyVar
        outdef = :(out = Vector{numtype(typeof(dual))}(L))
    else
        outdef = quote
            @assert length(outvar) == L
            out = outvar
        end
    end
    lastchunksize = L % C == 0 ? C : L % C
    fullchunks = div(L - lastchunksize, C)
    lastoffset = L - lastchunksize + 1
    reseedexpr = lastchunksize == C ? :() : :(seeds = fetchseeds(eltype(xdual), $(Val{lastchunksize}())))
    return quote
        @assert length(x) == L
        xdual = fetchxdual(x, len, chunk)
        seeds = fetchseeds(eltype(xdual))
        zeroseed = zero(Partials{C,eltype(x)})
        seedall!(xdual, x, len, zeroseed)

        # do first chunk manually for dynamic output definition
        seed!(xdual, x, 1, seeds)
        dual = f(xdual)
        seed!(xdual, x, 1, zeroseed)
        $(outdef)
        loadgradchunk!(out, dual, 1, chunk)

        # do middle chunks
        for c in 2:$(fullchunks)
            offset = ((c - 1) * C + 1)
            seed!(xdual, x, offset, seeds)
            dual = f(xdual)
            seed!(xdual, x, offset, zeroseed)
            loadgradchunk!(out, dual, offset, chunk)
        end

        # do final chunk manually
        $(reseedexpr)
        seed!(xdual, x, $(lastoffset), seeds)
        dual = f(xdual)
        loadgradchunk!(out, dual, $(lastoffset), $(Val{lastchunksize}()))

        return GradientChunkResult(value(dual), out)
    end
end

function loadgradchunk!{C}(out, dual, offset, chunk::Val{C})
    k = offset - 1
    for i in 1:C
        out[i + k] = partials(dual, i)
    end
end

if IS_MULTITHREADED_JULIA
    @generated function chunk_mode_gradient!{C,L}(multithread::Val{true}, chunk::Val{C}, len::Val{L}, outvar, f, x)
        if outvar <: DummyVar
            outdef = :(out = Vector{numtype(eltype(dual))}(L))
        else
            outdef = quote
                @assert length(outvar) == L
                out = outvar
            end
        end
        lastchunksize = L % C == 0 ? C : L % C
        fullchunks = div(L - lastchunksize, C)
        lastoffset = L - lastchunksize + 1
        reseedexpr = lastchunksize == C ? :() : :(seeds = fetchseeds(eltype(xdual), $(Val{lastchunksize}())))
        return quote
            @assert length(x) == L
            tid = compat_threadid()
            xduals = threaded_fetchxdual(x, len, chunk)
            xdual = xduals[tid] # this thread's xdual
            seeds = fetchseeds(eltype(xdual))
            zeroseed = zero(Partials{C,eltype(x)})

            Base.Threads.@threads for t in 1:NTHREADS
                seedall!(xduals[t], x, len, zeroseed)
            end

            # do first chunk manually
            seed!(xdual, x, 1, seeds)
            dual = f(xdual)
            seed!(xdual, x, 1, zeroseed)
            $(outdef)
            loadgradchunk!(out, dual, 1, chunk)

            # do middle chunks
            Base.Threads.@threads for c in 2:$(fullchunks)
                # see https://github.com/JuliaLang/julia/issues/14948
                local chunk_xdual = xduals[compat_threadid()]
                local chunk_offset = ((c - 1) * C + 1)
                seed!(chunk_xdual, x, chunk_offset, seeds)
                local chunk_dual = f(chunk_xdual)
                seed!(chunk_xdual, x, chunk_offset, zeroseed)
                loadgradchunk!(out, chunk_dual, chunk_offset, chunk)
            end

            # do final chunk manually
            $(reseedexpr)
            seed!(xdual, x, $(lastoffset), seeds)
            dual = f(xdual)
            loadgradchunk!(out, dual, $(lastoffset), $(Val{lastchunksize}()))

            return GradientChunkResult(value(dual), out)
        end
    end
else
    function chunk_mode_gradient!{C,L}(multithread::Val{true}, chunk::Val{C}, len::Val{L}, outvar, f, x)
        error("Multithreading is not enabled for this Julia installation.")
    end
end

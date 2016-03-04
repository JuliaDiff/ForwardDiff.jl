########################
# @jacobian!/@jacobian #
########################

macro jacobian!(args...)
    args, kwargs = separate_kwargs(args)
    arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS)
    return esc(:(ForwardDiff.jacobian_entry_point!($(arranged_kwargs...), $(last(args)), $(args[1:end-1]...))))
end

macro jacobian(args...)
    args, kwargs = separate_kwargs(args)
    arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS)
    return esc(:(ForwardDiff.jacobian_entry_point($(arranged_kwargs...), $(last(args)), $(args[1:end-1]...))))
end

##################
# JacobianResult #
##################

abstract JacobianResult <: ForwardDiffResult

immutable JacobianVectorResult{Y} <: JacobianResult
    len::Int
    ydiff::Y
end

immutable JacobianChunkResult{Y,J} <: JacobianResult
    len::Int
    ydiff::Y
    jac::J
end

function jacobian(result::JacobianVectorResult)
    out = similar(result.ydiff, numtype(eltype(result.ydiff)), length(result.ydiff), result.len)
    return jacobian!(out, result)
end

function jacobian!(out, result::JacobianVectorResult)
    ylength = length(result.ydiff)
    @assert size(out) == (ylength, result.len)
    for j in 1:result.len
        @simd for i in 1:ylength
            @inbounds out[i, j] = partials(result.ydiff[i], j)
        end
    end
    return out
end

jacobian(result::JacobianChunkResult) = copy(result.jac)

jacobian!(out, result::JacobianChunkResult) = copy!(out, result.jac)

function value(result::JacobianResult)
    out = similar(result.ydiff, numtype(eltype(result.ydiff)))
    return jacobian!(out, result)
end

function value!(out, result::JacobianResult)
    @assert length(out) == length(result.ydiff)
    @simd for i in 1:length(result.ydiff)
        @inbounds out[i] = value(result.ydiff[i])
    end
    return out
end

###############
# API methods #
###############

function jacobian_entry_point!(chunk, len, allresults, multithread, x, args...)
    return dispatch_jacobian!(pickchunk(chunk, len, x), allresults, multithread, x, args...)
end

function jacobian_entry_point(chunk, len, allresults, multithread, x, args...)
    return dispatch_jacobian(pickchunk(chunk, len, x), allresults, multithread, x, args...)
end

# vector mode #
#-------------#

@inline function dispatch_jacobian!{N}(::Tuple{Val{N}, Val{N}}, allresults, multithread, x, out, f!, y)
    result = vector_mode_jacobian!(Val{N}(), f!, y, x)
    value!(y, result)
    jacobian!(out, result)
    return pickresult(allresults, result, out)
end

@inline function dispatch_jacobian!{N}(::Tuple{Val{N}, Val{N}}, allresults, multithread, x, out, f)
    result = vector_mode_jacobian!(Val{N}(), f, x)
    jacobian!(out, result)
    return pickresult(allresults, result, out)
end

@inline function dispatch_jacobian{N}(::Tuple{Val{N}, Val{N}}, allresults, multithread, x, f!, y)
    result = vector_mode_jacobian!(Val{N}(), f!, y, x)
    value!(y, result)
    out = jacobian(result)
    return pickresult(allresults, result, out)
end

@inline function dispatch_jacobian{N}(::Tuple{Val{N}, Val{N}}, allresults, multithread, x, f)
    result = vector_mode_jacobian!(Val{N}(), f, x)
    out = jacobian(result)
    return pickresult(allresults, result, out)
end

# chunk mode #
#------------#

@inline function dispatch_jacobian!{C,L}(::Tuple{Val{C}, Val{L}}, allresults, multithread, x, out, f!, y)
    result = chunk_mode_jacobian!(multithread, Val{C}(), Val{L}(), out, f!, x, y)
    value!(y, result)
    return pickresult(allresults, result, out)
end

@inline function dispatch_jacobian!{C,L}(::Tuple{Val{C}, Val{L}}, allresults, multithread, x, out, f)
    result = chunk_mode_jacobian!(multithread, Val{C}(), Val{L}(), out, f, x, DummyVar())
    return pickresult(allresults, result, out)
end

@inline function dispatch_jacobian{C,L}(::Tuple{Val{C}, Val{L}}, allresults, multithread, x, f!, y)
    result = chunk_mode_jacobian!(multithread, Val{C}(), Val{L}(), DummyVar(), f!, x, y)
    value!(y, result)
    return pickresult(allresults, result, result.jac)
end

@inline function dispatch_jacobian{C,L}(::Tuple{Val{C}, Val{L}}, allresults, multithread, x, f)
    result = chunk_mode_jacobian!(multithread, Val{C}(), Val{L}(), DummyVar(), f, x, DummyVar())
    return pickresult(allresults, result, result.jac)
end

#######################
# workhorse functions #
#######################

# vector mode #
#-------------#

function vector_mode_jacobian!{L}(len::Val{L}, f, x)
    @assert length(x) == L
    xdiff = fetchxdiff(x, len, len)
    seeds = fetchseeds(xdiff)
    seed!(xdiff, x, seeds, 1)
    return JacobianVectorResult(L, f(xdiff))
end

function vector_mode_jacobian!{L}(len::Val{L}, f!, y, x)
    @assert length(x) == L
    xdiff = fetchxdiff(x, len, len)
    ydiff = Vector{DiffNumber{L,eltype(y)}}(length(y))
    seeds = fetchseeds(xdiff)
    seed!(xdiff, x, seeds, 1)
    f!(ydiff, xdiff)
    return JacobianVectorResult(L, ydiff)
end

# chunk mode #
#------------#

@generated function chunk_mode_jacobian!{C,L}(multithread::Val{false}, chunk::Val{C}, len::Val{L}, outvar, f, x, yvar)
    if outvar <: DummyVar
        outdef = :(out = Matrix{numtype(eltype(ydiff))}(length(ydiff), L))
    else
        outdef = quote
            @assert size(outvar) == (length(ydiff), L)
            out = outvar
        end
    end
    if yvar <: DummyVar
        ydiffdef = :()
        ydiffcompute = :(ydiff = f(xdiff))
    else
        ydiffdef = :(ydiff = Vector{DiffNumber{L,eltype(yvar)}}(length(yvar)))
        ydiffcompute = :(f(ydiff, xdiff))
    end
    R = L % C == 0 ? C : L % C
    fullchunks = div(L - R, C)
    lastoffset = L - R + 1
    reseedexpr = R == C ? :() : :(seeds = fetchseeds(xdiff, $(Val{R}())))
    return quote
        @assert length(x) == L
        xdiff = fetchxdiff(x, chunk, len)
        $(ydiffdef)
        seeds = fetchseeds(xdiff)
        zeroseed = zero(Partials{C,eltype(x)})
        seedall!(xdiff, x, len, zeroseed)

        # do first chunk manually
        seed!(xdiff, x, seeds, 1)
        $(ydiffcompute)
        seed!(xdiff, x, zeroseed, 1)
        $(outdef)
        jacloadchunk!(out, ydiff, chunk, 1)

        # do middle chunks
        for c in 2:$(fullchunks)
            offset = ((c - 1) * C + 1)
            seed!(xdiff, x, seeds, offset)
            $(ydiffcompute)
            seed!(xdiff, x, zeroseed, offset)
            jacloadchunk!(out, ydiff, chunk, offset)
        end

        # do final chunk manually
        $(reseedexpr)
        seed!(xdiff, x, seeds, $(lastoffset))
        $(ydiffcompute)
        jacloadchunk!(out, ydiff, $(Val{R}()), $(lastoffset))

        return JacobianChunkResult(L, ydiff, out)
    end
end

function jacloadchunk!{C}(out, ydiff, chunk::Val{C}, offset)
    k = offset - 1
    for i in 1:C
        col = i + k
        @simd for row in 1:length(ydiff)
            @inbounds out[row, col] = partials(ydiff[row], i)
        end
    end
end

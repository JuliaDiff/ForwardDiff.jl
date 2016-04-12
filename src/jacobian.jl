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
    ydual::Y
end

immutable JacobianChunkResult{Y,J} <: JacobianResult
    len::Int
    ydual::Y
    jac::J
end

function jacobian(result::JacobianVectorResult)
    out = similar(result.ydual, numtype(eltype(result.ydual)), length(result.ydual), result.len)
    return jacobian!(out, result)
end

function jacobian!(out, result::JacobianVectorResult)
    ylength = length(result.ydual)
    @assert size(out) == (ylength, result.len)
    for j in 1:result.len
        @simd for i in 1:ylength
            @inbounds out[i, j] = partials(result.ydual[i], j)
        end
    end
    return out
end

jacobian(result::JacobianChunkResult) = copy(result.jac)

jacobian!(out, result::JacobianChunkResult) = copy!(out, result.jac)

function value(result::JacobianResult)
    out = similar(result.ydual, numtype(eltype(result.ydual)))
    return jacobian!(out, result)
end

function value!(out, result::JacobianResult)
    @assert length(out) == length(result.ydual)
    @simd for i in 1:length(result.ydual)
        @inbounds out[i] = value(result.ydual[i])
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
    xdual = fetchxdual(x, len, len)
    seeds = fetchseeds(eltype(xdual))
    seed!(xdual, x, 1, seeds)
    return JacobianVectorResult(L, f(xdual))
end

function vector_mode_jacobian!{L}(len::Val{L}, f!, y, x)
    @assert length(x) == L
    xdual = fetchxdual(x, len, len)
    ydual = Vector{Dual{L,eltype(y)}}(length(y))
    seeds = fetchseeds(eltype(xdual))
    seed!(xdual, x, 1, seeds)
    f!(ydual, xdual)
    return JacobianVectorResult(L, ydual)
end

# chunk mode #
#------------#

@generated function chunk_mode_jacobian!{C,L}(multithread::Val{false}, chunk::Val{C}, len::Val{L}, outvar, f, x, yvar)
    if outvar <: DummyVar
        outdef = :(out = Matrix{numtype(eltype(ydual))}(length(ydual), L))
    else
        outdef = quote
            @assert size(outvar) == (length(ydual), L)
            out = outvar
        end
    end
    if yvar <: DummyVar
        ydualdef = :()
        ydualcompute = :(ydual = f(xdual))
    else
        ydualdef = :(ydual = Vector{Dual{L,eltype(yvar)}}(length(yvar)))
        ydualcompute = :(f(ydual, xdual))
    end
    R = L % C == 0 ? C : L % C
    fullchunks = div(L - R, C)
    lastoffset = L - R + 1
    reseedexpr = R == C ? :() : :(seeds = fetchseeds(eltype(xdual), $(Val{R}())))
    return quote
        @assert length(x) == L
        xdual = fetchxdual(x, len, chunk)
        $(ydualdef)
        seeds = fetchseeds(eltype(xdual))
        zeroseed = zero(Partials{C,eltype(x)})
        seedall!(xdual, x, len, zeroseed)

        # do first chunk manually
        seed!(xdual, x, 1, seeds)
        $(ydualcompute)
        seed!(xdual, x, 1, zeroseed)
        $(outdef)
        jacloadchunk!(out, ydual, 1, chunk)

        # do middle chunks
        for c in 2:$(fullchunks)
            offset = ((c - 1) * C + 1)
            seed!(xdual, x, offset, seeds)
            $(ydualcompute)
            seed!(xdual, x, offset, zeroseed)
            jacloadchunk!(out, ydual, chunk, offset)
        end

        # do final chunk manually
        $(reseedexpr)
        seed!(xdual, x, $(lastoffset), seeds)
        $(ydualcompute)
        jacloadchunk!(out, ydual, $(lastoffset), $(Val{R}()))

        return JacobianChunkResult(L, ydual, out)
    end
end

function jacloadchunk!{C}(out, ydual, offset, chunk::Val{C})
    k = offset - 1
    for i in 1:C
        col = i + k
        @simd for row in 1:length(ydual)
            @inbounds out[row, col] = partials(ydual[row], i)
        end
    end
end

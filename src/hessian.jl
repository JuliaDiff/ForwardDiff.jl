######################
# @hessian!/@hessian #
######################

macro hessian!(args...)
    args, kwargs = separate_kwargs(args)
    arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS)
    return esc(:(ForwardDiff.hessian_entry_point!($(arranged_kwargs...), $(last(args)), $(args[1:end-1]...))))
end

macro hessian(args...)
    args, kwargs = separate_kwargs(args)
    arranged_kwargs = arrange_kwargs(kwargs, KWARG_DEFAULTS)
    return esc(:(ForwardDiff.hessian_entry_point($(arranged_kwargs...), $(last(args)), $(args[1:end-1]...))))
end

##################
# HessianResult #
##################

abstract HessianResult <: ForwardDiffResult

immutable HessianVectorResult{N} <: HessianResult
    len::Int
    dual::N
end

function hessian(result::HessianVectorResult)
    out = Matrix{numtype(numtype(result.dual))}(result.len, result.len)
    return hessian!(out, result)
end

function hessian!(out, result::HessianVectorResult)
    @assert size(out) == (result.len, result.len)
    for j in 1:result.len
        @simd for i in 1:result.len
            @inbounds out[i, j] = partials(partials(result.dual, i), j)
        end
    end
    return out
end

function gradient(result::HessianVectorResult)
    out = Vector{numtype(numtype(result.dual))}(result.len)
    return gradient!(out, result)
end

function gradient!(out, result::HessianVectorResult)
    @assert length(out) == result.len
    dval = value(result.dual)
    @simd for i in 1:result.len
        @inbounds out[i] = partials(dval, i)
    end
    return out
end

value(result::HessianResult) = value(value(result.dual))

###############
# API methods #
###############

function hessian_entry_point!(chunk, len, allresults, multithread, x, args...)
    return dispatch_hessian!(pickchunk(chunk, len, x), allresults, multithread, x, args...)
end

function hessian_entry_point(chunk, len, allresults, multithread, x, args...)
    return dispatch_hessian(pickchunk(chunk, len, x), allresults, multithread, x, args...)
end

# vector mode #
#-------------#

@inline function dispatch_hessian!{N}(::Tuple{Val{N}, Val{N}}, allresults, multithread, x, out, f)
    result = vector_mode_hessian!(Val{N}(), f, x)
    hessian!(out, result)
    return pickresult(allresults, result, out)
end

@inline function dispatch_hessian{N}(::Tuple{Val{N}, Val{N}}, allresults, multithread, x, f)
    result = vector_mode_hessian!(Val{N}(), f, x)
    out = hessian(result)
    return pickresult(allresults, result, out)
end

# chunk mode #
#------------#

#TODO

#######################
# workhorse functions #
#######################

# vector mode #
#-------------#

function vector_mode_hessian!{L}(len::Val{L}, f, x)
    @assert length(x) == L
    xdual = fetchxdual(x, len, len, len)
    nseeds = fetchseeds(eltype(xdual))
    mseeds = fetchseeds(numtype(eltype(xdual)))
    seed!(xdual, x, 1, nseeds, mseeds)
    return HessianVectorResult(L, f(xdual))
end

# chunk mode #
#------------#

#TODO

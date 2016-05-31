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

# vector mode #
#-------------#

immutable HessianVectorResult{D} <: HessianResult
    len::Int
    dual::D
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

value(result::HessianVectorResult) = value(value(result.dual))

# chunk mode #
#------------#

immutable HessianChunkResult{T,G,H} <: HessianResult
    value::T
    grad::G
    hess::H
end

hessian(result::HessianChunkResult) = result.hess

hessian!(out, result::HessianChunkResult) = copy!(out, result.hess)

gradient(result::HessianChunkResult) = result.grad

gradient!(out, result::HessianChunkResult) = copy!(out, result.grad)

value(result::HessianChunkResult) = result.value

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

@inline function dispatch_hessian!{C,L}(::Tuple{Val{C}, Val{L}}, allresults, multithread, x, out, f)
    result = chunk_mode_hessian!(multithread, Val{C}(), Val{L}(), out, f, x)
    return pickresult(allresults, result, out)
end

@inline function dispatch_hessian{C,L}(::Tuple{Val{C}, Val{L}}, allresults, multithread, x, f)
    result = chunk_mode_hessian!(multithread, Val{C}(), Val{L}(), DummyVar(), f, x)
    return pickresult(allresults, result, hessian(result))
end

#######################
# workhorse functions #
#######################

# vector mode #
#-------------#

function vector_mode_hessian!{L}(len::Val{L}, f, x)
    @assert length(x) == L
    xdual = fetchxdualhess(x, len, len)
    inseeds = fetchseeds(numtype(eltype(xdual)))
    outseeds = fetchseeds(eltype(xdual))
    seed!(xdual, x, 1, inseeds, outseeds)
    return HessianVectorResult(L, f(xdual))
end

# chunk mode #
#------------#

@generated function chunk_mode_hessian!{C,L}(multithread::Val{false}, chunk::Val{C}, len::Val{L}, outvar, f, x)
    # TODO: allow user to pass in gradient vector
    if outvar <: DummyVar
        outdef = quote
            outhess = zeros(numtype(numtype(dual)), L, L)
            outgrad = Vector{numtype(numtype(dual))}(L)
        end
    else
        outdef = quote
            @assert size(outvar) == (L, L)
            outhess = outvar
            outgrad = Vector{eltype(outhess)}(L)
        end
    end
    # constants
    diaglastchunksize = L % C == 0 ? C : L % C
    diagfullchunks = div(L - diaglastchunksize, C)
    diaglastoffset = L - diaglastchunksize + 1
    diagreseedexpr = diaglastchunksize == C ? :() : :(seeds = fetchseeds(eltype(xdual), $(Val{diaglastchunksize}())))
    offC = C + 1
    return quote
        @assert length(x) == L
        T = eltype(x)

        # diagonal chunks #
        #-----------------#
        xdual = fetchxdualhess(x, len, chunk)
        inseeds = fetchseeds(Dual{C,T})
        outseeds = fetchseeds(Dual{C,Dual{C,T}})
        inzero = zero(Partials{C,T})
        outzero = zero(Partials{C,Dual{C,T}})
        seedall!(xdual, x, len, inzero, outzero)

        # We first compute the derivatives in blocks along
        # the diagonal. The size of these blocks is
        # determined by the chunk size.
        #
        # For example, if `chunk = 2` and `length(x) = 7`,
        # the numbers inside the slots below indicate the
        # iteration (i.e. `i`th call to `f`) in which they
        # are filled:
        #
        # 7x7 Hessian with chunk=2:
        # -----------------------------
        # | 1 | 1 |   |   |   |   |   |
        # -----------------------------
        # | 1 | 1 |   |   |   |   |   |
        # -----------------------------
        # |   |   | 2 | 2 |   |   |   |
        # -----------------------------
        # |   |   | 2 | 2 |   |   |   |
        # -----------------------------
        # |   |   |   |   | 3 | 3 |   |
        # -----------------------------
        # |   |   |   |   | 3 | 3 |   |
        # -----------------------------
        # |   |   |   |   |   |   | 4 |
        # -----------------------------

        # do first chunk manually for dynamic output definition
        seed!(xdual, x, 1, inseeds, outseeds)
        dual = f(xdual)
        $(outdef)
        loadhessdiagchunk!(outhess, dual, 1, chunk)
        loadgradchunk!(outgrad, value(dual), 1, chunk)
        seed!(xdual, x, 1, inzero, outzero)

        # do middle chunks
        for c in 2:$(diagfullchunks)
            offset = (c - 1) * C + 1
            seed!(xdual, x, offset, inseeds, outseeds)
            dual = f(xdual)
            seed!(xdual, x, offset, inzero, outzero)
            loadhessdiagchunk!(outhess, dual, offset, chunk)
            loadgradchunk!(outgrad, value(dual), offset, chunk)
        end

        # do final chunk manually
        $(diagreseedexpr)
        seed!(xdual, x, $(diaglastoffset), inseeds, outseeds)
        dual = f(xdual)
        loadhessdiagchunk!(outhess, dual, $(diaglastoffset), $(Val{diaglastchunksize}()))
        loadgradchunk!(outgrad, value(dual), $(diaglastoffset), $(Val{diaglastchunksize}()))

        # off-diagonal chunks #
        #---------------------#
        xdual = fetchxdualhess(x, len, $(Val{offC}()))
        inseeds = fetchseeds(Dual{$offC,T})
        outseeds = fetchseeds(Dual{$offC,Dual{$offC,T}})
        lastinseed = last(inseeds)
        lastoutseed = last(outseeds)
        inzero = zero(Partials{$offC,T})
        outzero = zero(Partials{$offC,Dual{$offC,T}})
        seedall!(xdual, x, len, inzero, outzero)

        # Now, we fill in the off-diagonal chunks. Like
        # the previous diagram, the numbers inside the
        # slots indicate the iteration (i.e. `i`th call
        # to `f`) in which they are filled:
        #
        # 7x7 Hessian with chunk=2:
        # -----------------------------
        # |   |   | 1 | 2 | 3 | 4 | 5 |
        # -----------------------------
        # |   |   | 1 | 2 | 3 | 4 | 5 |
        # -----------------------------
        # | 1 | 1 |   |   | 6 | 7 | 8 |
        # -----------------------------
        # | 2 | 2 |   |   | 6 | 7 | 8 |
        # -----------------------------
        # | 3 | 3 | 6 | 6 |   |   | 9 |
        # -----------------------------
        # | 4 | 4 | 7 | 7 |   |   | 9 |
        # -----------------------------
        # | 5 | 5 | 8 | 8 | 9 | 9 |   |
        # -----------------------------

        for c in 1:$(diagfullchunks + 1)
            offset = (c - 1) * C + 1
            offseed!(xdual, x, offset, inseeds, outseeds)
            for j in (offset + C):(L)
                offseedj!(xdual, x, j, lastinseed, lastoutseed)
                dual = f(xdual)
                loadhessoffchunk!(outhess, dual, offset, j, chunk)
                offseedj!(xdual, x, j, inzero, outzero)
            end
            offseed!(xdual, x, offset, inzero, outzero)
        end

        return HessianChunkResult(value(dual), outgrad, outhess)
    end
end

function loadhessdiagchunk!{C}(out, dual, offset, chunk::Val{C})
    k = offset - 1
    for j in 1:C, i in 1:C
        out[i+k, j+k] = partials(dual, i, j)
    end
end

function loadhessoffchunk!{C}(out, dual, offset, j, chunk::Val{C})
    k = offset - 1
    offC = C + 1
    for i in 1:C
        out[i+k, j] = partials(dual, i, offC)
        out[j, i+k] = partials(dual, offC, i)
    end
end

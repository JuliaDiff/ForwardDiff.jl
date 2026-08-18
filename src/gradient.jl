###############
# API methods #
###############

"""
    ForwardDiff.gradient(f, x::AbstractArray, cfg::GradientConfig = GradientConfig(f, x), check=Val{true}())

Return `∇f` evaluated at `x`, assuming `f` is called as `f(x)`.
The array `∇f` has the same shape as `x`, and its elements are
`∇f[j, k, ...] = ∂f/∂x[j, k, ...]`.

This method assumes that `isa(f(x), Real)`.

Set `check` to `Val{false}()` to disable tag checking. This can lead to perturbation confusion, so should be used with care.
"""
function gradient(f::F, x::AbstractArray, cfg::GradientConfig{T} = GradientConfig(f, x), ::Val{CHK}=Val{true}()) where {F, T, CHK}
    require_one_based_indexing(x)
    CHK && checktag(T, f, x)
    if chunksize(cfg) == structural_length(x)
        return vector_mode_gradient(f, x, cfg)
    else
        return chunk_mode_gradient(f, x, cfg)
    end
end

"""
    ForwardDiff.gradient!(result::Union{AbstractArray,DiffResult}, f, x::AbstractArray, cfg::GradientConfig = GradientConfig(f, x), check=Val{true}())

Compute `∇f` evaluated at `x` and store the result(s) in `result`, assuming `f` is called as
`f(x)`.

This method assumes that `isa(f(x), Real)`.

"""
function gradient!(result::Union{AbstractArray,DiffResult}, f::F, x::AbstractArray, cfg::GradientConfig{T} = GradientConfig(f, x), ::Val{CHK}=Val{true}()) where {T, CHK, F}
    result isa DiffResult ? require_one_based_indexing(x) : require_one_based_indexing(result, x)
    CHK && checktag(T, f, x)
    if chunksize(cfg) == structural_length(x)
        vector_mode_gradient!(result, f, x, cfg)
    else
        chunk_mode_gradient!(result, f, x, cfg)
    end
    return result
end

gradient(f, x::Real) = throw(DimensionMismatch("gradient(f, x) expects that x is an array. Perhaps you meant derivative(f, x)?"))

#####################
# result extraction #
#####################

# Derivatives are only computed with respect to the structurally non-zero entries of `x`, since only
# those are seeded. The positions to write to therefore have to be taken from `x`, not from `result`:
# the two may have different structure, e.g. `DiffResults.HessianResult` allocates a dense gradient
# buffer even for a structured `x`. The remaining entries of `result` are zeroed, their derivative
# being zero, unless `x` has as many seeded entries as `result` has entries. See #838.

function extract_gradient!(::Type{T}, result::DiffResult, y::Real, x) where {T}
    result = DiffResults.value!(result, y)
    grad = DiffResults.gradient(result)
    fill!(grad, zero(y))
    return result
end

function extract_gradient!(::Type{T}, result::MutableDiffResult, dual::Dual, x) where {T}
    result = DiffResults.value!(result, value(T, dual))
    extract_gradient!(T, DiffResults.gradient(result), dual, x)
    return result
end

# Immutable results cannot be written to entry by entry. Copying the partials wholesale is correct
# as long as every entry of `x` is seeded, which holds for the `StaticArray` gradient buffers that
# are the only source of such results; anything else throws on the length mismatch.
function extract_gradient!(::Type{T}, result::ImmutableDiffResult, dual::Dual, x) where {T}
    result = DiffResults.value!(result, value(T, dual))
    result = DiffResults.gradient!(result, partials(T, dual))
    return result
end

# Zeroes `result` unless extraction is going to write every entry of it; the written ones are
# overwritten immediately after. In chunk mode the sweep calls this once up front, since the entries
# that no chunk writes belong to none of them in particular. Comparing counts rather than matching
# entries up is exact for every pair that can hold the gradient: a differently structured `result`
# with the same count, an `UpperTriangular` one for a `LowerTriangular` `x`, cannot. `dual` is passed
# for its value type, which unlike `eltype(result)` is a number type even for an `Any` result.
function zero_unseeded!(::Type{T}, result::AbstractArray, dual, x) where {T}
    structural_length(x) == structural_length(result) || fill!(result, zero(valtype(T, dual)))
    return result
end
# Dispatched on `DiffResult`, not on `MutableDiffResult`: a `StaticArray` gradient buffer makes the
# result immutable even when the buffer itself can be written to, as an `MVector` can.
function zero_unseeded!(::Type{T}, result::DiffResult, dual, x) where {T}
    zero_unseeded!(T, DiffResults.gradient(result), dual, x)
    return result
end

extract_gradient!(::Type{T}, result::AbstractArray, y::Real, x) where {T} = fill!(result, zero(y))
function extract_gradient!(::Type{T}, result::AbstractArray, dual::Dual, x) where {T}
    zero_unseeded!(T, result, dual, x)
    idxs = structural_eachindex(x, result)
    for (i, idx) in zip(1:npartials(dual), idxs)
        result[idx] = partials(T, dual, i)
    end
    return result
end

function extract_gradient_chunk!(::Type{T}, result, dual, x, index, chunksize) where {T}
    offset = index - 1
    idxs = Iterators.drop(structural_eachindex(x, result), offset)
    for (i, idx) in zip(1:chunksize, idxs)
        result[idx] = partials(T, dual, i)
    end
    return result
end

function extract_gradient_chunk!(::Type{T}, result::DiffResult, dual, x, index, chunksize) where {T}
    extract_gradient_chunk!(T, DiffResults.gradient(result), dual, x, index, chunksize)
    return result
end

const GRAD_ERROR = DimensionMismatch("gradient(f, x) expects that f(x) is a real number. Perhaps you meant jacobian(f, x)?")

###############
# vector mode #
###############

function vector_mode_gradient(f::F, x, cfg::GradientConfig{T}) where {T, F}
    ydual = vector_mode_dual_eval!(f, cfg, x)
    ydual isa Real || throw(GRAD_ERROR)
    result = similar(x, valtype(T, ydual))
    return extract_gradient!(T, result, ydual, x)
end

function vector_mode_gradient!(result, f::F, x, cfg::GradientConfig{T}) where {T, F}
    ydual = vector_mode_dual_eval!(f, cfg, x)
    result = extract_gradient!(T, result, ydual, x)
    return result
end

##############
# chunk mode #
##############

function chunk_mode_gradient_expr(result_definition::Expr)
    return quote
        if structural_length(x) < N
            throw(ArgumentError(lazy"chunk size cannot be greater than ForwardDiff.structural_length(x) ($(N) > $(structural_length(x)))"))
        end

        # precalculate loop bounds
        xlen = structural_length(x)
        remainder = xlen % N
        lastchunksize = ifelse(remainder == 0, N, remainder)
        lastchunkindex = xlen - lastchunksize + 1
        middlechunks = 2:div(xlen - lastchunksize, N)

        # seed work vectors
        xdual = cfg.duals
        seeds = cfg.seeds

        # do first chunk manually to calculate output type. Seeding the first chunk and zeroing the
        # remaining elements partitions `xdual`, so every element is initialized exactly once.
        seed!(xdual, x, 1, seeds)
        seed_zero_partials!(xdual, x, N + 1, xlen - N)
        ydual = f(xdual)
        ydual isa Real || throw(GRAD_ERROR)
        $(result_definition)
        zero_unseeded!(T, result, ydual, x)
        extract_gradient_chunk!(T, result, ydual, x, 1, N)
        seed_zero_partials!(xdual, x, 1)

        # do middle chunks
        for c in middlechunks
            i = ((c - 1) * N + 1)
            seed!(xdual, x, i, seeds)
            ydual = f(xdual)
            extract_gradient_chunk!(T, result, ydual, x, i, N)
            seed_zero_partials!(xdual, x, i)
        end

        # do final chunk
        seed!(xdual, x, lastchunkindex, seeds, lastchunksize)
        ydual = f(xdual)
        extract_gradient_chunk!(T, result, ydual, x, lastchunkindex, lastchunksize)

        # get the value, this is a no-op unless result is a DiffResult
        extract_value!(T, result, ydual)

        return result
    end
end

@eval function chunk_mode_gradient(f::F, x, cfg::GradientConfig{T,V,N}) where {F,T,V,N}
    $(chunk_mode_gradient_expr(:(result = similar(x, valtype(T, ydual)))))
end

@eval function chunk_mode_gradient!(result, f::F, x, cfg::GradientConfig{T,V,N}) where {F,T,V,N}
    $(chunk_mode_gradient_expr(:()))
end

###############
# API methods #
###############

"""
    ForwardDiff.jacobian(f, x::AbstractArray, cfg::JacobianConfig = JacobianConfig(f, x), check=Val{true}())

Return `J(f)` evaluated at `x`, assuming `f` is called as `f(x)`.
Multidimensional arrays are flattened in iteration order: the array
`J(f)` has shape `length(f(x)) × length(x)`, and its elements are
`J(f)[j,k] = ∂f(x)[j]/∂x[k]`.  When `x` is a vector, this means
that `jacobian(x->[f(x)], x)` is the transpose of `gradient(f, x)`.

This method assumes that `isa(f(x), AbstractArray)`.

Set `check` to `Val{false}()` to disable tag checking. This can lead to perturbation confusion, so should be used with care.
"""
function jacobian(f::F, x::AbstractArray, cfg::JacobianConfig{T} = JacobianConfig(f, x), ::Val{CHK}=Val{true}()) where {F,T,CHK}
    require_one_based_indexing(x)
    CHK && checktag(T, f, x)
    checkstructure(cfg, x)
    if chunksize(cfg) == structural_length(x)
        return vector_mode_jacobian(f, x, cfg)
    else
        return chunk_mode_jacobian(f, x, cfg)
    end
end

"""
    ForwardDiff.jacobian(f!, y::AbstractArray, x::AbstractArray, cfg::JacobianConfig = JacobianConfig(f!, y, x), check=Val{true}())

Return `J(f!)` evaluated at `x`,  assuming `f!` is called as `f!(y, x)` where the result is
stored in `y`.

Set `check` to `Val{false}()` to disable tag checking. This can lead to perturbation confusion, so should be used with care.
"""
function jacobian(f!::F, y::AbstractArray, x::AbstractArray, cfg::JacobianConfig{T} = JacobianConfig(f!, y, x), ::Val{CHK}=Val{true}()) where {F,T, CHK}
    require_one_based_indexing(y, x)
    CHK && checktag(T, f!, x)
    checkstructure(cfg, y, x)
    if chunksize(cfg) == structural_length(x)
        return vector_mode_jacobian(f!, y, x, cfg)
    else
        return chunk_mode_jacobian(f!, y, x, cfg)
    end
end


"""
    ForwardDiff.jacobian!(result::Union{AbstractArray,DiffResult}, f, x::AbstractArray, cfg::JacobianConfig = JacobianConfig(f, x), check=Val{true}())

Compute `J(f)` evaluated at `x` and store the result(s) in `result`, assuming `f` is called
as `f(x)`.

This method assumes that `isa(f(x), AbstractArray)`.

Set `check` to `Val{false}()` to disable tag checking. This can lead to perturbation confusion, so should be used with care.
"""
function jacobian!(result::Union{AbstractArray,DiffResult}, f::F, x::AbstractArray, cfg::JacobianConfig{T} = JacobianConfig(f, x), ::Val{CHK}=Val{true}()) where {F,T, CHK}
    result isa DiffResult ? require_one_based_indexing(x) : require_one_based_indexing(result, x)
    CHK && checktag(T, f, x)
    checkstructure(cfg, x)
    if chunksize(cfg) == structural_length(x)
        vector_mode_jacobian!(result, f, x, cfg)
    else
        chunk_mode_jacobian!(result, f, x, cfg)
    end
    return result
end

"""
    ForwardDiff.jacobian!(result::Union{AbstractArray,DiffResult}, f!, y::AbstractArray, x::AbstractArray, cfg::JacobianConfig = JacobianConfig(f!, y, x), check=Val{true}())

Compute `J(f!)` evaluated at `x` and store the result(s) in `result`, assuming `f!` is
called as `f!(y, x)` where the result is stored in `y`.

This method assumes that `isa(f(x), AbstractArray)`.

Set `check` to `Val{false}()` to disable tag checking. This can lead to perturbation confusion, so should be used with care.
"""
function jacobian!(result::Union{AbstractArray,DiffResult}, f!::F, y::AbstractArray, x::AbstractArray, cfg::JacobianConfig{T} = JacobianConfig(f!, y, x), ::Val{CHK}=Val{true}()) where {F,T,CHK}
    result isa DiffResult ? require_one_based_indexing(y, x) : require_one_based_indexing(result, y, x)
    CHK && checktag(T, f!, x)
    checkstructure(cfg, y, x)
    if chunksize(cfg) == structural_length(x)
        vector_mode_jacobian!(result, f!, y, x, cfg)
    else
        chunk_mode_jacobian!(result, f!, y, x, cfg)
    end
    return result
end

jacobian(f, x::Real) = throw(DimensionMismatch("jacobian(f, x) expects that x is an array. Perhaps you meant derivative(f, x)?"))

#####################
# result extraction #
#####################

# The Jacobian is indexed by the linear indices of `x`: column `j` holds the derivatives with respect
# to `x[j]`. Only the seeded entries of `x` have a derivative to extract, so the columns of the
# structurally zero ones are zeroed instead. See #839.

# The columns that no chunk writes belong to none of them in particular, so the sweep zeroes once up
# front rather than each chunk zeroing its own.
function zero_unseeded_columns!(::Type{T}, out::AbstractArray, ydual, x) where {T}
    structural_length(x) == length(x) || fill!(out, zero(valtype(T, eltype(ydual))))
    return nothing
end

# Vector mode is a single chunk that covers every seeded entry of `x`.
function extract_jacobian!(::Type{T}, result::AbstractArray, ydual::AbstractArray, x::AbstractArray,
                           indices) where {T}
    out_reshaped = reshape_jacobian(result, ydual, x)
    zero_unseeded_columns!(T, out_reshaped, ydual, x)
    extract_jacobian_chunk!(T, out_reshaped, ydual, indices, 1, structural_length(x))
    return result
end

function extract_jacobian!(::Type{T}, result::MutableDiffResult, ydual::AbstractArray,
                           x::AbstractArray, indices) where {T}
    extract_jacobian!(T, DiffResults.jacobian(result), ydual, x, indices)
    return result
end

# The positions are linear indices of `x`, which is what a Jacobian column is numbered by, so they
# select the columns directly -- for a dense `x` `structural_chunk` even hands back a `UnitRange`.
function extract_jacobian_chunk!(::Type{T}, result, ydual, indices, index, chunksize) where {T}
    ydual_reshaped = vec(ydual)
    irange = 1:chunksize
    cols = structural_chunk(indices, index, chunksize)
    # Use closure to avoid GPU broadcasting with Type
    partials_wrap(ydual, nrange) = partials(T, ydual, nrange)
    result[:, cols] .= partials_wrap.(ydual_reshaped, transpose(irange))
    return result
end

function reshape_jacobian(result::AbstractMatrix, ydual, x)
    size(result) == (length(ydual), length(x)) || throw(DimensionMismatch(
        lazy"cannot store the $(length(ydual))×$(length(x)) Jacobian in a result of size $(size(result))"))
    return result
end
reshape_jacobian(result::AbstractArray, ydual, x) = reshape(result, length(ydual), length(x))
reshape_jacobian(result::DiffResult, ydual, x) = reshape_jacobian(DiffResults.jacobian(result), ydual, x)

###############
# vector mode #
###############

function vector_mode_jacobian(f::F, x, cfg::JacobianConfig{T}) where {F,T}
    ydual = vector_mode_dual_eval!(f, cfg, x)
    ydual isa AbstractArray || throw(JACOBIAN_ERROR)
    result = similar(ydual, valtype(T, eltype(ydual)), length(ydual), length(x))
    extract_jacobian!(T, result, ydual, x, cfg.indices)
    extract_value!(T, result, ydual)
    return result
end

function vector_mode_jacobian(f!::F, y, x, cfg::JacobianConfig{T}) where {F,T}
    ydual = vector_mode_dual_eval!(f!, cfg, y, x)
    result = similar(y, length(y), length(x))
    extract_jacobian!(T, result, ydual, x, cfg.indices[2])
    map!(d -> value(T,d), y, ydual)
    return result
end

function vector_mode_jacobian!(result, f::F, x, cfg::JacobianConfig{T}) where {F,T}
    ydual = vector_mode_dual_eval!(f, cfg, x)
    ydual isa AbstractArray || throw(JACOBIAN_ERROR)
    extract_jacobian!(T, result, ydual, x, cfg.indices)
    extract_value!(T, result, ydual)
    return result
end

function vector_mode_jacobian!(result, f!::F, y, x, cfg::JacobianConfig{T}) where {F,T}
    ydual = vector_mode_dual_eval!(f!, cfg, y, x)
    map!(d -> value(T,d), y, ydual)
    extract_jacobian!(T, result, ydual, x, cfg.indices[2])
    extract_value!(T, result, y, ydual)
    return result
end

const JACOBIAN_ERROR = DimensionMismatch("jacobian(f, x) expects that f(x) is an array. Perhaps you meant gradient(f, x)?")

# chunk mode #
#------------#

function jacobian_chunk_mode_expr(work_array_definition::Expr, compute_ydual::Expr,
                                  result_definition::Expr, y_definition::Expr)
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

        # seed work arrays
        $(work_array_definition)
        seeds = cfg.seeds

        # do first chunk manually to calculate output type. Seeding the first chunk and zeroing the
        # remaining elements partitions `xdual`, so every element is initialized exactly once.
        seed!(xdual, x, indices, 1, seeds)
        seed_zero_partials!(xdual, x, indices, N + 1, xlen - N)
        $(compute_ydual)
        ydual isa AbstractArray || throw(JACOBIAN_ERROR)
        $(result_definition)
        out_reshaped = reshape_jacobian(result, ydual, x)
        zero_unseeded_columns!(T, out_reshaped, ydual, x)
        extract_jacobian_chunk!(T, out_reshaped, ydual, indices, 1, N)
        seed_zero_partials!(xdual, x, indices, 1)

        # do middle chunks
        for c in middlechunks
            i = ((c - 1) * N + 1)
            seed!(xdual, x, indices, i, seeds)
            $(compute_ydual)
            extract_jacobian_chunk!(T, out_reshaped, ydual, indices, i, N)
            seed_zero_partials!(xdual, x, indices, i)
        end

        # do final chunk
        seed!(xdual, x, indices, lastchunkindex, seeds, lastchunksize)
        $(compute_ydual)
        extract_jacobian_chunk!(T, out_reshaped, ydual, indices, lastchunkindex, lastchunksize)

        $(y_definition)

        return result
    end
end

@eval function chunk_mode_jacobian(f::F, x, cfg::JacobianConfig{T,V,N}) where {F,T,V,N}
    $(jacobian_chunk_mode_expr(:((xdual = cfg.duals; indices = cfg.indices)),
                               :(ydual = f(xdual)),
                               :(result = similar(ydual, valtype(T, eltype(ydual)), length(ydual), length(x))),
                               :()))
end

@eval function chunk_mode_jacobian(f!::F, y, x, cfg::JacobianConfig{T,V,N}) where {F,T,V,N}
    $(jacobian_chunk_mode_expr(:(((ydual, xdual) = cfg.duals; (yindices, indices) = cfg.indices)),
                               :(f!(seed_zero_partials!(ydual, y, yindices), xdual)),
                               :(result = similar(y, length(y), length(x))),
                               :(map!(d -> value(T,d), y, ydual))))
end

@eval function chunk_mode_jacobian!(result, f::F, x, cfg::JacobianConfig{T,V,N}) where {F,T,V,N}
    $(jacobian_chunk_mode_expr(:((xdual = cfg.duals; indices = cfg.indices)),
                               :(ydual = f(xdual)),
                               :(),
                               :(extract_value!(T, result, ydual))))
end

@eval function chunk_mode_jacobian!(result, f!::F, y, x, cfg::JacobianConfig{T,V,N}) where {F,T,V,N}
    $(jacobian_chunk_mode_expr(:(((ydual, xdual) = cfg.duals; (yindices, indices) = cfg.indices)),
                               :(f!(seed_zero_partials!(ydual, y, yindices), xdual)),
                               :(),
                               :(extract_value!(T, result, y, ydual))))
end

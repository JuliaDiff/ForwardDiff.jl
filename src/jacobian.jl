###############
# API methods #
###############

function jacobian(f, x, opts::Options = Options(x))
    if chunksize(opts) == length(x)
        return vector_mode_jacobian(f, x, opts)
    else
        return chunk_mode_jacobian(f, x, opts)
    end
end

function jacobian(f!, y, x, opts::Options = Options(y, x))
    if chunksize(opts) == length(x)
        return vector_mode_jacobian(f!, y, x, opts)
    else
        return chunk_mode_jacobian(f!, y, x, opts)
    end
end

function jacobian!(out, f, x, opts::Options = Options(x))
    if chunksize(opts) == length(x)
        vector_mode_jacobian!(out, f, x, opts)
    else
        chunk_mode_jacobian!(out, f, x, opts)
    end
    return out
end

function jacobian!(out, f!, y, x, opts::Options = Options(y, x))
    @assert !(isa(out, DiffResult)) "use jacobian!(out::DiffResult, f!, x, ...) instead of jacobian!(out::DiffResult, f!, y, x, ...)"
    if chunksize(opts) == length(x)
        vector_mode_jacobian!(out, f!, y, x, opts)
    else
        chunk_mode_jacobian!(out, f!, y, x, opts)
    end
    return out
end

function jacobian!{N,T,D<:Tuple}(out::DiffResult, f!, x, opts::Options{N,T,D} = Options(DiffBase.value(out), x))
    jacobian!(DiffBase.jacobian(out), f!, DiffBase.value(out), x, opts)
    return out
end

#####################
# result extraction #
#####################

function extract_jacobian!(out::AbstractArray, ydual::AbstractArray, n)
    out_reshaped = reshape(out, length(ydual), n)
    for col in 1:size(out_reshaped, 2), row in 1:size(out_reshaped, 1)
        out_reshaped[row, col] = partials(ydual[row], col)
    end
    return out
end

function extract_jacobian!(out::DiffResult, ydual::AbstractArray, n)
    DiffBase.value!(value, out, ydual)
    extract_jacobian!(DiffBase.jacobian(out), ydual, n)
    return out
end

function extract_jacobian_chunk!(out, ydual, index, chunksize)
    offset = index - 1
    for i in 1:chunksize
        col = i + offset
        for row in eachindex(ydual)
            out[row, col] = partials(ydual[row], i)
        end
    end
    return out
end

reshape_jacobian(out, ydual, xdual) = reshape(out, length(ydual), length(xdual))
reshape_jacobian(out::DiffResult, ydual, xdual) = reshape_jacobian(DiffBase.jacobian(out), ydual, xdual)

###############
# vector mode #
###############

function vector_mode_jacobian{N}(f, x, opts::Options{N})
    ydual = vector_mode_dual_eval(f, x, opts)
    out = similar(ydual, valtype(eltype(ydual)), length(ydual), N)
    return extract_jacobian!(out, ydual, N)
end

function vector_mode_jacobian{N,T,D<:Tuple}(f!, y, x, opts::Options{N,T,D})
    ydual = vector_mode_dual_eval(f!, y, x, opts)
    map!(value, y, ydual)
    out = similar(y, length(y), N)
    return extract_jacobian!(out, ydual, N)
end

function vector_mode_jacobian!{N,T,D<:AbstractArray}(out, f, x, opts::Options{N,T,D})
    ydual = vector_mode_dual_eval(f, x, opts)
    extract_jacobian!(out, ydual, N)
    return out
end

function vector_mode_jacobian!{N}(out::AbstractArray, f!, y, x, opts::Options{N})
    ydual = vector_mode_dual_eval(f!, y, x, opts)
    map!(value, y, ydual)
    extract_jacobian!(out, ydual, N)
    return out
end

# chunk mode #
#------------#

function jacobian_chunk_mode_expr(work_array_definition::Expr, compute_ydual::Expr,
                                  out_definition::Expr, y_definition::Expr)
    return quote
        @assert length(x) >= N "chunk size cannot be greater than length(x) ($(N) > $(length(x)))"

        # precalculate loop bounds
        xlen = length(x)
        remainder = xlen % N
        lastchunksize = ifelse(remainder == 0, N, remainder)
        lastchunkindex = xlen - lastchunksize + 1
        middlechunks = 2:div(xlen - lastchunksize, N)

        # seed work arrays
        $(work_array_definition)
        seeds = opts.seeds

        # do first chunk manually to calculate output type
        seed!(xdual, x, 1, seeds)
        $(compute_ydual)
        seed!(xdual, x, 1)
        $(out_definition)
        out_reshaped = reshape_jacobian(out, ydual, xdual)
        extract_jacobian_chunk!(out_reshaped, ydual, 1, N)

        # do middle chunks
        for c in middlechunks
            i = ((c - 1) * N + 1)
            seed!(xdual, x, i, seeds)
            $(compute_ydual)
            seed!(xdual, x, i)
            extract_jacobian_chunk!(out_reshaped, ydual, i, N)
        end

        # do final chunk
        seed!(xdual, x, lastchunkindex, seeds, lastchunksize)
        $(compute_ydual)
        extract_jacobian_chunk!(out_reshaped, ydual, lastchunkindex, lastchunksize)

        $(y_definition)

        return out
    end
end

@eval function chunk_mode_jacobian{N}(f, x, opts::Options{N})
    $(jacobian_chunk_mode_expr(quote
                                   xdual = opts.duals
                                   seed!(xdual, x)
                               end,
                               :(ydual = f(xdual)),
                               :(out = similar(ydual, valtype(eltype(ydual)), length(ydual), xlen)),
                               :()))
end

@eval function chunk_mode_jacobian{N}(f!, y, x, opts::Options{N})
    $(jacobian_chunk_mode_expr(quote
                                   ydual, xdual = opts.duals
                                   seed!(xdual, x)
                               end,
                               :(f!(seed!(ydual, y), xdual)),
                               :(out = similar(y, length(y), xlen)),
                               :(map!(value, y, ydual))))
end

@eval function chunk_mode_jacobian!{N}(out, f, x, opts::Options{N})
    $(jacobian_chunk_mode_expr(quote
                                   xdual = opts.duals
                                   seed!(xdual, x)
                               end,
                               :(ydual = f(xdual)),
                               :(),
                               :(extract_value!(out, ydual))))
end

@eval function chunk_mode_jacobian!{N}(out, f!, y, x, opts::Options{N})
    $(jacobian_chunk_mode_expr(quote
                                   ydual, xdual = opts.duals
                                   seed!(xdual, x)
                               end,
                               :(f!(seed!(ydual, y), xdual)),
                               :(),
                               :(map!(value, y, ydual))))
end

#################
# HessianResult #
#################

type HessianResult{V,G,H} <: ForwardDiffResult
    value::V
    gradient::G
    hessian::H
end

HessianResult(x) = HessianResult(first(x), similar(x), similar(x, length(x), length(x)))

value(result::HessianResult) = result.value
gradient(result::HessianResult) = result.gradient
hessian(result::HessianResult) = result.hessian

###############
# API methods #
###############

function hessian{N}(f, x, chunk::Chunk{N} = pickchunk(x);
                    multithread::Bool = false,
                    usecache::Bool = true)
    if N == length(x)
        return vector_mode_hessian(f, x, chunk, usecache)
    else
        return chunk_mode_hessian(f, x, chunk, multithread, usecache)
    end
end

function hessian!{N}(out, f, x, chunk::Chunk{N} = pickchunk(x);
                     multithread::Bool = false,
                     usecache::Bool = true)
    if N == length(x)
        vector_mode_hessian!(out, f, x, chunk, usecache)
    else
        chunk_mode_hessian!(out, f, x, chunk, multithread, usecache)
    end
    return out
end

#######################
# workhorse functions #
#######################

# result extraction #
#-------------------#

@inline load_hessian_value!(out, dual) = out

function load_hessian_value!(out::HessianResult, dual)
    out.value = value(value(dual))
    return out
end

@inline load_hessian_gradient!(out, dual) = out

function load_hessian_gradient!(out::HessianResult, dual)
    grad = out.gradient
    val = value(dual)
    for i in eachindex(grad)
        grad[i] = partials(val, i)
    end
    return out
end

function load_hessian!(out, dual)
    for col in 1:size(out, 2), row in 1:size(out, 1)
        out[row, col] = partials(dual, row, col)
    end
    return out
end

@inline function load_hessian!(out::HessianResult, dual)
    load_hessian!(out.hessian, dual)
    return out
end

# vector mode #
#-------------#

function compute_vector_mode_hessian(f, x, chunk, usecache)
    cache = hessian_cachefetch!(x, chunk, usecache)
    xdual = cache.duals
    seedhess!(xdual, x, cache.inseeds, cache.outseeds)
    return f(xdual)
end

function vector_mode_hessian(f, x, chunk, usecache)
    dual = compute_vector_mode_hessian(f, x, chunk, usecache)
    out = similar(x, numtype(numtype(dual)), length(x), length(x))
    return load_hessian!(out, dual)
end

function vector_mode_hessian!(out, f, x, chunk, usecache)
    dual = compute_vector_mode_hessian(f, x, chunk, usecache)
    load_hessian_value!(out, dual)
    load_hessian_gradient!(out, dual)
    load_hessian!(out, dual)
    return out
end

# chunk mode #
#------------#

function chunk_mode_hessian(f, x, chunk::Chunk, multithread::Bool, usecache::Bool)
    gradf = y -> ForwardDiff.gradient(f, y, chunk; multithread = multithread, usecache = usecache)
    return ForwardDiff.jacobian(gradf, x, chunk; usecache = usecache)
end

function chunk_mode_hessian!(out, f, x, chunk::Chunk, multithread::Bool, usecache::Bool)
    gradf = y -> ForwardDiff.gradient(f, y, chunk; multithread = multithread, usecache = usecache)
    ForwardDiff.jacobian!(out, gradf, x, chunk; usecache = usecache)
    return out
end

function chunk_mode_hessian!{T,N}(out::HessianResult{T}, f, x, chunk::Chunk{N},
                                  multithread::Bool, usecache::Bool)
    gradvec = similar(out.gradient, Dual{N,T})
    gradresult = GradientResult(first(gradvec), gradvec)
    gradf = y -> begin
        ForwardDiff.gradient!(gradresult, f, y, chunk; multithread = multithread, usecache = usecache)
        return ForwardDiff.gradient(gradresult)
    end
    ForwardDiff.jacobian!(out.hessian, gradf, x, chunk; usecache = usecache)
    out.value = ForwardDiff.value(ForwardDiff.value(gradresult))
    for i in eachindex(gradvec)
        out.gradient[i] = ForwardDiff.value(gradvec[i])
    end
    return out
end

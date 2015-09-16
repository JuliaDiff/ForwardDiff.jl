# Note for the files in this folder: 
# Following convention, methods whose names are prefixed 
# with an underscore are unsafe to use outside of a strictly 
# controlled context - such methods assume that all 
# boundary-checking is done by the caller.

const tuple_usage_threshold = 10
const default_chunk_size = 0

abstract AllResults

function check_chunk_size(xlen::Int, chunk_size::Int)        
    if chunk_size != default_chunk_size
        @assert chunk_size > 0 "Invalid chunk_size: $chunk_size. chunk_size cannot be negative."
        @assert xlen % chunk_size == 0 "Length of input vector is indivisible by chunk size (length(x) = $xlen, chunk size = $chunk_size)"
    end
end

function chunk_size_matches_vec_mode(xlen::Int, chunk_size::Int)
    return (chunk_size == default_chunk_size) || (chunk_size == xlen)
end

include("cache.jl")

const dummy_cache = make_dummy_cache()

include("ForwardDiffResult.jl")
include("derivative.jl")
include("gradient.jl")
include("jacobian.jl")
include("hessian.jl")
include("tensor.jl")
include("deprecated.jl")

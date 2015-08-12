# Note for the files in this folder: 
# Following convention, methods whose names are prefixed 
# with an underscore are unsafe to use outside of a strictly 
# controlled context - such methods assume that all 
# boundary-checking is done by the caller.

const tuple_usage_threshold = 10
const default_chunk = 0

function check_chunk_size(x::Vector, chunk_size::Int)
    @assert length(x) % chunk_size == 0 "Length of input vector is indivisible by chunk size (length(x) = $(length(x)), chunk size = $chunk_size)"
end

function chunk_size_matches_full(x::Vector, chunk_size::Int)
    return (chunk_size == default_chunk) || (chunk_size == length(x))
end

include("cache.jl")
include("derivative.jl")
include("gradient.jl")
include("jacobian.jl")
include("hessian.jl")
include("tensor.jl")
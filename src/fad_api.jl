######################
# Taking Derivatives #
######################

# Exposed API methods #
#---------------------#
derivative!(output::Array, f, x::Number) = load_derivative!(output, f(GradientNumber(x, one(x))))
derivative(f, x::Number) = load_derivative(f(GradientNumber(x, one(x))))

function derivative(f; mutates=false)
    if mutates
        derivf!(output::Array, x::Number) = derivative!(output, f, x)
        return derivf!
    else
        derivf(x::Number) = derivative(f, x)
        return derivf
    end
end

# Helper functions #
#------------------#
function load_derivative!(output::Array, arr::Array)
    @assert length(arr) == length(output)
    @simd for i in eachindex(output)
        @inbounds output[i] = grad(arr[i], 1)
    end
    return output
end

load_derivative(arr::Array) = load_derivative!(similar(arr, eltype(eltype(arr))), arr)
load_derivative(n::ForwardDiffNumber{1}) = grad(n, 1)

####################
# Taking Gradients #
####################

# Exposed API methods #
#---------------------#
function gradient!{T}(output::Vector{T}, f, x::Vector; chunk_size=nothing)
    gradvec = gen_gradvec(x, chunk_size)
    return calc_gradient!(output, f, x, gradvec)
end

function gradient{T}(f, x::Vector{T}; chunk_size=nothing)
    gradvec = gen_gradvec(x, chunk_size)
    return calc_gradient!(similar(x), f, x, gradvec)
end

function gradient(f; mutates=false)
    gradvecs = Dict()
    partial_chunks = Dict()
    if mutates
        function gradf!{T}(output::Vector, x::Vector{T}; chunk_size=nothing)
            gradvec = pick_gradvec!(gradvecs, x, chunk_size)
            pchunk = pick_partials!(partial_chunks, eltype(gradvec))
            return calc_gradient!(output, f, x, gradvec, pchunk)
        end
        return gradf!
    else
        function gradf{T}(x::Vector{T}; chunk_size=nothing)
            gradvec = pick_gradvec!(gradvecs, x, chunk_size)
            pchunk = pick_partials!(partial_chunks, eltype(gradvec))
            return calc_gradient!(similar(x), f, x, gradvec, pchunk)
        end
        return gradf
    end
end

# Calculate gradient of a given function #
#----------------------------------------#
function calc_gradient!{S,N,T,C}(output::Vector{S},
                                 f::Function,
                                 x::Vector{T},
                                 gradvec::Vector{GradientNumber{N,T,C}},
                                 pchunk=gen_partials_chunk(eltype(gradvec))) 
    xlen = length(x)
    G = eltype(gradvec)

    @assert xlen == length(output) "The output array must be the same length as x"
    @assert xlen == length(gradvec) "The GradientNumber vector must be the same length as the input vector"
    @assert xlen % N == 0 "Length of input vector is indivisible by chunk size (length(x) = $xlen, chunk size = $N)"

    # We can do less work filling and
    # zeroing out gradvec if xlen == N
    if xlen == N
        @simd for i in eachindex(gradvec)
            @inbounds gradvec[i] = G(x[i], pchunk[i])
        end

        result = f(gradvec)

        @simd for i in eachindex(output)
            @inbounds output[i] = grad(result, i)
        end
    else
        zpartials = zero_partials(G)

        # load x[i]-valued GradientNumbers into gradvec 
        @simd for i in 1:xlen
            @inbounds gradvec[i] = G(x[i], zpartials)
        end

        for i in 1:N:xlen
            # load GradientNumbers with single
            # partial components into current 
            # chunk of gradvec
            @simd for j in 0:(N-1)
                m = i+j
                @inbounds gradvec[m] = G(x[m], pchunk[j+1])
            end

            chunk_result = f(gradvec)

            # load resultant partials components
            # into output, replacing them with 
            # zeros in gradvec
            @simd for j in 0:(N-1)
                m = i+j
                @inbounds output[m] = grad(chunk_result, j+1)
                @inbounds gradvec[m] = G(x[m], zpartials)
            end
        end
    end

    return output::Vector{S}
end

# Helper functions #
#------------------#
zero_partials{N,T}(::Type{GradNumVec{N,T}}) = zeros(T, N)
zero_partials{N,T}(::Type{GradNumTup{N,T}}) = (z = zero(T); return ntuple(i->z, Val{N}))

function pick_gradvec!{T}(gradvecs::Dict, x::Vector{T}, chunk_size)
    key = (T, length(x), chunk_size)
    if haskey(gradvecs, key)
        return gradvecs[key]
    else
        gradvec = gen_gradvec(x, chunk_size)
        gradvecs[key] = gradvec
        return gradvec
    end
end

function pick_partials!{N,T,C}(partial_chunks::Dict, G::Type{GradientNumber{N,T,C}})
    if haskey(partial_chunks, G)
        return partial_chunks[G]
    else
        pchunk = gen_partials_chunk(G)
        partial_chunks[G] = pchunk
        return pchunk
    end
end

function gen_gradvec{T}(x::Vector{T}, chunk_size::Int)
    N = chunk_size
    return Vector{GradientNumber{N,T,NTuple{N,T}}}(length(x))
end

function gen_gradvec{T}(x::Vector{T}, chunk_size::Void)
    N = length(x)
    return Vector{GradientNumber{N,T,pick_implementation(T, N)}}(length(x))
end

####################
# Taking Jacobians #
####################

# Exposed API methods #
#---------------------#
# TODO

# Calculate Jacobian of a given function #
#----------------------------------------#
# TODO

# Helper functions #
#------------------#
# TODO

###################
# Taking Hessians #
###################

# Exposed API methods #
#---------------------#
# TODO

# Calculate Hessian of a given function #
#---------------------------------------#
# TODO

# Helper functions #
#------------------#
zero_partials{N,T,C}(::Type{HessianNumber{N,T,C}}) = zeros(T, halfhesslen(N))

##################
# Taking Tensors #
##################

# Exposed API methods #
#---------------------#
# TODO

# Calculate third order Taylor series term of a given function #
#--------------------------------------------------------------#
# TODO

# Helper functions #
#------------------#
zero_partials{N,T,C}(::Type{TensorNumber{N,T,C}}) = zeros(T, halftenslen(N))

############################
# General Helper Functions #
############################
const tuple_usage_threshold = 10

function pick_implementation{T}(::Type{T}, chunk_size::Int)
    return chunk_size > tuple_usage_threshold ? Vector{T} : NTuple{chunk_size,T}
end

function gen_partials_chunk{N,T}(::Type{GradNumVec{N,T}})
    chunk_arr = Array(Vector{T}, N)
    @simd for i in eachindex(chunk_arr)
        @inbounds chunk_arr[i] = setindex!(zeros(T, N), one(T), i)
    end
    return chunk_arr
end

@generated function gen_partials_chunk{N,T}(::Type{GradNumTup{N,T}})
    
    if N > tuple_usage_threshold
        ex = quote
            pchunk = Vector{NTuple{$N,$T}}($N)
            @simd for i in eachindex(pchunk)
                @inbounds pchunk[i] = ntuple(x -> ifelse(x == i, o, z), Val{$N})
            end
            return pchunk
        end
    else
        ex = quote
            return ntuple(i -> ntuple(x -> ifelse(x == i, o, z), Val{$N}), Val{$N})
        end
    end

    return quote 
        z = zero(T)
        o = one(T)
        $ex
    end
end
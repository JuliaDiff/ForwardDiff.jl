# NOTE: Following convention, methods whose names are
# prefixed with an underscore are unsafe to use outside of
# a strictly controlled context - such methods assume that
# all boundary-checking is done by the caller.

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
function gradient!(output::Vector, f, x::Vector; chunk_size=nothing)
    gradvec = gen_gradvec(x, chunk_size)
    return calc_gradient!(output, f, x, gradvec)
end

function gradient(f, x::Vector; chunk_size=nothing)
    gradvec = gen_gradvec(x, chunk_size)
    return calc_gradient!(similar(x), f, x, gradvec)
end

function gradient(f; mutates=false)
    gradvecs = Dict()
    partial_chunks = Dict()
    if mutates
        function gradf!(output::Vector, x::Vector; chunk_size=nothing)
            gradvec = pick_gradvec!(gradvecs, x, chunk_size)
            pchunk = pick_partials!(partial_chunks, eltype(gradvec))
            return calc_gradient!(output, f, x, gradvec, pchunk)
        end
        return gradf!
    else
        function gradf(x::Vector; chunk_size=nothing)
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
                                 f,
                                 x::Vector{T},
                                 gradvec::Vector{GradientNumber{N,T,C}},
                                 pchunk=gen_partials_chunk(eltype(gradvec))) 
    xlen = length(x)
    G = eltype(gradvec)

    @assert xlen == length(output) "The output vector must be the same length as the input vector"
    perform_gradvec_assertions(xlen, gradvec)

    if xlen == N
        _load_gradvec_with_x_partials!(gradvec, x, pchunk)

        result = f(gradvec)

        @simd for i in eachindex(output)
            @inbounds output[i] = grad(result, i)
        end
    else
        zpartials = zero_partials(G)

        _load_gradvec_with_x_zeros!(gradvec, x, zpartials)

        for i in 1:N:xlen
            _load_gradvec_with_x_partials!(gradvec, x, pchunk, i)
            
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
function perform_gradvec_assertions{N,T,C}(xlen, gradvec::Vector{GradientNumber{N,T,C}})
    @assert xlen == length(gradvec) "The GradientNumber vector must be the same length as the input vector"
    @assert xlen % N == 0 "Length of input vector is indivisible by chunk size (length(x) = $xlen, chunk size = $N)"
end

function _load_gradvec_with_x_partials!(gradvec, x, pchunk)
    # fill gradvec with GradientNumbers of single partial components
    G = eltype(gradvec)
    @simd for i in eachindex(gradvec)
        @inbounds gradvec[i] = G(x[i], pchunk[i])
    end
end

function _load_gradvec_with_x_partials!{N,T,C}(gradvec::Vector{GradientNumber{N,T,C}}, x, pchunk, init)
    # fill current chunk gradvec with GradientNumbers of single partial components
    G = eltype(gradvec)
    @simd for j in 0:(N-1)
        m = init+j
        @inbounds gradvec[m] = G(x[m], pchunk[j+1])
    end
end

function _load_gradvec_with_x_zeros!(gradvec, x, zpartials)
    # fill gradvec with x[i]-valued GradientNumbers
    G = eltype(gradvec)
    @simd for i in eachindex(x)
        @inbounds gradvec[i] = G(x[i], zpartials)
    end
end

zero_partials{N,T}(::Type{GradNumVec{N,T}}) = zeros(T, N)
zero_partials{N,T}(::Type{GradNumTup{N,T}}) = zero_tuple(NTuple{N,T})

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
function jacobian!(output::Matrix, f, x::Vector; chunk_size=nothing)
    gradvec = gen_gradvec(x, chunk_size)
    return calc_jacobian!(output, f, x, gradvec)
end

function jacobian(f, x::Vector; chunk_size=nothing)
    gradvec = gen_gradvec(x, chunk_size)
    return calc_jacobian(f, x, gradvec)
end

function jacobian(f; mutates=false)
    gradvecs = Dict()
    partial_chunks = Dict()
    if mutates
        function jacf!(output::Matrix, x::Vector; chunk_size=nothing)
            gradvec = pick_gradvec!(gradvecs, x, chunk_size)
            pchunk = pick_partials!(partial_chunks, eltype(gradvec))
            return calc_jacobian!(output, f, x, gradvec, pchunk)
        end
        return jacf!
    else
        function jacf(x::Vector; chunk_size=nothing)
            gradvec = pick_gradvec!(gradvecs, x, chunk_size)
            pchunk = pick_partials!(partial_chunks, eltype(gradvec))
            return calc_jacobian(f, x, gradvec, pchunk)
        end
        return jacf
    end
end

# Calculate Jacobian of a given function #
#----------------------------------------#
function calc_jacobian!{S,N,T,C}(output::Matrix{S},
                                 f,
                                 x::Vector{T},
                                 gradvec::Vector{GradientNumber{N,T,C}},
                                 pchunk=gen_partials_chunk(eltype(gradvec))) 
    xlen = length(x)
    G = eltype(gradvec)

    @assert xlen == size(output, 2) "The output matrix must have a number of columns equal to the length of the input vector"
    perform_gradvec_assertions(xlen, gradvec)

    if xlen == N
        _load_gradvec_with_x_partials!(gradvec, x, pchunk)

        result = f(gradvec)

        for i in eachindex(result), j in eachindex(x)
            output[i,j] = grad(result[i], j)
        end
    else
        zpartials = zero_partials(G)

        _load_gradvec_with_x_zeros!(gradvec, x, zpartials)

        _calc_jac_chunks!(output, f, x, gradvec, pchunk, zpartials, 1)
    end

    return output::Matrix{S}
end

function calc_jacobian{N,T,C}(f,
                              x::Vector{T},
                              gradvec::Vector{GradientNumber{N,T,C}},
                              pchunk=gen_partials_chunk(eltype(gradvec))) 
    xlen = length(x)
    G = eltype(gradvec)

    perform_gradvec_assertions(xlen, gradvec)

    if xlen == N
        _load_gradvec_with_x_partials!(gradvec, x, pchunk)

        result = f(gradvec)
        output = Array(T, length(result), xlen)

        for i in eachindex(result), j in eachindex(x)
            output[i,j] = grad(result[i], j)
        end
    else
        zpartials = zero_partials(G)

        _load_gradvec_with_x_zeros!(gradvec, x, zpartials)

        # Perform the first chunk "manually" so that
        # we get to inspect the size of the output
        i = 1

        _load_gradvec_with_x_partials!(gradvec, x, pchunk, i)

        chunk_result = f(gradvec)

        output = Array(T, length(chunk_result), xlen)

        for j in 0:(N-1)
            m = i+j
            current_partial = j+1
            for n in 1:size(output,1)
                @inbounds output[n,m] = grad(chunk_result[n], current_partial)
            end
            @inbounds gradvec[m] = G(x[m], zpartials)
        end

        # Now perform the rest of the chunks, filling in the output matrix
        _calc_jac_chunks!(output, f, x, gradvec, pchunk, zpartials, N+1)
    end

    return output::Matrix{T}
end

# Helper functions #
#------------------#
function _calc_jac_chunks!{N,T,C}(output,
                                  f,
                                  x::Vector{T}, 
                                  gradvec::Vector{GradientNumber{N,T,C}},
                                  pchunk, 
                                  zpartials, 
                                  init)
    G = eltype(gradvec)

    for i in init:N:length(x)
        _load_gradvec_with_x_partials!(gradvec, x, pchunk, i)

        chunk_result = f(gradvec)

        for j in 0:(N-1)
            m = i+j
            current_partial = j+1
            for n in 1:size(output,1)
                @inbounds output[n,m] = grad(chunk_result[n], current_partial)
            end
            @inbounds gradvec[m] = G(x[m], zpartials)
        end
    end

    return output
end

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
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
    gradvec = build_workvec(GradientNumber, x, chunk_size)
    return _calc_gradient!(output, f, x, gradvec)
end

function gradient(f, x::Vector; chunk_size=nothing) 
    return gradient!(similar(x), f, x, chunk_size=chunk_size)
end

function gradient(f; mutates=false)
    gradvecs = Dict()
    partial_chunks = Dict()
    if mutates
        function gradf!(output::Vector, x::Vector; chunk_size=nothing)
            gradvec = pick_workvec!(gradvecs, GradientNumber, x, chunk_size)
            partials_chunk = pick_partials!(partial_chunks, eltype(gradvec))
            return _calc_gradient!(output, f, x, gradvec, partials_chunk)
        end
        return gradf!
    else
        function gradf(x::Vector; chunk_size=nothing)
            gradvec = pick_workvec!(gradvecs, GradientNumber, x, chunk_size)
            partials_chunk = pick_partials!(partial_chunks, eltype(gradvec))
            return _calc_gradient!(similar(x), f, x, gradvec, partials_chunk)
        end
        return gradf
    end
end

# Calculate gradient of a given function #
#----------------------------------------#
function _calc_gradient!{S,N,T,C}(output::Vector{S},
                                  f,
                                  x::Vector{T},
                                  gradvec::Vector{GradientNumber{N,T,C}},
                                  partials_chunk=build_partials_chunk(eltype(gradvec))) 
    xlen = length(x)
    G = eltype(gradvec)

    @assert xlen == length(output) "The output vector must be the same length as the input vector"
    chunk_assertion(xlen, N)

    if xlen == N
        _load_gradvec_with_x_partials!(gradvec, x, partials_chunk)

        result = f(gradvec)

        @simd for i in eachindex(output)
            @inbounds output[i] = grad(result, i)
        end
    else
        zero_partials = build_zero_partials(G)

        _load_gradvec_with_x_zeros!(gradvec, x, zero_partials)

        for i in 1:N:xlen
            _load_gradvec_with_x_partials!(gradvec, x, partials_chunk, i)
            
            chunk_result = f(gradvec)

            # load resultant partials components
            # into output, replacing them with 
            # zeros in gradvec
            @simd for j in 1:N
                q = i+j-1
                @inbounds output[q] = grad(chunk_result, j)
                @inbounds gradvec[q] = G(x[q], zero_partials)
            end
        end
    end

    return output::Vector{S}
end

# Helper functions #
#------------------#

function _load_gradvec_with_x_partials!(gradvec, x, partials_chunk)
    # fill gradvec with GradientNumbers of single partial components
    G = eltype(gradvec)
    @simd for i in eachindex(gradvec)
        @inbounds gradvec[i] = G(x[i], partials_chunk[i])
    end
end

function _load_gradvec_with_x_partials!{N,T,C}(gradvec::Vector{GradientNumber{N,T,C}}, x, partials_chunk, init)
    # fill current chunk gradvec with GradientNumbers of single partial components
    G = eltype(gradvec)
    @simd for j in 1:N
        m = init+j-1
        @inbounds gradvec[m] = G(x[m], partials_chunk[j])
    end
end

function _load_gradvec_with_x_zeros!(gradvec, x, zero_partials)
    # fill gradvec with x[i]-valued GradientNumbers
    G = eltype(gradvec)
    @simd for i in eachindex(x)
        @inbounds gradvec[i] = G(x[i], zero_partials)
    end
end

build_zero_partials{N,T}(::Type{GradNumVec{N,T}}) = zeros(T, N)
build_zero_partials{N,T}(::Type{GradNumTup{N,T}}) = zero_tuple(NTuple{N,T})

####################
# Taking Jacobians #
####################

# Exposed API methods #
#---------------------#
function jacobian!(output::Matrix, f, x::Vector; chunk_size=nothing)
    gradvec = build_workvec(GradientNumber, x, chunk_size)
    return _calc_jacobian!(output, f, x, gradvec)
end

function jacobian(f, x::Vector; chunk_size=nothing)
    gradvec = build_workvec(GradientNumber, x, chunk_size)
    return _calc_jacobian(f, x, gradvec)
end

function jacobian(f; mutates=false)
    gradvecs = Dict()
    partial_chunks = Dict()
    if mutates
        function jacf!(output::Matrix, x::Vector; chunk_size=nothing)
            gradvec = pick_workvec!(gradvecs, GradientNumber, x, chunk_size)
            partials_chunk = pick_partials!(partial_chunks, eltype(gradvec))
            return _calc_jacobian!(output, f, x, gradvec, partials_chunk)
        end
        return jacf!
    else
        function jacf(x::Vector; chunk_size=nothing)
            gradvec = pick_workvec!(gradvecs, GradientNumber, x, chunk_size)
            partials_chunk = pick_partials!(partial_chunks, eltype(gradvec))
            return _calc_jacobian(f, x, gradvec, partials_chunk)
        end
        return jacf
    end
end

# Calculate Jacobian of a given function #
#----------------------------------------#
function _calc_jacobian!{S,N,T,C}(output::Matrix{S},
                                  f,
                                  x::Vector{T},
                                  gradvec::Vector{GradientNumber{N,T,C}},
                                  partials_chunk=build_partials_chunk(eltype(gradvec))) 
    xlen = length(x)
    G = eltype(gradvec)

    @assert xlen == size(output, 2) "The output matrix must have a number of columns equal to the length of the input vector"
    chunk_assertion(xlen, N)

    if xlen == N
        _load_gradvec_with_x_partials!(gradvec, x, partials_chunk)

        result = f(gradvec)

        for i in eachindex(result), j in eachindex(x)
            output[i,j] = grad(result[i], j)
        end
    else
        zero_partials = build_zero_partials(G)

        _load_gradvec_with_x_zeros!(gradvec, x, zero_partials)

        _calc_jac_chunks!(output, f, x, gradvec, partials_chunk, zero_partials, 1)
    end

    return output::Matrix{S}
end

function _calc_jacobian{N,T,C}(f,
                               x::Vector{T},
                               gradvec::Vector{GradientNumber{N,T,C}},
                               partials_chunk=build_partials_chunk(eltype(gradvec))) 
    xlen = length(x)
    G = eltype(gradvec)

    chunk_assertion(xlen, N)

    if xlen == N
        _load_gradvec_with_x_partials!(gradvec, x, partials_chunk)

        result = f(gradvec)
        output = Array(T, length(result), xlen)

        for i in eachindex(result), j in eachindex(x)
            output[i,j] = grad(result[i], j)
        end
    else
        zero_partials = build_zero_partials(G)

        _load_gradvec_with_x_zeros!(gradvec, x, zero_partials)

        # Perform the first chunk "manually" so that
        # we get to inspect the size of the output
        i = 1

        _load_gradvec_with_x_partials!(gradvec, x, partials_chunk, i)

        chunk_result = f(gradvec)

        output = Array(T, length(chunk_result), xlen)

        for j in 1:N
            m = i+j-1
            for n in 1:size(output,1)
                @inbounds output[n,m] = grad(chunk_result[n], j)
            end
            @inbounds gradvec[m] = G(x[m], zero_partials)
        end

        # Now perform the rest of the chunks, filling in the output matrix
        _calc_jac_chunks!(output, f, x, gradvec, partials_chunk, zero_partials, N+1)
    end

    return output::Matrix{T}
end

# Helper functions #
#------------------#
function _calc_jac_chunks!{N,T,C}(output,
                                  f,
                                  x::Vector{T}, 
                                  gradvec::Vector{GradientNumber{N,T,C}},
                                  partials_chunk, 
                                  zero_partials, 
                                  init)
    G = eltype(gradvec)

    for i in init:N:length(x)
        _load_gradvec_with_x_partials!(gradvec, x, partials_chunk, i)

        chunk_result = f(gradvec)

        for j in 1:N
            m = i+j-1
            for n in 1:size(output,1)
                @inbounds output[n,m] = grad(chunk_result[n], j)
            end
            @inbounds gradvec[m] = G(x[m], zero_partials)
        end
    end

    return output
end

###################
# Taking Hessians #
###################

# Exposed API methods #
#---------------------#
function hessian!(output::Matrix, f, x::Vector; chunk_size=nothing)
    hessvec = build_workvec(HessianNumber, x, chunk_size)
    return _calc_hessian!(output, f, x, hessvec)
end

function hessian(f, x::Vector; chunk_size=nothing)
    xlen = length(x)
    return hessian!(similar(x, xlen, xlen), f, x, chunk_size=chunk_size)
end

function hessian(f; mutates=false)
    hessvecs = Dict()
    partial_chunks = Dict()
    if mutates
        function hessf!(output::Matrix, x::Vector; chunk_size=nothing)
            hessvec = pick_workvec!(hessvecs, HessianNumber, x, chunk_size)
            partials_chunk = pick_partials!(partial_chunks, eltype(hessvec))
            return _calc_hessian!(output, f, x, hessvec, partials_chunk)
        end
        return hessf!
    else
        function hessf(x::Vector; chunk_size=nothing)
            hessvec = pick_workvec!(hessvecs, HessianNumber, x, chunk_size)
            partials_chunk = pick_partials!(partial_chunks, eltype(hessvec))
            xlen = length(x)
            return _calc_hessian!(similar(x, xlen, xlen), f, x, hessvec, partials_chunk)
        end
        return hessf
    end
end

# Calculate Hessian of a given function #
#---------------------------------------#
function _calc_hessian!{S,N,T,C}(output::Matrix{S},
                                 f,
                                 x::Vector{T},
                                 hessvec::Vector{HessianNumber{N,T,C}},
                                 partials_chunk=build_partials_chunk(eltype(hessvec))) 
    xlen = length(x)
    G = GradientNumber{N,T,C}
    H = eltype(hessvec)

    @assert (xlen, xlen) == size(output) "The output matrix must have size (length(input), length(input))"
    
    partials_chunk = build_partials_chunk(G)
    zero_hess_partials = zeros(T, halfhesslen(N))

    if xlen == N
        chunk_assertion(xlen, N)

        @simd for i in eachindex(hessvec)
            @inbounds hessvec[i] = H(G(x[i], partials_chunk[i]), zero_hess_partials)
        end

        result = f(hessvec)

        q = 1
        for i in 1:N
            for j in 1:i
                val = hess(result, q)
                @inbounds output[i, j] = val
                @inbounds output[j, i] = val
                q += 1
            end
        end
    else
        # Keep in mind that build_work_vec increments 
        # input chunk_size by one for HessianNumbers
        # when users input a non-xlen chunk_size (this 
        # allows simplification of loop alignment for 
        # the chunk evaluating code below)
        M = N-1
        chunk_assertion(xlen, M)

        zero_grad_partials = build_zero_partials(G)

        @simd for i in eachindex(x)
            @inbounds hessvec[i] = H(G(x[i], zero_grad_partials), zero_hess_partials) 
        end

        # The below loop fills triangular blocks 
        # along diagonal. The size of these blocks
        # is determined by the chunk size.
        #
        # For example, if N = 3 and xlen = 6, the 
        # numbers inside the slots below indicate the
        # iteration of the loop (i.e. ith call of f) 
        # in which they are filled:
        #
        # Hessian matrix:
        # -------------------------
        # | 1 | 1 |   |   |   |   |
        # -------------------------
        # | 1 | 1 |   |   |   |   |
        # -------------------------
        # |   |   | 2 | 2 |   |   |
        # -------------------------
        # |   |   | 2 | 2 |   |   |
        # -------------------------
        # |   |   |   |   | 3 | 3 |
        # -------------------------
        # |   |   |   |   | 3 | 3 |
        # -------------------------
        for i in 1:M:xlen
            @simd for j in 1:N
                q = i+j-1
                @inbounds hessvec[q] = H(G(x[q], partials_chunk[j]), zero_hess_partials)
            end

            chunk_result = f(hessvec)
            
            q = 1
            for j in i:(i+M-1)
                for k in i:j
                    val = hess(chunk_result, q)
                    @inbounds output[j, k] = val
                    @inbounds output[k, j] = val
                    q += 1
                end
            end

            @simd for j in 1:N
                q = i+j-1
                @inbounds hessvec[q] = H(G(x[q], zero_grad_partials), zero_hess_partials)
            end
        end

        # The below loop fills in the rest. Once 
        # again, using N = 3 and xlen = 6, with each 
        # iteration (i.e. ith call of f) filling the 
        # corresponding slots, and where 'x' indicates
        # previously filled slots:
        #
        # -------------------------
        # | x | x | 1 | 1 | 2 | 2 |
        # -------------------------
        # | x | x | 3 | 3 | 4 | 4 |
        # -------------------------
        # | 1 | 3 | x | x | 5 | 5 |
        # -------------------------
        # | 1 | 3 | x | x | 6 | 6 |
        # -------------------------
        # | 2 | 4 | 5 | 6 | x | x |
        # -------------------------
        # | 2 | 4 | 5 | 6 | x | x |
        # -------------------------
        for offset in M:M:(xlen-M)
            col_offset = offset - M
            for j in 1:M
                col = col_offset + j
                @inbounds hessvec[col] = H(G(x[col], partials_chunk[1]), zero_hess_partials)
                for row_offset in offset:M:(xlen-1)
                    for i in 1:M
                        row = row_offset + i
                        @inbounds hessvec[row] = H(G(x[row], partials_chunk[i+1]), zero_hess_partials)
                    end

                    chunk_result = f(hessvec)

                    for i in 1:M
                        row = row_offset + i
                        q = halfhesslen(i) + 1
                        val = hess(chunk_result, q)
                        @inbounds output[row, col] = val
                        @inbounds output[col, row] = val
                        @inbounds hessvec[row] = H(G(x[row], zero_grad_partials), zero_hess_partials)
                    end
                end
                @inbounds hessvec[col] = H(G(x[col], zero_grad_partials), zero_hess_partials)
            end
        end
    end

    return output::Matrix{S}
end

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
build_zero_partials{N,T,C}(::Type{TensorNumber{N,T,C}}) = zeros(T, halftenslen(N))

############################
# General Helper Functions #
############################
const tuple_usage_threshold = 10

function pick_implementation{T}(::Type{T}, chunk_size::Int)
    return chunk_size > tuple_usage_threshold ? Vector{T} : NTuple{chunk_size,T}
end

function chunk_assertion(xlen, N)
    @assert xlen % N == 0 "Length of input vector is indivisible by chunk size (length(x) = $xlen, chunk size = $N)"
end

# Work Vectors #
#--------------#
function build_workvec{F,T}(::Type{F}, x::Vector{T}, chunk_size::Int)
    N = chunk_size
    return Vector{F{N,T,NTuple{N,T}}}(length(x))
end

function build_workvec{F,T}(::Type{F}, x::Vector{T}, chunk_size::Void)
    N = length(x)
    return Vector{F{N,T,pick_implementation(T, N)}}(length(x))
end

function build_workvec{T}(::Type{HessianNumber}, x::Vector{T}, chunk_size::Int)
    N = chunk_size == length(x) ? chunk_size : chunk_size + 1
    return Vector{HessianNumber{N,T,NTuple{N,T}}}(length(x))
end

# Cache retrieval #
#-----------------#
function pick_workvec!{F,T}(workvecs::Dict, ::Type{F}, x::Vector{T}, chunk_size)
    key = (T, length(x), chunk_size)
    if haskey(workvecs, key)
        return workvecs[key]
    else
        workvec = build_workvec(F, x, chunk_size)
        workvecs[key] = workvec
        return workvec
    end
end

function pick_partials!{F}(partial_chunks::Dict, ::Type{F})
    if haskey(partial_chunks, F)
        return partial_chunks[F]
    else
        partials_chunk = build_partials_chunk(F)
        partial_chunks[F] = partials_chunk
        return partials_chunk
    end
end

# Partials chunks #
#-----------------#
function build_partials_chunk{N,T}(::Type{GradNumVec{N,T}})
    chunk_arr = Array(Vector{T}, N)
    @simd for i in eachindex(chunk_arr)
        @inbounds chunk_arr[i] = setindex!(zeros(T, N), one(T), i)
    end
    return chunk_arr
end

@generated function build_partials_chunk{N,T}(::Type{GradNumTup{N,T}})
    
    if N > tuple_usage_threshold
        ex = quote
            partials_chunk = Vector{NTuple{$N,$T}}($N)
            @simd for i in eachindex(partials_chunk)
                @inbounds partials_chunk[i] = ntuple(x -> ifelse(x == i, o, z), Val{$N})
            end
            return partials_chunk
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

build_partials_chunk{N,T,C}(::Type{HessianNumber{N,T,C}}) = build_partials_chunk(GradientNumber{N,T,C})
build_partials_chunk{N,T,C}(::Type{TensorNumber{N,T,C}}) = build_partials_chunk(GradientNumber{N,T,C})

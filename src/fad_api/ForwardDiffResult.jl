# This file contains methods to handle the results
# of ForwardDiff calculations (generally either 
# ForwardDiffNumbers or Arrays of ForwardDiffNumbers. 
#
# Note that in the case where the result is an Array, the
# below methods work under the assumption that the result
# contains elements of homogenous type, even if the actual 
# eltype of the array is ambiguous due to poor type 
# inferencing (hence the use of `eltype(first(...))` in 
# a lot of places).

immutable ForwardDiffResult{T}
    data::T
    ForwardDiffResult(data::ForwardDiffNumber) = new(data)
    ForwardDiffResult(data::Array) = new(data)
end

ForwardDiffResult{T}(data::T) = ForwardDiffResult{T}(data)

data(result::ForwardDiffResult) = result.data

##########
# Values #
##########
value!(output, result::ForwardDiffResult) = get_value!(output, data(result))
value(result::ForwardDiffResult) =  get_value(data(result))

function _load_value!(output::Array, arr::Array)
    @simd for i in eachindex(output)
        @inbounds output[i] = get_value(arr[i])
    end
    return output
end

function get_value!(output::Array, arr::Array)
    @assert length(arr) == length(output)
    return _load_value!(output, arr)
end

get_value(arr::Array) = _load_value!(similar(arr, eltype(first(arr))), arr)
get_value(n::ForwardDiffNumber) = value(n)

###############
# Derivatives #
###############
derivative!(output, result::ForwardDiffResult) = get_derivative!(output, data(result))
derivative(result::ForwardDiffResult) =  get_derivative(data(result))

function _load_derivative!(output::Array, arr::Array)
    @simd for i in eachindex(output)
        @inbounds output[i] = get_derivative(arr[i])
    end
    return output
end

function get_derivative!(output::Array, arr::Array)
    @assert length(arr) == length(output)
    return _load_derivative!(output, arr)
end

get_derivative(arr::Array) = _load_derivative!(similar(arr, eltype(first(arr))), arr)
get_derivative(n::ForwardDiffNumber{1}) = first(grad(n))

#############
# Gradients #
#############
gradient!(output, result::ForwardDiffResult) = get_gradient!(output, data(result))
gradient(result::ForwardDiffResult) = get_gradient(data(result))

function _load_gradient!(output, n::ForwardDiffNumber)
    @simd for i in eachindex(output)
        @inbounds output[i] = grad(n, i)
    end
    return output
end

function get_gradient!{N}(output, n::ForwardDiffNumber{N})
    @assert length(output) == N
    return _load_gradient!(output, n)
end

get_gradient{N,T}(n::ForwardDiffNumber{N,T,NTuple{N,T}}) = _load_gradient!(Array(T, N), n)
get_gradient{N,T}(n::ForwardDiffNumber{N,T,Vector{T}}) = grad(n)

#############
# Jacobians #
#############
jacobian!(output, result::ForwardDiffResult) = get_jacobian!(output, data(result))
jacobian(result::ForwardDiffResult) = get_jacobian(data(result))

function _load_jacobian!(output, arr::Array)
    nrows, ncols = size(output)
    for j in 1:ncols
        @simd for i in 1:nrows
            @inbounds output[i,j] = grad(arr[i], j)
        end
    end
    return output
end

function get_jacobian!(output, arr::Array)
    @assert size(output, 1) == length(arr)
    @assert size(output, 2) == npartials(first(arr))
    return _load_jacobian!(output, arr)
end

function get_jacobian(arr::Array)
    init = first(arr)
    output = Array(eltype(init), length(arr), npartials(init))
    return _load_jacobian!(output, arr)
end

############
# Hessians #
############
hessian!(output, result::ForwardDiffResult) = get_hessian!(output, data(result))
hessian(result::ForwardDiffResult) = get_hessian(data(result))

function _load_hessian!{N}(output, n::ForwardDiffNumber{N})
    q = 1
    for i in 1:N
        @simd for j in 1:i
            val = hess(n, q)
            @inbounds output[i, j] = val
            @inbounds output[j, i] = val
            q += 1
        end
    end
    return output
end

function get_hessian!{N}(output, n::ForwardDiffNumber{N})
    @assert size(output) == (N, N)
    return _load_hessian!(output, n)
end

get_hessian{N,T}(n::ForwardDiffNumber{N,T}) = _load_hessian!(Array(T, N, N), n)

############
# Hessians #
############
tensor!(output, result::ForwardDiffResult) = get_tensor!(output, data(result))
tensor(result::ForwardDiffResult) = get_tensor(data(result))

function _load_tensor!{N}(output, n::ForwardDiffNumber{N})
    q = 1
    for i in 1:N
        for j in i:N
            for k in i:j
                @inbounds output[j, k, i] = tens(n, q)
                q += 1
            end
        end
        for j in 1:(i-1)
            for k in 1:j
                @inbounds output[j, k, i] = output[i, j, k]
            end
        end    
        for j in i:N
            for k in 1:(i-1)
                @inbounds output[j, k, i] = output[i, j, k]
            end
        end
        for j in 1:N
            for k in (j+1):N
                @inbounds output[j, k, i] = output[k, j, i]
            end
        end
    end
    return output
end

function get_tensor!{N}(output, n::ForwardDiffNumber{N})
    @assert size(output) == (N, N, N)
    return _load_tensor!(output, n)
end

get_tensor{N,T}(n::ForwardDiffNumber{N,T}) = _load_tensor!(Array(T, N, N, N), n)

abstract Dim{N} # used to configure the input dimension of objective function

######################
# Taking Derivatives #
######################

# Load derivative from ForwardDiffNum #
#-------------------------------------#
function load_derivative!(output::Array, arr::Array)
    @assert length(arr) == length(output)
    @simd for i in eachindex(output)
        @inbounds output[i] = grad(arr[i], 1)
    end
    return output
end

load_derivative(arr::Array) = load_derivative!(similar(arr, eltype(eltype(arr))), arr)
load_derivative(n::ForwardDiffNum{1}) = grad(n, 1)

# Derivative from function/Exposed API methods #
#----------------------------------------------#
derivative!(output::Array, f, x::Number) = load_derivative!(output, f(GradientNum(x, one(x))))
derivative(f, x::Number) = load_derivative(f(GradientNum(x, one(x))))

function derivative(f; mutates=false)
    if mutates
        derivf!(output::Array, x::Number) = derivative!(output, f, x)
        return derivf!
    else
        derivf(x::Number) = derivative(f, x)
        return derivf
    end
end

####################
# Taking Gradients #
####################

# Load gradient from ForwardDiffNum #
#-----------------------------------#
function load_gradient!(output::Vector, n::ForwardDiffNum)
    @assert npartials(n) == length(output) "The output vector must be the same length as the input vector"
    @simd for i in eachindex(output)
        @inbounds output[i] = grad(n, i)
    end
    return output
end

load_gradient{N,T,C}(n::ForwardDiffNum{N,T,C}) = load_gradient!(Array(T, N), n)

# Gradient from function #
#------------------------#
function calc_gradnum!{N,T,C}(f,
                              x::Vector{T},
                              gradvec::Vector{GradientNum{N,T,C}}) 
    Grad = eltype(gradvec)

    @assert length(x) == N "Length of input must be equal to the number of partials components used"
    @assert length(gradvec) == N "The GradientNum vector must be the same length as the input vector"

    pchunk = partials_chunk(Grad)

    @simd for i in eachindex(gradvec)
        @inbounds gradvec[i] = Grad(x[i], pchunk[i])
    end

    return f(gradvec)
end

take_gradient!(output::Vector, f, x::Vector, gradvec::Vector) = load_gradient!(output, calc_gradnum!(f, x, gradvec))
take_gradient!(f, x::Vector, gradvec::Vector) = load_gradient(calc_gradnum!(f, x, gradvec))
take_gradient!{T,D<:Dim}(output::Vector, f, x::Vector{T}, ::Type{D}) = take_gradient!(output, f, x, grad_workvec(D, T))
take_gradient{T,D<:Dim}(f, x::Vector{T}, ::Type{D}) = take_gradient!(f, x, grad_workvec(D, T))

# Exposed API methods #
#---------------------#
gradient!{T}(output::Vector{T}, f, x::Vector) = take_gradient!(output, f, x, Dim{length(x)})::Vector{T}
gradient{T}(f, x::Vector{T}) = take_gradient(f, x, Dim{length(x)})::Vector{T}

function gradient(f; mutates=false)
    if mutates
        gradf!(output::Vector, x::Vector) = gradient!(output, f, x)
        return gradf!
    else
        gradf(x::Vector) = gradient(f, x)
        return gradf
    end
end

####################
# Taking Jacobians #
####################

# Load Jacobian from ForwardDiffNum #
#-----------------------------------#
function load_jacobian!(output, jacvec::Vector)
    # assumes jacvec is actually homogenous,
    # though it may not be well-inferenced.
    N = npartials(first(jacvec))
    for i in 1:length(jacvec), j in 1:N
        output[i,j] = grad(jacvec[i], j)
    end
    return output
end

function load_jacobian(jacvec::Vector)
    # assumes jacvec is actually homogenous,
    # though it may not be well-inferenced.
    F = typeof(first(jacvec))
    return load_jacobian!(Array(eltype(F), length(jacvec), npartials(F)), jacvec)
end

# Jacobian from function #
#------------------------#
function calc_jacnum!{N,T,C}(f,
                             x::Vector{T},
                             gradvec::Vector{GradientNum{N,T,C}})
    Grad = eltype(gradvec)

    @assert length(x) == N "Length of input must be equal to the number of partials components used"
    @assert length(gradvec) == N "The GradientNum vector must be the same length as the input vector"

    pchunk = partials_chunk(Grad)

    @simd for i in eachindex(gradvec)
       @inbounds gradvec[i] = Grad(x[i], pchunk[i])
    end

    return f(gradvec)
end

take_jacobian!(output::Matrix, f, x::Vector, gradvec::Vector) = load_jacobian!(output, calc_jacnum!(f, x, gradvec))
take_jacobian!(f, x::Vector, gradvec::Vector) = load_jacobian(calc_jacnum!(f, x, gradvec))
take_jacobian!{T,D<:Dim}(output::Matrix, f, x::Vector{T}, ::Type{D}) = take_jacobian!(output, f, x, grad_workvec(D, T))
take_jacobian{T,D<:Dim}(f, x::Vector{T}, ::Type{D}) = take_jacobian!(f, x, grad_workvec(D, T))

# Exposed API methods #
#---------------------#
jacobian!{T}(output::Matrix{T}, f, x::Vector) = take_jacobian!(output, f, x, Dim{length(x)})::Matrix{T}
jacobian{T}(f, x::Vector{T}) = take_jacobian(f, x, Dim{length(x)})::Matrix{T}

function jacobian(f; mutates=false)
    if mutates
        jacf!(output::Matrix, x::Vector) = jacobian!(output, f, x)
        return jacf!
    else
        jacf(x::Vector) = jacobian(f, x)
        return jacf
    end
end

###################
# Taking Hessians #
###################

# Load Hessian from ForwardDiffNum #
#----------------------------------#
function load_hessian!{N}(output, n::ForwardDiffNum{N})
    @assert (N, N) == size(output) "The output matrix must have size (length(input), length(input))"
    q = 1
    for i in 1:N
        for j in 1:i
            val = hess(n, q)
            @inbounds output[i, j] = val
            @inbounds output[j, i] = val
            q += 1
        end
    end
    return output
end

load_hessian{N,T}(n::ForwardDiffNum{N,T}) = load_hessian!(Array(T, N, N), n)

# Hessian from function #
#-----------------------#
function calc_hessnum!{N,T,C}(f,
                              x::Vector{T},
                              hessvec::Vector{HessianNum{N,T,C}}) 
    Grad = GradientNum{N,T,C}

    @assert length(x) == N "Length of input must be equal to the number of partials components used"
    @assert length(hessvec) == N "The HessianNum vector must be the same length as the input vector"

    pchunk = partials_chunk(Grad)
    zhess = zero_partials(eltype(hessvec))

    @simd for i in eachindex(hessvec)
        @inbounds hessvec[i] = HessianNum(Grad(x[i], pchunk[i]), zhess)
    end

    return f(hessvec)
end

take_hessian!(output::Matrix, f, x::Vector, hessvec::Vector) = load_hessian!(output, calc_hessnum!(f, x, hessvec))
take_hessian!(f, x::Vector, hessvec::Vector) = load_hessian(calc_hessnum!(f, x, hessvec))
take_hessian!{T,D<:Dim}(output::Matrix, f, x::Vector{T}, ::Type{D}) = take_hessian!(output, f, x, hess_workvec(D, T))
take_hessian{T,D<:Dim}(f, x::Vector{T}, ::Type{D}) = take_hessian!(f, x, hess_workvec(D, T))

# Exposed API methods #
#---------------------#
hessian!{T}(output::Matrix{T}, f, x::Vector) = take_hessian!(output, f, x, Dim{length(x)})::Matrix{T}
hessian{T}(f, x::Vector{T}) = take_hessian(f, x, Dim{length(x)})::Matrix{T}

function hessian(f; mutates=false)
    if mutates
        hessf!(output::Matrix, x::Vector) = hessian!(output, f, x)
        return hessf!
    else
        hessf(x::Vector) = hessian(f, x)
        return hessf
    end
end

##################
# Taking Tensors #
##################

# Load Tensor from ForwardDiffNum #
#---------------------------------#
function load_tensor!{N,T,C}(output, n::ForwardDiffNum{N,T,C})
    @assert (N, N, N) == size(output) "The output array must have size (length(input), length(input), length(input))"
    
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

load_tensor{N,T,C}(n::ForwardDiffNum{N,T,C}) = load_tensor!(Array(T, N, N, N), n)

# Tensor from function #
#----------------------#
function calc_tensnum!{N,T,C}(f,
                              x::Vector{T},
                              tensvec::Vector{TensorNum{N,T,C}}) 
    Grad = GradientNum{N,T,C}

    @assert length(x) == N "Length of input must be equal to the number of partials components used"
    @assert length(tensvec) == N "The TensorNum vector must be the same length as the input"

    pchunk = partials_chunk(Grad)
    zhess = zero_partials(HessianNum{N,T,C})
    ztens = zero_partials(eltype(tensvec))

    @simd for i in eachindex(tensvec)
        @inbounds tensvec[i] = TensorNum(HessianNum(Grad(x[i], pchunk[i]), zhess), ztens)
    end

    return f(tensvec)
end

take_tensor!{S}(output::Array{S,3}, f, x::Vector, tensvec::Vector) = load_tensor!(output, calc_tensnum!(f, x, tensvec))
take_tensor!(f, x::Vector, tensvec::Vector) = load_tensor(calc_tensnum!(f, x, tensvec))
take_tensor!{T,S,D<:Dim}(output::Array{S,3}, f, x::Vector{T}, ::Type{D}) = take_tensor!(output, f, x, tens_workvec(D, T))
take_tensor{T,D<:Dim}(f, x::Vector{T}, ::Type{D}) = take_tensor!(f, x, tens_workvec(D, T))

# Exposed API methods #
#---------------------#
tensor!{T}(output::Array{T,3}, f, x::Vector) = take_tensor!(output, f, x, Dim{length(x)})::Array{T,3}
tensor{T}(f, x::Vector{T}) = take_tensor(f, x, Dim{length(x)})::Array{T,3}

function tensor(f; mutates=false)
    if mutates
        tensf!{T}(output::Array{T,3}, x::Vector) = tensor!(output, f, x)
        return tensf!
    else
        tensf(x::Vector) = tensor(f, x)
        return tensf
    end
end

####################
# Helper Functions #
####################
# Use @generated functions to essentially cache the
# zeros/partial components generated by the input type.
# This caching allows for a higher degree of efficieny 
# when calculating derivatives of f at multiple points, 
# as these values get reused rather than instantiated 
# every time.
#
# This method has the potential to incur a large memory 
# cost (and could even be considered a leak) if a 
# downstream program use many *different* partial components, 
# though I can't think of any use cases in which that would be 
# relevant.

@generated function pick_implementation{N,T}(::Type{Dim{N}}, ::Type{T})
    if N > 10
        return :(Vector{$T})
    else
        return :(NTuple{$N,$T})
    end
end

@generated function grad_workvec{N,T}(::Type{Dim{N}}, ::Type{T})
    result = Vector{GradientNum{N,T,pick_implementation(Dim{N},T)}}(N)
    return :($result)
end

@generated function hess_workvec{N,T}(::Type{Dim{N}}, ::Type{T})
    result = Vector{HessianNum{N,T,pick_implementation(Dim{N},T)}}(N)
    return :($result)
end

@generated function tens_workvec{N,T}(::Type{Dim{N}}, ::Type{T})
    result = Vector{TensorNum{N,T,pick_implementation(Dim{N},T)}}(N)
    return :($result)
end

@generated function zero_partials{N,T}(::Type{GradNumVec{N,T}})
    result = zeros(T, N)
    return :($result)
end

@generated function zero_partials{N,T}(::Type{GradNumTup{N,T}})
    z = zero(T)
    result = ntuple(i->z, Val{N})
    return :($result)
end

@generated function zero_partials{N,T,C}(::Type{HessianNum{N,T,C}})
    result = zeros(T, halfhesslen(N))
    return :($result)
end

@generated function zero_partials{N,T,C}(::Type{TensorNum{N,T,C}})
    result = zeros(T, halftenslen(N))
    return :($result) 
end

@generated function partials_chunk{N,T}(::Type{GradNumVec{N,T}})
    dus_arr = Array(Vector{T}, N)
    @simd for i in eachindex(dus_arr)
        @inbounds dus_arr[i] = setindex!(zeros(T,N), one(T), i)
    end
    return :($dus_arr)
end

@generated function partials_chunk{N,T}(::Type{GradNumTup{N,T}})
    dus_arr = Array(NTuple{N,T}, N)
    z = zero(T)
    o = one(T)
    @simd for i in eachindex(dus_arr)
        @inbounds dus_arr[i] = ntuple(x -> ifelse(x == i, o, z), Val{N})
    end
    return :($dus_arr)
end
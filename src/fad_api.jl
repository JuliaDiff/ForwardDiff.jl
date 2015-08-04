abstract Dim{N} # used to configure the input dimension of objective function

######################
# Taking Derivatives #
######################

# Load derivative from ForwardDiffNum #
#-------------------------------------#
function load_derivative!{F<:ForwardDiffNum}(arr::Array{F}, output::Array)
    @assert length(arr) == length(output)
    @inbounds @simd for i in eachindex(output)
        output[i] = grad(arr[i], 1)
    end
    return output
end

load_derivative{F<:ForwardDiffNum}(arr::Array{F}) = load_derivative!(arr, similar(arr, eltype(F)))
load_derivative(n::ForwardDiffNum{1}) = grad(n, 1)

# Derivative from function/Exposed API methods #
#----------------------------------------------#
derivative!(f, x::Number, output::Array) = load_derivative!(f(GradientNum(x, one(x))), output)
derivative(f, x::Number) = load_derivative(f(GradientNum(x, one(x))))

function derivative(f; mutates=false)
    if mutates
        derivf!(x::Number, output::Array) = derivative!(f, x, output)
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
function load_gradient!(n::ForwardDiffNum, output::Vector)
    @assert npartials(n) == length(output) "The output vector must be the same length as the input vector"
    @inbounds @simd for i in eachindex(output)
        output[i] = grad(n, i)
    end
    return output
end

load_gradient{N,T,C}(n::ForwardDiffNum{N,T,C}) = load_gradient!(n, Array(T, N))

# Gradient from function #
#------------------------#
function calc_gradnum!{N,T,C}(f,
                              x::Vector{T},
                              gradvec::Vector{GradientNum{N,T,C}}) 
    Grad = eltype(gradvec)

    @assert length(x) == N "Length of input must be equal to the number of partials components used"
    @assert length(gradvec) == N "The GradientNum vector must be the same length as the input vector"

    pchunk = partials_chunk(Grad)

    @inbounds @simd for i in 1:N
        gradvec[i] = Grad(x[i], pchunk[i])
    end

    return f(gradvec)
end

take_gradient!(f, x::Vector, output::Vector, gradvec::Vector) = load_gradient!(calc_gradnum!(f, x, gradvec), output)
take_gradient!(f, x::Vector, gradvec::Vector) = load_gradient(calc_gradnum!(f, x, gradvec))
take_gradient!{T,D<:Dim}(f, x::Vector{T}, output::Vector, ::Type{D}) = take_gradient!(f, x, output, grad_workvec(D, T))
take_gradient{T,D<:Dim}(f, x::Vector{T}, ::Type{D}) = take_gradient!(f, x, grad_workvec(D, T))

# Exposed API methods #
#---------------------#
gradient!{T,S}(f, x::Vector{T}, output::Vector{S}) = take_gradient!(f, x, output, Dim{length(x)})::Vector{S}
gradient{T,S}(f, x::Vector{T}, ::Type{S}=T) = take_gradient(f, x, Dim{length(x)})::Vector{S}

function gradient(f; mutates=false)
    if mutates
        gradf!(x::Vector, output::Vector) = gradient!(f, x, output)
        return gradf!
    else
        gradf{T,S}(x::Vector{T}, ::Type{S}=T) = gradient(f, x, S)
        return gradf
    end
end

####################
# Taking Jacobians #
####################

# Load Jacobian from ForwardDiffNum #
#-----------------------------------#
function load_jacobian!{F<:ForwardDiffNum}(v::Vector{F}, output)
    N = npartials(F)
    @assert (length(v), N) == size(output)
    for i in 1:length(v), j in 1:N
        output[i,j] = grad(v[i], j)
    end
    return output
end

load_jacobian{F<:ForwardDiffNum}(v::Vector{F}) = load_jacobian!(v, Array(eltype(F), length(v), npartials(F)))

# Jacobian from function #
#------------------------#
function calc_jacnum!{N,T,C}(f,
                             x::Vector{T},
                             gradvec::Vector{GradientNum{N,T,C}})
    Grad = eltype(gradvec)

    @assert length(x) == N "Length of input must be equal to the number of partials components used"
    @assert length(gradvec) == N "The GradientNum vector must be the same length as the input vector"

    pchunk = partials_chunk(Grad)

    @inbounds @simd for j in 1:N
        gradvec[j] = Grad(x[j], pchunk[j])
    end

    return f(gradvec)
end

take_jacobian!(f, x::Vector, output::Matrix, gradvec::Vector) = load_jacobian!(calc_jacnum!(f, x, gradvec), output)
take_jacobian!(f, x::Vector, gradvec::Vector) = load_jacobian(calc_jacnum!(f, x, gradvec))
take_jacobian!{T,D<:Dim}(f, x::Vector{T}, output::Matrix, ::Type{D}) = take_jacobian!(f, x, output, grad_workvec(D, T))
take_jacobian{T,D<:Dim}(f, x::Vector{T}, ::Type{D}) = take_jacobian!(f, x, grad_workvec(D, T))

# Exposed API methods #
#---------------------#
jacobian!{T}(f, x::Vector{T}, output::Matrix{T}) = take_jacobian!(f, x, output, Dim{length(x)})::Matrix{T}
jacobian{T,S}(f, x::Vector{T}, ::Type{S}=T) = take_jacobian(f, x, Dim{length(x)})::Matrix{S}

function jacobian(f; mutates=false)
    if mutates
        jacf!(x::Vector, output::Matrix) = jacobian!(f, x, output)
        return jacf!
    else
        jacf{T,S}(x::Vector{T}, ::Type{S}=T) = jacobian(f, x, S)
        return jacf
    end
end

###################
# Taking Hessians #
###################

# Load Hessian from ForwardDiffNum #
#----------------------------------#
function load_hessian!{N}(n::ForwardDiffNum{N}, output)
    @assert (N, N) == size(output) "The output matrix must have size (length(input), length(input))"
    q = 1
    for i in 1:N
        for j in 1:i
            val = hess(n, q)
            output[i, j] = val
            output[j, i] = val
            q += 1
        end
    end
    return output
end

load_hessian{N,T}(n::ForwardDiffNum{N,T}) = load_hessian!(n, Array(T, N, N))

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

    @inbounds @simd for i in 1:N
        hessvec[i] = HessianNum(Grad(x[i], pchunk[i]), zhess)
    end

    return f(hessvec)
end

take_hessian!(f, x::Vector, output::Matrix, hessvec::Vector) = load_hessian!(calc_hessnum!(f, x, hessvec), output)
take_hessian!(f, x::Vector, hessvec::Vector) = load_hessian(calc_hessnum!(f, x, hessvec))
take_hessian!{T,D<:Dim}(f, x::Vector{T}, output::Matrix, ::Type{D}) = take_hessian!(f, x, output, hess_workvec(D, T))
take_hessian{T,D<:Dim}(f, x::Vector{T}, ::Type{D}) = take_hessian!(f, x, hess_workvec(D, T))

# Exposed API methods #
#---------------------#
hessian!{T,S}(f, x::Vector{T}, output::Matrix{S}) = take_hessian!(f, x, output, Dim{length(x)})::Matrix{S}
hessian{T,S}(f, x::Vector{T}, ::Type{S}=T) = take_hessian(f, x, Dim{length(x)})::Matrix{S}

function hessian(f; mutates=false)
    if mutates
        hessf!(x::Vector, output::Matrix) = hessian!(f, x, output)
        return hessf!
    else
        hessf{T,S}(x::Vector{T}, ::Type{S}=T) = hessian(f, x, S)
        return hessf
    end
end

##################
# Taking Tensors #
##################

# Load Tensor from ForwardDiffNum #
#---------------------------------#
function load_tensor!{N,T,C}(n::ForwardDiffNum{N,T,C}, output)
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

load_tensor{N,T,C}(n::ForwardDiffNum{N,T,C}) = load_tensor!(n, Array(T, N, N, N))

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

    @inbounds @simd for i in 1:N
        tensvec[i] = TensorNum(HessianNum(Grad(x[i], pchunk[i]), zhess), ztens)
    end

    return f(tensvec)
end

take_tensor!{S}(f, x::Vector, output::Array{S,3}, tensvec::Vector) = load_tensor!(calc_tensnum!(f, x, tensvec), output)
take_tensor!(f, x::Vector, tensvec::Vector) = load_tensor(calc_tensnum!(f, x, tensvec))
take_tensor!{T,S,D<:Dim}(f, x::Vector{T}, output::Array{S,3}, ::Type{D}) = take_tensor!(f, x, output, tens_workvec(D, T))
take_tensor{T,D<:Dim}(f, x::Vector{T}, ::Type{D}) = take_tensor!(f, x, tens_workvec(D, T))

# Exposed API methods #
#---------------------#
tensor!{T,S}(f, x::Vector{T}, output::Array{S,3}) = take_tensor!(f, x, output, Dim{length(x)})::Array{S,3}
tensor{T,S}(f, x::Vector{T}, ::Type{S}=T) = take_tensor(f, x, Dim{length(x)})::Array{S,3}

function tensor(f; mutates=false)
    if mutates
        tensf!{T}(x::Vector, output::Array{T,3}) = tensor!(f, x, output)
        return tensf!
    else
        tensf{T}(x::Vector{T}, ::Type{S}=T) = tensor(f, x, S)
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
    for i in 1:length(dus_arr)
        dus_arr[i] = setindex!(zeros(T,N), one(T), i)
    end
    return :($dus_arr)
end

@generated function partials_chunk{N,T}(::Type{GradNumTup{N,T}})
    dus_arr = Array(NTuple{N,T}, N)
    z = zero(T)
    o = one(T)
    for i in 1:length(dus_arr)
        dus_arr[i] = ntuple(x -> ifelse(x == i, o, z), Val{N})
    end
    return :($dus_arr)
end
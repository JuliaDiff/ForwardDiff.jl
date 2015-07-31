######################
# Taking Derivatives #
######################

# Load derivative from ForwardDiffNum #
#-------------------------------------#
# generalize to arbitrary array dimensions?
function load_derivative!{F<:ForwardDiffNum}(v::Vector{F}, output::Vector)
    @assert length(v) == length(output)
    @inbounds @simd for i in eachindex(result)
        output[i] = grad(v[i], 1)
    end
    return output
end

load_derivative{F<:ForwardDiffNum}(arr::Array{F}) = load_derivative!(arr, similar(arr, eltype(F)))
load_derivative(n::ForwardDiffNum{1}) = grad(n, 1)

# Derivative from function/Exposed API methods #
#----------------------------------------------#
derivative!(f::Function, x::Real, output::Array) = load_derivative!(f(GradientNum(x, one(x))), output)
derivative(f::Function, x::Real) = load_derivative(f(GradientNum(x, one(x))))

function derivative_func(f::Function; mutates=false)
    if mutates
        derivf(x::Real, output::Array) = derivative!(f, x, output)
    else
        derivf(x::Real) = derivative(f, x)
    end
    return derivf
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
function calc_gradnum!{N,T,C}(f::Function,
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

function take_gradient!(f::Function, x::Vector, output::Vector, gradvec::Vector) 
    return load_gradient!(calc_gradnum!(f, x, gradvec), output)
end

function take_gradient!(f::Function, x::Vector, gradvec::Vector)
    return load_gradient(calc_gradnum!(f, x, gradvec))
end

# Exposed API methods #
#---------------------#
function gradient!{N,T}(f::Function, x::Vector{T}, output::Vector, P::Type{Partials{N}})
    return take_gradient!(f, x, output, grad_workvec(P, T))
end

function gradient{N,T}(f::Function, x::Vector{T}, P::Type{Partials{N}})
    return take_gradient!(f, x, grad_workvec(P, T))
end

function gradient_func{N}(f::Function, P::Type{Partials{N}}; mutates=false)
    if mutates
        gradf!{T}(x::Vector{T}, output::Vector) = gradient!(f, x, output, P)
        return gradf!
    else
        gradf{T}(x::Vector{T}) = gradient(f, x, P)
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
function calc_jacnum!{N,T,C}(f::Function,
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

function take_jacobian!(f::Function, x::Vector, output::Matrix, gradvec::Vector)
    return load_jacobian!(calc_jacnum!(f, x, gradvec), output)
end

function take_jacobian!(f::Function, x::Vector, gradvec::Vector)
    return load_jacobian(calc_jacnum!(f, x, gradvec))
end

# Exposed API methods #
#---------------------#
function jacobian!{N,T}(f::Function, x::Vector{T}, output::Matrix, P::Type{Partials{N}})
    return take_jacobian!(f, x, output, grad_workvec(P, T))
end

function jacobian{N,T}(f::Function, x::Vector{T}, P::Type{Partials{N}})
    return take_jacobian!(f, x, grad_workvec(P, T))
end

function jacobian_func{N}(f::Function, P::Type{Partials{N}}; mutates=false)
    if mutates
        jacf!{T}(x::Vector{T}, output::Matrix) = jacobian!(f, x, output, P)
        return jacf!
    else
        jacf{T}(x::Vector{T}) = jacobian(f, x, P)
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
function calc_hessnum!{N,T,C}(f::Function,
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

function take_hessian!(f::Function, x::Vector, output::Matrix, hessvec::Vector)
    return load_hessian!(calc_hessnum!(f, x, hessvec), output)
end

function take_hessian!(f::Function, x::Vector, hessvec::Vector)
    return load_hessian(calc_hessnum!(f, x, hessvec))
end

# Exposed API methods #
#---------------------#
function hessian!{N,T}(f::Function, x::Vector{T}, output::Matrix, P::Type{Partials{N}})
    return take_hessian!(f, x, output, hess_workvec(P, T))
end

function hessian{N,T}(f::Function, x::Vector{T}, P::Type{Partials{N}})
    return take_hessian!(f, x, hess_workvec(P, T))
end

function hessian_func{N}(f::Function, P::Type{Partials{N}}; mutates=false)
    if mutates
        hessf!{T}(x::Vector{T}, output::Matrix) = hessian!(f, x, output, P)
        return hessf!
    else
        hessf{T}(x::Vector{T}) = hessian(f, x, P)
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
                tval = tens(n, q)
                output[i, j, k] = tval
                output[i, k, j] = tval
                output[j, i, k] = tval
                output[j, k, i] = tval
                output[k, i, j] = tval
                output[k, j, i] = tval
                q += 1
            end
        end

        # for j in 1:(i-1)
        #     for k in 1:j
        #         output[i, j, k] = output[, j, k]
        #     end
        # end

        # for j in i:N
        #     for k in 1:(i-1)
        #         output[j, k, i] = output[i, j, k]
        #     end
        # end

        # for j in 1:N
        #     for k in (j+1):N
        #         output[j, k, i] = output[k, j, i]
        #     end
        # end
    end
end

load_tensor{N,T,C}(n::ForwardDiffNum{N,T,C}) = load_tensor!(n, Array(T, N, N, N))

# Tensor from function #
#----------------------#
function calc_tensnum!{N,T,C}(f::Function,
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

function take_tensor!{S}(f::Function, x::Vector, output::Array{S,3}, tensvec::Vector)
    return load_tensor!(calc_tensnum!(f, x, tensvec), output)
end

function take_tensor!(f::Function, x::Vector, tensvec::Vector)
    return load_tensor(calc_tensnum!(f, x, tensvec))
end

# Exposed API methods #
#---------------------#
function tensor!{N,T,S}(f::Function, x::Vector{T}, output::Array{S,3}, P::Type{Partials{N}})
    return take_tensor!(f, x, output, tens_workvec(P, T))
end

function tensor{N,T}(f::Function, x::Vector{T}, P::Type{Partials{N}})
    return take_tensor!(f, x, tens_workvec(P, T))
end

function tensor_func{N}(f::Function, ::Type{Partials{N}}; mutates=false)
    if mutates
        tensf!{T,S}(x::Vector{T}, output::Array{S,3}) = tensor!(f, x, output, P)
        return tensf!
    else
        tensf{T}(x::Vector{T}) = tensor(f, x, P)
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

@generated function grad_workvec{N,T}(::Type{Partials{N}}, ::Type{T})
    result = Vector{GradientNum{N,T,pick_implementation(Partials{N},T)}}(N)
    return :($result)
end

@generated function hess_workvec{N,T}(::Type{Partials{N}}, ::Type{T})
    result = Vector{HessianNum{N,T,pick_implementation(Partials{N},T)}}(N)
    return :($result)
end

@generated function tens_workvec{N,T}(::Type{Partials{N}}, ::Type{T})
    result = Vector{TensorNum{N,T,pick_implementation(Partials{N},T)}}(N)
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
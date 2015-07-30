######################
# Taking Derivatives #
######################

# Derivative from Vector{F<:ForwardDiffNum} #
#-------------------------------------------#
function take_derivative!{F<:ForwardDiffNum}(v::Vector{F}, output::Vector)
    @assert length(v) == length(output)
    @inbounds @simd for i in eachindex(result)
        output[i] = grad(v[i], 1)
    end
    return output
end

# Derivative from function #
#--------------------------#
function derivative!(f::Function, x::Real, output::Vector)
    g = GradientNum(x, one(x))
    v = f(g)
    return take_derivative!(v, output)
end

function derivative(f::Function, x::Real, ::Type{Number})
    g = GradientNum(x, one(x))
    return grad(f(g),1)
end

function derivative(f::Function, x::Real, ::Type{Vector})
    g = GradientNum(x, one(x))
    v = f(g)
    return take_derivative!(v, Vector{typeof(x)}(length(v)))
end

derivative(f::Function, x::Real) = derivative(f, x, Number)

function derivative_func(f::Function, ::Type{Number})
    derivf(x::Real) = derivative(f, x, Number)
    return derivf
end

function derivative_func(f::Function, ::Type{Vector}; mutates=true)
    if mutates
        derivf(x::Real, output::Vector) = derivative!(f, x, output)
    else
        derivf(x::Real) = derivative(f, x, Vector)
    end
    return derivf
end

####################
# Taking Gradients #
####################

# Gradient from ForwardDiffNum #
#------------------------------#
function take_gradient!(n::ForwardDiffNum, output)
    @assert npartials(n) == length(output)
    @inbounds @simd for i in eachindex(output)
        output[i] = grad(n, i)
    end
    return output
end

take_gradient{N,T,C}(n::ForwardDiffNum{N,T,C}) = take_gradient!(n, Array(T, N))

# Gradient from function #
#------------------------#
function take_gradient!{N,T,C,S}(f::Function,
                                 x::Vector{T},
                                 output::Vector{S},
                                 gradvec::Vector{GradientNum{N,T,C}}) 
    Grad = eltype(gradvec)
    ResultGrad = GradientNum{N,S,switch_eltype(C,S)}

    @assert length(x) == N "Length of input must be equal to the number of partials components used"
    @assert N == length(output) "The output vector must be the same length as the input vector"
    @assert N == length(gradvec) "The GradientNum vector must be the same length as the input vector"

    pchunk = partials_chunk(Grad)

    @inbounds @simd for i in 1:N
        gradvec[i] = Grad(x[i], pchunk[i])
    end

    result::ResultGrad = f(gradvec)
    
    take_gradient!(result, output)

    return output
end

function take_gradient!{N,T,C}(f::Function, x::Vector{T}, gradvec::Vector{GradientNum{N,T,C}})
    return take_gradient!(f, x, similar(x), gradvec)
end

function gradient!{N,T}(f::Function, x::Vector{T}, output::Vector, P::Type{Partials{N,T}})
    return take_gradient!(f, x, output, similar(x, GradientNum{N,T,pick_implementation(P)}))
end

function gradient{N,T}(f::Function, x::Vector{T}, P::Type{Partials{N,T}})
    return gradient!(f, x, similar(x), P)
end

function gradient_func{N,T}(f::Function, P::Type{Partials{N,T}}; mutates=true)
    gradvec = Vector{GradientNum{N,T,pick_implementation(P)}}(N)
    if mutates
        gradf!{T}(x::Vector{T}, output::Vector) = take_gradient!(f, x, output, gradvec)
        return gradf!
    else
        gradf{T}(x::Vector{T}) = take_gradient!(f, x, gradvec)
        return gradf
    end
end

####################
# Taking Jacobians #
####################

# Jacobian from Vector{F<:ForwardDiffNum} #
#-----------------------------------------#
function take_jacobian!{F<:ForwardDiffNum}(v::Vector{F}, output)
    N = npartials(F)
    @assert (length(v), N) == size(output)
    for i in 1:length(v), j in 1:N
        output[i,j] = grad(v[i], j)
    end
    return output
end

take_jacobian{F<:ForwardDiffNum}(v::Vector{F}) = take_jacobian!(v, Array(eltype(F), length(v), npartials(F)))

# Jacobian from function #
#------------------------#
function take_jacobian!{N,T,C,S}(f::Function,
                                 x::Vector{T},
                                 output::Matrix{S},
                                 gradvec::Vector{GradientNum{N,T,C}})
    Grad = eltype(gradvec)
    ResultGradVec = Vector{GradientNum{N,S,switch_eltype(C,S)}}

    @assert length(x) == N "Length of input must be equal to the number of partials components used"
    @assert N == size(output, 2) "The number of columns of the output matrix must equal the length of the input vector"
    @assert N == length(gradvec) "The GradientNum vector must be the same length as the input vector"

    pchunk = partials_chunk(Grad)

    @inbounds @simd for j in 1:N
        gradvec[j] = Grad(x[j], pchunk[j])
    end

    result::ResultGradVec = f(gradvec)

    take_jacobian!(result, output)

    return output
end

function take_jacobian!{N,T,C}(f::Function, x::Vector{T}, ylen::Int, gradvec::Vector{GradientNum{N,T,C}})
    return take_jacobian!(f, x, Array(T, ylen, length(x)), gradvec)
end

function jacobian!{N,T}(f::Function, x::Vector{T}, output::Matrix, P::Type{Partials{N,T}})
    return take_jacobian!(f, x, output, similar(x, GradientNum{N,T,pick_implementation(P)}))
end

function jacobian{N,T}(f::Function, x::Vector{T}, P::Type{Partials{N,T}}, ylen::Int)
    return jacobian!(f, x, Array(T, ylen, length(x)), P)
end

function jacobian_func{N,T}(f::Function, P::Type{Partials{N,T}}, ylen::Int; mutates=true)
    gradvec = Vector{GradientNum{N,T,pick_implementation(P)}}(N)
    if mutates
        jacf!{T}(x::Vector{T}, output::Matrix) = take_jacobian!(f, x, output, gradvec)
        return jacf!
    else
        jacf{T}(x::Vector{T}) = take_jacobian!(f, x, ylen, gradvec)
        return jacf
    end
end

###################
# Taking Hessians #
###################

# Hessian from ForwardDiffNum #
#-----------------------------#
function take_hessian!{N}(n::ForwardDiffNum{N}, output)
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

take_hessian{N,T}(n::ForwardDiffNum{N,T}) = take_hessian!(n, Array(T, N, N))

# Hessian from function #
#-----------------------#
function take_hessian!{N,T,C,S}(f::Function,
                                x::Vector{T},
                                output::Matrix{S},
                                hessvec::Vector{HessianNum{N,T,C}}) 
    Grad = GradientNum{N,T,C}
    ResultHessian = HessianNum{N,S,switch_eltype(C,S)}

    @assert length(x) == N "Length of input must be equal to the number of partials components used"
    @assert (N, N) == size(output) "The output matrix must have size (length(input), length(input))"
    @assert N == length(hessvec) "The HessianNum vector must be the same length as the input vector"

    pchunk = partials_chunk(Grad)
    zhess = zero_partials(eltype(hessvec))

    @inbounds @simd for i in 1:N
        hessvec[i] = HessianNum(Grad(x[i], pchunk[i]), zhess)
    end

    result::ResultHessian = f(hessvec)

    take_hessian!(result, output)

    return output
end

function take_hessian!{N,T,C}(f::Function, x::Vector{T}, hessvec::Vector{HessianNum{N,T,C}})
    return take_hessian!(f, x, Array(T, N, N), hessvec)
end

function hessian!{N,T}(f::Function, x::Vector{T}, output::Matrix, P::Type{Partials{N,T}})
    return take_hessian!(f, x, output, similar(x, HessianNum{N,T,pick_implementation(P)}))
end

function hessian{N,T}(f::Function, x::Vector{T}, P::Type{Partials{N,T}})
    return hessian!(f, x, Array(T, N, N), P)
end

function hessian_func{N,T}(f::Function, P::Type{Partials{N,T}}; mutates=true)
    hessvec = Vector{HessianNum{N,T,pick_implementation(P)}}(N)
    if mutates
        hessf!{T}(x::Vector{T}, output::Matrix) = take_hessian!(f, x, output, hessvec)
        return hessf!
    else
        hessf{T}(x::Vector{T}) = take_hessian!(f, x, hessvec)
        return hessf
    end
end

##################
# Taking Tensors #
##################

# Tensor from ForwardDiffNum #
#----------------------------#
function take_tensor!{N,T,C}(n::ForwardDiffNum{N,T,C}, output)
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

take_tensor{N,T,C}(n::ForwardDiffNum{N,T,C}) = take_tensor!(n, Array(T, N, N, N))

# Tensor from function #
#----------------------#
function take_tensor!{N,T,C,S}(f::Function,
                          x::Vector{T},
                          output::Array{S,3},
                          tensvec::Vector{TensorNum{N,T,C}}) 
    xlen = 
    Grad = GradientNum{N,T,C}
    ResultTensor = TensorNum{N,S,switch_eltype(C,S)}

    @assert length(x) == N "Length of input must be equal to the number of partials components used"
    @assert (N, N, N) == size(output) "The output array must have size (length(input), length(input), length(input))"
    @assert N == length(tensvec) "The TensorNum vector must be the same length as the input"

    pchunk = partials_chunk(Grad)
    zhess = zero_partials(HessianNum{N,T,C})
    ztens = zero_partials(eltype(tensvec))

    @inbounds @simd for i in 1:N
        tensvec[i] = TensorNum(HessianNum(Grad(x[i], pchunk[i]), zhess), ztens)
    end

    result::ResultTensor = f(tensvec)

    take_tensor!(result, output)
    
    return output
end


function take_tensor!{N,T,C}(f::Function, x::Vector{T}, tensvec::Vector{TensorNum{N,T,C}})
    return take_tensor!(f, x, Array(T, N, N, N), tensvec)
end

function tensor!{N,T,S}(f::Function, x::Vector{T}, output::Array{S,3}, P::Type{Partials{N,T}})
    return take_tensor!(f, x, output, similar(x, TensorNum{N,T,pick_implementation(P)}))
end

function tensor{N,T}(f::Function, x::Vector{T}, P::Type{Partials{N,T}})
    return tensor!(f, x, Array(T, N, N, N), P)
end

function tensor_func{N,T}(f::Function, ::Type{Partials{N,T}}; mutates=true)
    tensvec = Vector{TensorNum{N,T,pick_implementation(P)}}(N)
    if mutates
        tensf!{T,S}(x::Vector{T}, output::Array{S,3}) = take_tensor!(f, x, output, tensvec)
        return tensf!
    else
        tensf{T}(x::Vector{T}) = take_tensor!(f, x, tensvec)
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
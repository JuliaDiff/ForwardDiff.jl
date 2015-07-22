abstract ForwardDiffNum{N,T<:Real,C} <: Number
# Subtypes F<:ForwardDiffNum should define:
#    npartials(::Type{F}) --> N from ForwardDiffNum{N,T,C}
#    eltype(::Type{F}) --> T from ForwardDiffNum{N,T,C}
#    grad(n::F) --> the dual number stored by n
#    hess(n::F) --> a vector corresponding to the lower 
#                   triangular half of the symmetric 
#                   Hessian (including the diagonal)
#    tens(n::F) --> a vector of corresponding to lower 
#                   tetrahedral half of the symmetric 
#                   Tensor (including the diagonal)
#    isconstant(n::F) --> returns true if all partials stored by n are zero
#
#...as well as:
#    ==(a::F, b::F)
#    isequal(a::F, b::F)
#    hash(n::F)
#    read(io::IO, ::Type{F})
#    write(io::IO, n::F)
##############################
# Utility/Accessor Functions #
##############################
halfhesslen(n) = div(n*(n+1),2) # correct length(hess(::ForwardDiffNum))
halftenslen(n) = div(n*(n+1)*(n+2),6) # correct length(tens(::ForwardDiffNum))

switch_eltype{T,S}(::Type{Vector{T}}, ::Type{S}) = Vector{S}
switch_eltype{N,T,S}(::Type{NTuple{N,T}}, ::Type{S}) = NTuple{N,S}

grad(n::ForwardDiffNum, i) = partials(grad(n), i)
hess(n::ForwardDiffNum, i) = hess(n)[i]
tens(n::ForwardDiffNum, i) = tens(n)[i]

value(n::ForwardDiffNum) = value(grad(n))

eps(n::ForwardDiffNum) = eps(value(n))
eps{F<:ForwardDiffNum}(::Type{F}) = eps(eltype(F))

isnan(n::ForwardDiffNum) = isnan(value(n))
isfinite(n::ForwardDiffNum) = isfinite(value(n))

npartials(n::ForwardDiffNum) = npartials(grad(n))
eltype(n::ForwardDiffNum) = eltype(grad(n))

npartials{N,T,C}(::Type{ForwardDiffNum{N,T,C}}) = N
eltype{N,T,C}(::Type{ForwardDiffNum{N,T,C}}) = T

==(n::ForwardDiffNum, x::Real) = isconstant(n) && (value(n) == x)
==(x::Real, n::ForwardDiffNum) = ==(n, x)

isequal(n::ForwardDiffNum, x::Real) = isconstant(n) && isequal(value(n), x)
isequal(x::Real, n::ForwardDiffNum) = isequal(n, x)

isless(a::ForwardDiffNum, b::ForwardDiffNum) = value(a) < value(b)
isless(x::Real, n::ForwardDiffNum) = x < value(n)
isless(n::ForwardDiffNum, x::Real) = value(n) < x

copy(n::ForwardDiffNum) = n # assumes all types of ForwardDiffNums are immutable

##################
# Math Functions #
##################
conj(n::ForwardDiffNum) = n
transpose(n::ForwardDiffNum) = n
ctranspose(n::ForwardDiffNum) = n

##################################
# Gradient from a ForwardDiffNum #
##################################
function gradient!(n::ForwardDiffNum, output)
    @assert npartials(n) == length(output)
    for i in eachindex(output)
        output[i] = grad(n, i)
    end
    return output
end

gradient{N,T,C}(n::ForwardDiffNum{N,T,C}) = gradient!(n, Array(T, N))

#################################
# Hessian from a ForwardDiffNum #
#################################
function hessian!{N}(n::ForwardDiffNum{N}, output)
    @assert (N, N) == size(output)
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

hessian{N,T}(n::ForwardDiffNum{N,T}) = hessian!(n, Array(T, N, N))

#########################################
# Jacobian from a ForwardDiffNum Vector #
#########################################
function jacobian!{F<:ForwardDiffNum}(v::Vector{F}, output)
    N = npartials(F)
    @assert (length(v), N) == size(output)
    for i in 1:length(v), j in 1:N
        output[i,j] = grad(v[i], j)
    end
    return output
end

jacobian{F<:ForwardDiffNum}(v::Vector{F}) = jacobian!(v, Array(eltype(F), length(v), npartials(F)))

################################
# Tensor from a ForwardDiffNum #
################################
function tensor!{N,T,C}(n::ForwardDiffNum{N,T,C}, output)
    @assert (N, N, N) == size(output)
    q = 1
    for k in 1:N
        for i in k:N
            for j in k:i 
                val = tens(n, q)
                output[i, j, k] = val
                output[j, i, k] = val
                output[j, k, i] = val
                q += 1
            end
        end
    end
    return output
end

tensor{N,T,C}(n::ForwardDiffNum{N,T,C}) = tensor!(n, Array(T, N, N, N))
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

grad(adn::ForwardDiffNum, i) = partials(grad(adn), i)
hess(adn::ForwardDiffNum, i) = hess(adn)[i]
tens(adn::ForwardDiffNum, i) = tens(adn)[i]

value(adn::ForwardDiffNum) = value(grad(adn))

eps(adn::ForwardDiffNum) = eps(value(adn))
eps{F<:ForwardDiffNum}(::Type{F}) = eps(eltype(F))

isnan(adn::ForwardDiffNum) = isnan(value(adn))
isfinite(adn::ForwardDiffNum) = isfinite(value(adn))

npartials(adn::ForwardDiffNum) = npartials(grad(adn))
eltype(adn::ForwardDiffNum) = eltype(grad(adn))

npartials{N,T,C}(::Type{ForwardDiffNum{N,T,C}}) = N
eltype{N,T,C}(::Type{ForwardDiffNum{N,T,C}}) = T

==(adn::ForwardDiffNum, x::Real) = isconstant(adn) && (value(adn) == x)
==(x::Real, adn::ForwardDiffNum) = ==(adn, x)

isequal(adn::ForwardDiffNum, x::Real) = isconstant(adn) && isequal(value(adn), x)
isequal(x::Real, adn::ForwardDiffNum) = isequal(adn, x)

isless(a::ForwardDiffNum, b::ForwardDiffNum) = value(a) < value(b)
isless(x::Real, adn::ForwardDiffNum) = x < value(adn)
isless(adn::ForwardDiffNum, x::Real) = value(adn) < x

copy(adn::ForwardDiffNum) = adn # assumes all types of ForwardDiffNums are immutable

##################
# Math Functions #
##################
conj(adn::ForwardDiffNum) = adn
transpose(adn::ForwardDiffNum) = adn
ctranspose(adn::ForwardDiffNum) = adn

##################################
# Gradient from a ForwardDiffNum #
##################################
function gradient!(adn::ForwardDiffNum, output)
    @assert npartials(adn) == length(output)
    for i in eachindex(output)
        output[i] = grad(adn, i)
    end
    return output
end

gradient{N,T,C}(adn::ForwardDiffNum{N,T,C}) = gradient!(adn, Array(T, N))

#################################
# Hessian from a ForwardDiffNum #
#################################
function hessian!{N}(adn::ForwardDiffNum{N}, output)
    @assert (N, N) == size(output)
    q = 1
    for i in 1:N
        for j in 1:i
            val = hess(adn, q)
            output[i, j] = val
            output[j, i] = val
            q += 1
        end
    end
    return output
end

hessian{N,T}(adn::ForwardDiffNum{N,T}) = hessian!(adn, Array(T, N, N))

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
function tensor!{N,T,C}(adn::ForwardDiffNum{N,T,C}, output)
    @assert (N, N, N) == size(output)
    q = 1
    for k in 1:N
        for i in k:N
            for j in k:i 
                val = tens(adn, q)
                output[i, j, k] = val
                output[j, i, k] = val
                output[j, k, i] = val
                q += 1
            end
        end
    end
    return output
end

tensor{N,T,C}(adn::ForwardDiffNum{N,T,C}) = tensor!(adn, Array(T, N, N, N))
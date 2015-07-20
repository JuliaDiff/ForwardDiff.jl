abstract AutoDiffNum{N,T<:Real,C} <: Number
# Subtypes F<:AutoDiffNum should define:
#    npartials(::Type{F}) --> N from AutoDiffNum{N,T,C}
#    eltype(::Type{F}) --> T from AutoDiffNum{N,T,C}
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
halfhesslen(n) = div(n*(n+1),2) # correct length(hess(::AutoDiffNum))
halftenslen(n) = div(n*(n+1)*(n+2),6) # correct length(tens(::AutoDiffNum))

switch_eltype{T,S}(::Type{Vector{T}}, ::Type{S}) = Vector{S}
switch_eltype{N,T,S}(::Type{NTuple{N,T}}, ::Type{S}) = NTuple{N,S}

grad(adn::AutoDiffNum, i) = partials(grad(adn), i)
hess(adn::AutoDiffNum, i) = hess(adn)[i]
tens(adn::AutoDiffNum, i) = tens(adn)[i]

value(adn::AutoDiffNum) = value(grad(adn))

eps(adn::AutoDiffNum) = eps(value(adn))
eps{F<:AutoDiffNum}(::Type{F}) = eps(eltype(F))

isnan(adn::AutoDiffNum) = isnan(value(adn))
isfinite(adn::AutoDiffNum) = isfinite(value(adn))

npartials(adn::AutoDiffNum) = npartials(grad(adn))
eltype(adn::AutoDiffNum) = eltype(grad(adn))

npartials{N,T,C}(::Type{AutoDiffNum{N,T,C}}) = N
eltype{N,T,C}(::Type{AutoDiffNum{N,T,C}}) = T

==(adn::AutoDiffNum, x::Real) = isconstant(adn) && (value(adn) == x)
==(x::Real, adn::AutoDiffNum) = ==(adn, x)

isequal(adn::AutoDiffNum, x::Real) = isconstant(adn) && isequal(value(adn), x)
isequal(x::Real, adn::AutoDiffNum) = isequal(adn, x)

isless(a::AutoDiffNum, b::AutoDiffNum) = value(a) < value(b)
isless(x::Real, adn::AutoDiffNum) = x < value(adn)
isless(adn::AutoDiffNum, x::Real) = value(adn) < x

copy(adn::AutoDiffNum) = adn # assumes all types of AutoDiffNums are immutable

##################
# Math Functions #
##################
conj(adn::AutoDiffNum) = adn
transpose(adn::AutoDiffNum) = adn
ctranspose(adn::AutoDiffNum) = adn

#############################
# Gradient from a AutoDiffNum #
#############################
function gradient!(adn::AutoDiffNum, output)
    @assert npartials(adn) == length(output)
    for i in eachindex(output)
        output[i] = grad(adn, i)
    end
    return output
end

gradient{N,T,C}(adn::AutoDiffNum{N,T,C}) = gradient!(adn, Array(T, N))

############################
# Hessian from a AutoDiffNum #
############################
function hessian!{N}(adn::AutoDiffNum{N}, output)
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

hessian{N,T}(adn::AutoDiffNum{N,T}) = hessian!(adn, Array(T, N, N))

####################################
# Jacobian from a AutoDiffNum Vector #
####################################
function jacobian!{F<:AutoDiffNum}(v::Vector{F}, output)
    N = npartials(F)
    @assert (length(v), N) == size(output)
    for i in 1:length(v), j in 1:N
        output[i,j] = grad(v[i], j)
    end
    return output
end

jacobian{F<:AutoDiffNum}(v::Vector{F}) = jacobian!(v, Array(eltype(F), length(v), npartials(F)))

###########################
# Tensor from a AutoDiffNum #
###########################
function tensor!{N,T,C}(adn::AutoDiffNum{N,T,C}, output)
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

tensor{N,T,C}(adn::AutoDiffNum{N,T,C}) = tensor!(adn, Array(T, N, N, N))
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

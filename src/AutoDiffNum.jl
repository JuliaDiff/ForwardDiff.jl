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

grad(fad::AutoDiffNum, i) = partials(grad(fad), i)
hess(fad::AutoDiffNum, i) = hess(fad)[i]
tens(fad::AutoDiffNum, i) = tens(fad)[i]

value(fad::AutoDiffNum) = value(grad(fad))

eps(fad::AutoDiffNum) = eps(value(fad))
eps{F<:AutoDiffNum}(::Type{F}) = eps(eltype(F))

isnan(fad::AutoDiffNum) = isnan(value(fad))
isfinite(fad::AutoDiffNum) = isfinite(value(fad))

npartials(fad::AutoDiffNum) = npartials(grad(fad))
eltype(fad::AutoDiffNum) = eltype(grad(fad))

npartials{N,T,C}(::Type{AutoDiffNum{N,T,C}}) = N
eltype{N,T,C}(::Type{AutoDiffNum{N,T,C}}) = T

==(fad::AutoDiffNum, x::Real) = isconstant(fad) && (value(fad) == x)
==(x::Real, fad::AutoDiffNum) = ==(fad, x)

isequal(fad::AutoDiffNum, x::Real) = isconstant(fad) && isequal(value(fad), x)
isequal(x::Real, fad::AutoDiffNum) = isequal(fad, x)

isless(a::AutoDiffNum, b::AutoDiffNum) = value(a) < value(b)
isless(x::Real, fad::AutoDiffNum) = fad < value(g)
isless(fad::AutoDiffNum, x::Real) = value(g) < fad

copy(fad::AutoDiffNum) = fad # assumes all types of AutoDiffNums are immutable

##################
# Math Functions #
##################
conj(fad::AutoDiffNum) = fad
transpose(fad::AutoDiffNum) = fad
ctranspose(fad::AutoDiffNum) = fad

#############################
# Gradient from a AutoDiffNum #
#############################
function gradient!(fad::AutoDiffNum, output)
    @assert npartials(fad) == length(output)
    for i in eachindex(output)
        output[i] = grad(fad, i)
    end
    return output
end

gradient{N,T,C}(fad::AutoDiffNum{N,T,C}) = gradient!(fad, Array(T, N))

############################
# Hessian from a AutoDiffNum #
############################
function hessian!{N}(fad::AutoDiffNum{N}, output)
    @assert (N, N) == size(output)
    q = 1
    for i in 1:N
        for j in 1:i
            val = hess(fad, q)
            output[i, j] = val
            output[j, i] = val
            q += 1
        end
    end
    return output
end

hessian{N,T}(fad::AutoDiffNum{N,T}) = hessian!(fad, Array(T, N, N))

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
function tensor!{N,T,C}(fad::AutoDiffNum{N,T,C}, output)
    @assert (N, N, N) == size(output)
    q = 1
    for k in 1:N
        for i in k:N
            for j in k:i 
                val = tens(fad, q)
                output[i, j, k] = val
                output[j, i, k] = val
                output[j, k, i] = val
                q += 1
            end
        end
    end
    return output
end

tensor{N,T,C}(fad::AutoDiffNum{N,T,C}) = tensor!(fad, Array(T, N, N, N))
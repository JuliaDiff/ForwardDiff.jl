abstract FADNumber{N,T<:Real,D<:NDual} <: Number
# Subtypes F<:FADNumber should define:
#    neps(::Type{F}) --> N from FADNumber{N,T,D}
#    eltype(::Type{F}) --> T from FADNumber{N,T,D}
#    grad(n::F) --> the dual number stored by n
#    hess(n::F) --> a vector corresponding to the lower 
#                   triangular half of the symmetric 
#                   Hessian (including the diagonal)
#    tens(n::F) --> a vector of corresponding to lower 
#                   tetrahedral half of the symmetric 
#                   Tensor (including the diagonal)
#    isconstant(n::F) --> returns true if all partials stored by n are zero

##############################
# Utility/Accessor Functions #
##############################
halfhesslen(n) = div(n*(n+1),2) # correct length(hess(::FADNumber))
halftenslen(n) = div(n*(n+1)*(n+2),6) # correct length(tens(::FADNumber))

grad(fad::FADNumber, i) = epsilon(grad(fad), i)
hess(fad::FADNumber, i) = hess(fad)[i]
tens(fad::FADNumber, i) = tens(fad)[i]

value(fad::FADNumber) = value(grad(fad))

neps(fad::FADNumber) = neps(grad(fad))
eltype(fad::FADNumber) = eltype(grad(fad))

neps{N,T,D}(::Type{FADNumber{N,T,D}}) = N
eltype{N,T,D}(::Type{FADNumber{N,T,D}}) = T

==(fad::FADNumber, x::Real) = isconstant(fad) && (value(fad) == x)
==(x::Real, fad::FADNumber) = ==(fad, x)

copy(fad::FADNumber) = fad # assumes all types of FADNumbers are immutable

#############################
# Gradient from a FADNumber #
#############################
function gradient!(fad::FADNumber, output)
    @assert neps(fad) == length(output)
    for i in eachindex(output)
        output[i] = grad(fad, i)
    end
    return output
end

gradient{N,T,D}(fad::FADNumber{N,T,D}) = gradient!(fad, Array(T, N))

############################
# Hessian from a FADNumber #
############################
function hessian!{N}(fad::FADNumber{N}, output)
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

hessian{N,T}(fad::FADNumber{N,T}) = hessian!(fad, Array(T, N, N))

####################################
# Jacobian from a FADNumber Vector #
####################################
function jacobian!{F<:FADNumber}(v::Vector{F}, output)
    N = neps(F)
    @assert (length(v), N) == size(output)
    for i in 1:length(v), j in 1:N
        output[i,j] = grad(v[i], j)
    end
    return output
end

jacobian{F<:FADNumber}(v::Vector{F}) = jacobian!(v, Array(eltype(F), length(v), neps(F)))

###########################
# Tensor from a FADNumber #
###########################
function tensor!{N,T,D}(fad::FADNumber{N,T,D}, output)
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

tensor{N,T,D}(fad::FADNumber{N,T,D}) = tensor!(fad, Array(T, N, N, N))
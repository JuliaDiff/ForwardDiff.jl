abstract ForwardDiffNumber{N,T<:Number,C} <: Number

# Subtypes F<:ForwardDiffNumber should define:
#    npartials(::Type{F}) --> N from ForwardDiffNumber{N,T,C}
#    eltype(::Type{F}) --> T from ForwardDiffNumber{N,T,C}
#    value(n::F) --> the value of n
#    grad(n::F) --> a container corresponding to all first order partials
#    hess(n::F) --> a container corresponding to the lower
#                   triangular half of the symmetric
#                   Hessian (including the diagonal)
#    tens(n::F) --> a container of corresponding to lower
#                   tetrahedral half of the symmetric
#                   Tensor (including the diagonal)
#    isconstant(n::F) --> returns true if all partials stored by n are zero
#
#...as well as:
#    ==(a::F, b::F)
#    isequal(a::F, b::F)
#    zero(::Type{F})
#    one(::Type{F})
#    rand(::Type{F})
#    hash(n::F)
#    read(io::IO, ::Type{F})
#    write(io::IO, n::F)
#    conversion/promotion rules

##############################
# Utility/Accessor Functions #
##############################
@inline promote_typeof(a, b) = promote_type(typeof(a), typeof(b))
@inline promote_typeof(a, b, c) = promote_type(promote_typeof(a, b), typeof(c))
@inline promote_typeof(a, b, c, d) = promote_type(promote_typeof(a, b, c), typeof(d))

@inline halfhesslen(n) = div(n*(n+1),2) # correct length(hess(::ForwardDiffNumber))
@inline halftenslen(n) = div(n*(n+1)*(n+2),6) # correct length(tens(::ForwardDiffNumber))

@inline switch_eltype{T,S}(::Type{Vector{T}}, ::Type{S}) = Vector{S}
@inline switch_eltype{N,T,S}(::Type{NTuple{N,T}}, ::Type{S}) = NTuple{N,S}

@inline grad(n::ForwardDiffNumber, i) = grad(n)[i]
@inline hess(n::ForwardDiffNumber, i) = hess(n)[i]
@inline tens(n::ForwardDiffNumber, i) = tens(n)[i]

@inline npartials{N}(::ForwardDiffNumber{N}) = N
@inline eltype{N,T}(::ForwardDiffNumber{N,T}) = T

@inline npartials{N,T,C}(::Type{ForwardDiffNumber{N,T,C}}) = N
@inline eltype{N,T,C}(::Type{ForwardDiffNumber{N,T,C}}) = T

zero(n::ForwardDiffNumber) = zero(typeof(n))
one(n::ForwardDiffNumber) = one(typeof(n))

==(n::ForwardDiffNumber, x::Real) = isconstant(n) && (value(n) == x)
==(x::Real, n::ForwardDiffNumber) = ==(n, x)

isequal(n::ForwardDiffNumber, x::Real) = isconstant(n) && isequal(value(n), x)
isequal(x::Real, n::ForwardDiffNumber) = isequal(n, x)

isless(a::ForwardDiffNumber, b::ForwardDiffNumber) = value(a) < value(b)
isless(x::Real, n::ForwardDiffNumber) = x < value(n)
isless(n::ForwardDiffNumber, x::Real) = value(n) < x

copy(n::ForwardDiffNumber) = n # assumes all types of ForwardDiffNumbers are immutable

eps(n::ForwardDiffNumber) = eps(value(n))
eps{F<:ForwardDiffNumber}(::Type{F}) = eps(eltype(F))

isnan(n::ForwardDiffNumber) = isnan(value(n))
isfinite(n::ForwardDiffNumber) = isfinite(value(n))
isinf(n::ForwardDiffNumber) = isinf(value(n))
isreal(n::ForwardDiffNumber) = isconstant(n)

##################
# Math Functions #
##################
conj(n::ForwardDiffNumber) = n
transpose(n::ForwardDiffNumber) = n
ctranspose(n::ForwardDiffNumber) = n
float(n::ForwardDiffNumber) = n

# NaNMath Helper Functions #
#--------------------------#
function to_nanmath(x::Expr)
    if x.head == :call
        funsym = Expr(:.,:NaNMath,Base.Meta.quot(x.args[1]))
        return Expr(:call,funsym,[to_nanmath(z) for z in x.args[2:end]]...)
    else
        return Expr(:call,[to_nanmath(z) for z in x.args]...)
    end
end

to_nanmath(x) = x

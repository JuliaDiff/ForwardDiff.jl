abstract ForwardDiffNumber{N,T<:Real,C} <: Real

# Subtypes F<:ForwardDiffNumber should define:
#    npartials(::Type{F}) --> N from ForwardDiffNumber{N,T,C}
#    eltype(::Type{F}) --> T from ForwardDiffNumber{N,T,C}
#    containtype(::Type{F}) --> C from ForwardDiffNumber{N,T,C}
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
@inline halfhesslen(n) = div(n*(n+1),2) # correct length(hess(::ForwardDiffNumber))
@inline halftenslen(n) = div(n*(n+1)*(n+2),6) # correct length(tens(::ForwardDiffNumber))

@inline grad(n::ForwardDiffNumber, i) = grad(n)[i]
@inline hess(n::ForwardDiffNumber, i) = hess(n)[i]
@inline tens(n::ForwardDiffNumber, i) = tens(n)[i]

@inline npartials{N}(::ForwardDiffNumber{N}) = N
@inline eltype{N,T}(::ForwardDiffNumber{N,T}) = T
@inline containtype{N,T,C}(::ForwardDiffNumber{N,T,C}) = C

@inline npartials{N,T,C}(::Type{ForwardDiffNumber{N,T,C}}) = N
@inline eltype{N,T,C}(::Type{ForwardDiffNumber{N,T,C}}) = T
@inline containtype{N,T,C}(::Type{ForwardDiffNumber{N,T,C}}) = C

@defambiguous ==(a::ForwardDiffNumber, b::ForwardDiffNumber)
@defambiguous isequal(a::ForwardDiffNumber, b::ForwardDiffNumber)

for T in (Base.Irrational, AbstractFloat, Real)
    @eval begin
        ==(n::ForwardDiffNumber, x::$T) = isconstant(n) && (value(n) == x)
        ==(x::$T, n::ForwardDiffNumber) = ==(n, x)

        isequal(n::ForwardDiffNumber, x::$T) = isconstant(n) && isequal(value(n), x)
        isequal(x::$T, n::ForwardDiffNumber) = isequal(n, x)
    end
end

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

########################
# Conversion/Promotion #
########################
zero(n::ForwardDiffNumber) = zero(typeof(n))
one(n::ForwardDiffNumber) = one(typeof(n))

function float(n::ForwardDiffNumber)
    T = promote_type(eltype(n), Float16)
    return convert(switch_eltype(typeof(n), T), n)
end

##################
# Math Functions #
##################
conj(n::ForwardDiffNumber) = n
transpose(n::ForwardDiffNumber) = n
ctranspose(n::ForwardDiffNumber) = n

@inline abs(n::ForwardDiffNumber) = signbit(value(n)) ? -n : n
@inline abs2(n::ForwardDiffNumber) = n*n

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

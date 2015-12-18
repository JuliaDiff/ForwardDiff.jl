###############################
# Abstract Types/Type Aliases #
###############################

@eval typealias ExternalReal Union{$(subtypes(Real)...)}
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

####################
# Type Definitions #
####################

immutable GradientNumber{N,T,C} <: ForwardDiffNumber{N,T,C}
    value::T
    partials::Partials{T,C}
end

immutable HessianNumber{N,T,C} <: ForwardDiffNumber{N,T,C}
    gradnum::GradientNumber{N,T,C}
    hess::Vector{T}
    function HessianNumber(gradnum, hess)
        @assert length(hess) == halfhesslen(N)
        return new(gradnum, hess)
    end
end

immutable TensorNumber{N,T,C} <: ForwardDiffNumber{N,T,C}
    hessnum::HessianNumber{N,T,C}
    tens::Vector{T}
    function TensorNumber(hessnum, tens)
        @assert length(tens) == halftenslen(N)
        return new(hessnum, tens)
    end
end


##################################
# Ambiguous Function Definitions #
##################################
ambiguous_error{A,B}(f, a::A, b::B) = error("""Oops - $f(::$A, ::$B) should never have been called.
                                               It was defined to resolve ambiguity, and was supposed to
                                               fall back to a more specific method defined elsewhere.
                                               Please report this bug to ForwardDiff.jl's issue tracker.""")

ambiguous_binary_funcs = [:(==), :isequal, :isless, :<, :+, :-, :*, :/, :^, :atan2, :calc_atan2]
fdnum_ambiguous_binary_funcs = [:(==), :isequal, :isless, :<]
fdnum_types = [:GradientNumber, :HessianNumber, :TensorNumber]

for f in ambiguous_binary_funcs
    if f in fdnum_ambiguous_binary_funcs
        @eval $f(a::ForwardDiffNumber, b::ForwardDiffNumber) = ambiguous_error($f, a, b)
    end
    for A in fdnum_types, B in fdnum_types
        @eval $f(a::$A, b::$B) = ambiguous_error($f, a, b)
    end
end

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

for T in fdnum_types
    @eval isless(a::$T, b::$T) = value(a) < value(b)
    @eval <(a::$T, b::$T) = isless(a, b)
end

for T in (Base.Irrational, AbstractFloat, Real)
    @eval begin
        ==(n::ForwardDiffNumber, x::$T) = isconstant(n) && (value(n) == x)
        ==(x::$T, n::ForwardDiffNumber) = ==(n, x)

        isequal(n::ForwardDiffNumber, x::$T) = isconstant(n) && isequal(value(n), x)
        isequal(x::$T, n::ForwardDiffNumber) = isequal(n, x)

        isless(x::$T, n::ForwardDiffNumber) = x < value(n)
        isless(n::ForwardDiffNumber, x::$T) = value(n) < x

        <(x::$T, n::ForwardDiffNumber) = isless(x, n)
        <(n::ForwardDiffNumber, x::$T) = isless(n, x)
    end
end

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

convert(::Type{Integer}, n::ForwardDiffNumber) = isconstant(n) ? Integer(value(n)) : throw(InexactError())
convert(::Type{Bool}, n::ForwardDiffNumber) = isconstant(n) ? Bool(value(n)) : throw(InexactError())
convert{T<:ExternalReal}(::Type{T}, n::ForwardDiffNumber) = isconstant(n) ? T(value(n)) : throw(InexactError())

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

# Overloading of `promote_array_type` #
#-------------------------------------#
Base.promote_array_type{S<:ForwardDiff.ForwardDiffNumber, A<:AbstractFloat}(F, ::Type{S}, ::Type{A}) = S

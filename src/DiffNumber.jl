##############
# DiffNumber #
##############

immutable DiffNumber{N,T<:Real} <: Real
    value::T
    partials::Partials{N,T}
end

################
# Constructors #
################

DiffNumber{N,T}(value::T, partials::Partials{N,T}) = DiffNumber{N,T}(value, partials)

function DiffNumber{N,A,B}(value::A, partials::Partials{N,B})
    T = promote_type(A, B)
    return DiffNumber(convert(T, value), convert(Partials{N,T}, partials))
end

DiffNumber(value::Real, partials::Tuple) = DiffNumber(value, Partials(partials))
DiffNumber(value::Real, partials::Tuple{}) = DiffNumber(value, Partials{0,typeof(value)}(partials))
DiffNumber(value::Real, partials::Real...) = DiffNumber(value, partials)

##############################
# Utility/Accessor Functions #
##############################

@inline value(x::Real) = x
@inline value(n::DiffNumber) = n.value

@inline partials(x::Real) = Partials{0,typeof(x)}(tuple())
@inline partials(n::DiffNumber) = n.partials
@inline partials(n::DiffNumber, i) = n.partials[i]
@inline partials(n::DiffNumber, i, j) = partials(n, i).partials[j]
@inline partials(n::DiffNumber, i, j, k) = partials(n, i, j).partials[k]

@inline npartials{N}(::DiffNumber{N}) = N
@inline npartials{N,T}(::Type{DiffNumber{N,T}}) = N

@inline degree{T}(::T) = degree(T)
@inline degree{T}(::Type{T}) = 0
degree{N,T}(::Type{DiffNumber{N,T}}) = 1 + degree(T)

@inline numtype{N,T}(::DiffNumber{N,T}) = T
@inline numtype{N,T}(::Type{DiffNumber{N,T}}) = T

#####################
# Generic Functions #
#####################

macro ambiguous(ex)
    def = ex.head == :macrocall ? ex.args[2] : ex
    f = def.args[1].args[1].args[1]
    return quote
        $(f)(a::DiffNumber, b::DiffNumber) = error("npartials($(typeof(a))) != npartials($(typeof(b)))")
        $(esc(ex))
    end
end

Base.copy(n::DiffNumber) = n

Base.eps(n::DiffNumber) = eps(value(n))
Base.eps{F<:DiffNumber}(::Type{F}) = eps(numtype(F))

Base.floor{T<:Real}(::Type{T}, n::DiffNumber) = floor(T, value(n))
Base.ceil{T<:Real}(::Type{T}, n::DiffNumber) = ceil(T, value(n))
Base.trunc{T<:Real}(::Type{T}, n::DiffNumber) = trunc(T, value(n))
Base.round{T<:Real}(::Type{T}, n::DiffNumber) = round(T, value(n))

const FDNUM_HASH = hash(DiffNumber)

Base.hash(n::DiffNumber) = hash(value(n))
Base.hash(n::DiffNumber, hsh::UInt64) = hash(value(n), hsh)

function Base.read{N,T}(io::IO, ::Type{DiffNumber{N,T}})
    value = read(io, T)
    partials = read(io, Partials{N,T})
    return DiffNumber{N,T}(value, partials)
end

function Base.write(io::IO, n::DiffNumber)
    write(io, value(n))
    write(io, partials(n))
end

@inline Base.zero(n::DiffNumber) = zero(typeof(n))
@inline Base.zero{N,T}(::Type{DiffNumber{N,T}}) = DiffNumber(zero(T), zero(Partials{N,T}))

@inline Base.one(n::DiffNumber) = one(typeof(n))
@inline Base.one{N,T}(::Type{DiffNumber{N,T}}) = DiffNumber(one(T), zero(Partials{N,T}))

@inline Base.rand(n::DiffNumber) = rand(typeof(n))
@inline Base.rand{N,T}(::Type{DiffNumber{N,T}}) = DiffNumber(rand(T), zero(Partials{N,T}))
@inline Base.rand(rng::AbstractRNG, n::DiffNumber) = rand(rng, typeof(n))
@inline Base.rand{N,T}(rng::AbstractRNG, ::Type{DiffNumber{N,T}}) = DiffNumber(rand(rng, T), zero(Partials{N,T}))

# Predicates #
#------------#

isconstant(n::DiffNumber) = iszero(partials(n))

@ambiguous Base.isequal{N}(a::DiffNumber{N}, b::DiffNumber{N}) = isequal(value(a), value(b))
@ambiguous Base.(:(==)){N}(a::DiffNumber{N}, b::DiffNumber{N}) = value(a) == value(b)
@ambiguous Base.isless{N}(a::DiffNumber{N}, b::DiffNumber{N}) = value(a) < value(b)
@ambiguous Base.(:<){N}(a::DiffNumber{N}, b::DiffNumber{N}) = isless(a, b)
@ambiguous Base.(:(<=)){N}(a::DiffNumber{N}, b::DiffNumber{N}) = <=(value(a), value(b))

for T in (AbstractFloat, Irrational, Real)
    Base.isequal(n::DiffNumber, x::T) = isequal(value(n), x)
    Base.isequal(x::T, n::DiffNumber) = isequal(n, x)

    Base.(:(==))(n::DiffNumber, x::T) = (value(n) == x)
    Base.(:(==))(x::T, n::DiffNumber) = ==(n, x)

    Base.isless(n::DiffNumber, x::T) = value(n) < x
    Base.isless(x::T, n::DiffNumber) = x < value(n)

    Base.(:<)(n::DiffNumber, x::T) = isless(n, x)
    Base.(:<)(x::T, n::DiffNumber) = isless(x, n)

    Base.(:(<=))(n::DiffNumber, x::T) = <=(value(n), x)
    Base.(:(<=))(x::T, n::DiffNumber) = <=(x, value(n))
end

Base.isnan(n::DiffNumber) = isnan(value(n))
Base.isfinite(n::DiffNumber) = isfinite(value(n))
Base.isinf(n::DiffNumber) = isinf(value(n))
Base.isreal(n::DiffNumber) = isreal(value(n))
Base.isinteger(n::DiffNumber) = isinteger(value(n))
Base.iseven(n::DiffNumber) = iseven(value(n))
Base.isodd(n::DiffNumber) = isodd(value(n))

########################
# Promotion/Conversion #
########################

Base.promote_rule{N1,N2,A<:Real,B<:Real}(D1::Type{DiffNumber{N1,A}}, D2::Type{DiffNumber{N2,B}}) = error("can't promote $(D1) and $(D2)")
Base.promote_rule{N,A<:Real,B<:Real}(::Type{DiffNumber{N,A}}, ::Type{DiffNumber{N,B}}) = DiffNumber{N,promote_type(A, B)}
Base.promote_rule{N,T<:Real}(::Type{DiffNumber{N,T}}, ::Type{BigFloat}) = DiffNumber{N,promote_type(T, BigFloat)}
Base.promote_rule{N,T<:Real}(::Type{BigFloat}, ::Type{DiffNumber{N,T}}) = DiffNumber{N,promote_type(BigFloat, T)}
Base.promote_rule{N,T<:Real}(::Type{DiffNumber{N,T}}, ::Type{Bool}) = DiffNumber{N,promote_type(T, Bool)}
Base.promote_rule{N,T<:Real}(::Type{Bool}, ::Type{DiffNumber{N,T}}) = DiffNumber{N,promote_type(Bool, T)}
Base.promote_rule{N,T<:Real,s}(::Type{DiffNumber{N,T}}, ::Type{Irrational{s}}) = DiffNumber{N,promote_type(T, Irrational{s})}
Base.promote_rule{N,s,T<:Real}(::Type{Irrational{s}}, ::Type{DiffNumber{N,T}}) = DiffNumber{N,promote_type(Irrational{s}, T)}
Base.promote_rule{N,A<:Real,B<:Real}(::Type{DiffNumber{N,A}}, ::Type{B}) = DiffNumber{N,promote_type(A, B)}
Base.promote_rule{N,A<:Real,B<:Real}(::Type{A}, ::Type{DiffNumber{N,B}}) = DiffNumber{N,promote_type(A, B)}

Base.convert(::Type{DiffNumber}, n::DiffNumber) = n
Base.convert{N1,N2,T<:Real}(D::Type{DiffNumber{N1,T}}, n::DiffNumber{N2}) = error("can't convert $(typeof(n)) to $(D)")
Base.convert{N,T<:Real}(::Type{DiffNumber{N,T}}, n::DiffNumber{N}) = DiffNumber(convert(T, value(n)), convert(Partials{N,T}, partials(n)))
Base.convert{N,T<:Real}(::Type{DiffNumber{N,T}}, n::DiffNumber{N,T}) = n
Base.convert{N,T<:Real}(::Type{DiffNumber{N,T}}, x::Real) = DiffNumber(convert(T, x), zero(Partials{N,T}))
Base.convert(::Type{DiffNumber}, x::Real) = DiffNumber(x)

Base.promote_array_type{D<:DiffNumber, A<:AbstractFloat}(F, ::Type{D}, ::Type{A}) = D

Base.float{N,T}(n::DiffNumber{N,T}) = DiffNumber{N,promote_type(T, Float16)}(n)

########
# Math #
########

# Addition/Subtraction #
#----------------------#

@ambiguous @inline Base.(:+){N}(n1::DiffNumber{N}, n2::DiffNumber{N}) = DiffNumber(value(n1) + value(n2), partials(n1) + partials(n2))
@inline Base.(:+)(n::DiffNumber, x::Real) = DiffNumber(value(n) + x, partials(n))
@inline Base.(:+)(x::Real, n::DiffNumber) = n + x

@ambiguous @inline Base.(:-){N}(n1::DiffNumber{N}, n2::DiffNumber{N}) = DiffNumber(value(n1) - value(n2), partials(n1) - partials(n2))
@inline Base.(:-)(n::DiffNumber, x::Real) = DiffNumber(value(n) - x, partials(n))
@inline Base.(:-)(x::Real, n::DiffNumber) = DiffNumber(x - value(n), -(partials(n)))
@inline Base.(:-)(n::DiffNumber) = DiffNumber(-(value(n)), -(partials(n)))

# Multiplication #
#----------------#

@inline Base.(:*)(n::DiffNumber, x::Bool) = x ? n : (signbit(value(n))==0 ? zero(n) : -zero(n))
@inline Base.(:*)(x::Bool, n::DiffNumber) = n * x

@ambiguous @inline function Base.(:*){N}(n1::DiffNumber{N}, n2::DiffNumber{N})
    v1, v2 = value(n1), value(n2)
    return DiffNumber(v1 * v2, _mul_partials(partials(n1), partials(n2), v2, v1))
end

@inline Base.(:*)(n::DiffNumber, x::Real) = DiffNumber(value(n) * x, partials(n) * x)
@inline Base.(:*)(x::Real, n::DiffNumber) = n * x

# Division #
#----------#

@ambiguous @inline function Base.(:/){N}(n1::DiffNumber{N}, n2::DiffNumber{N})
    v1, v2 = value(n1), value(n2)
    return DiffNumber(v1 / v2, _div_partials(partials(n1), partials(n2), v1, v2))
end

@inline function Base.(:/)(x::Real, n::DiffNumber)
    v = value(n)
    divv = x / v
    return DiffNumber(divv, -(divv / v) * partials(n))
end

@inline Base.(:/)(n::DiffNumber, x::Real) = DiffNumber(value(n) / x, partials(n) / x)

# Exponentiation #
#----------------#

for f in (:(Base.(:^)), :(NaNMath.pow))

    @eval begin
        @ambiguous @inline function ($f){N}(n1::DiffNumber{N}, n2::DiffNumber{N})
            if iszero(partials(n2))
                return $(f)(n1, value(n2))
            else
                v1, v2 = value(n1), value(n2)
                expv = ($f)(v1, v2)
                powval = v2 * ($f)(v1, v2 - 1)
                logval = expv * log(v1)
                new_partials = _mul_partials(partials(n1), partials(n2), powval, logval)
                return DiffNumber(expv, new_partials)
            end
        end

        @inline ($f)(::Base.Irrational{:e}, n::DiffNumber) = exp(n)
    end

    for T in (:Integer, :Rational, :Real)
        @eval begin
            @inline function ($f)(n::DiffNumber, x::$(T))
                v = value(n)
                expv = ($f)(v, x)
                deriv = x * ($f)(v, x - 1)
                return DiffNumber(expv, deriv * partials(n))
            end

            @inline function ($f)(x::$(T), n::DiffNumber)
                v = value(n)
                expv = ($f)(x, v)
                deriv = expv*log(x)
                return DiffNumber(expv, deriv * partials(n))
            end
        end
    end
end

# Unary Math Functions #
#--------------------- #

function to_nanmath(x::Expr)
    if x.head == :call
        funsym = Expr(:.,:NaNMath,Base.Meta.quot(x.args[1]))
        return Expr(:call,funsym,[to_nanmath(z) for z in x.args[2:end]]...)
    else
        return Expr(:call,[to_nanmath(z) for z in x.args]...)
    end
end

to_nanmath(x) = x

@inline Base.conj(n::DiffNumber) = n
@inline Base.transpose(n::DiffNumber) = n
@inline Base.ctranspose(n::DiffNumber) = n
@inline Base.abs(n::DiffNumber) = signbit(value(n)) ? -n : n

for fsym in AUTO_DEFINED_UNARY_FUNCS
    v = :v
    deriv = Calculus.differentiate(:($(fsym)($v)), v)

    @eval begin
        @inline function Base.$(fsym)(n::DiffNumber)
            $(v) = value(n)
            return DiffNumber($(fsym)($v), $(deriv) * partials(n))
        end
    end

    # extend corresponding NaNMath methods
    if fsym in NANMATH_FUNCS
        nan_deriv = to_nanmath(deriv)
        @eval begin
            @inline function NaNMath.$(fsym)(n::DiffNumber)
                v = value(n)
                return DiffNumber(NaNMath.$(fsym)($v), $(nan_deriv) * partials(n))
            end
        end
    end
end

#################
# Special Cases #
#################

# Manually Optimized Functions #
#------------------------------#

@inline function Base.exp{N}(n::DiffNumber{N})
    expv = exp(value(n))
    return DiffNumber(expv, expv * partials(n))
end

@inline function Base.sqrt{N}(n::DiffNumber{N})
    sqrtv = sqrt(value(n))
    deriv = 0.5 / sqrtv
    return DiffNumber(sqrtv, deriv * partials(n))
end

# Other Functions #
#-----------------#

@inline function calc_atan2(y, x)
    z = y/x
    v= value(z)
    atan2v = atan2(value(y), value(x))
    deriv = inv(one(v) + v*v)
    return DiffNumber(atan2v, deriv * partials(z))
end

@ambiguous @inline Base.atan2{N}(y::DiffNumber{N}, x::DiffNumber{N}) = calc_atan2(y, x)
@inline Base.atan2(y::Real, x::DiffNumber) = calc_atan2(y, x)
@inline Base.atan2(y::DiffNumber, x::Real) = calc_atan2(y, x)

###################
# Pretty Printing #
###################

function Base.show{N,T}(io::IO, n::DiffNumber{N,T})
    d = degree(n)
    print(io, "(", value(n))
    for i in 1:N
        p = partials(n, i)
        signbit(p) ? print(io, " - $(abs(p))*ϵ[$d,$i]") : print(io, " + $(p)*ϵ[$d,$i]")
    end
    print(io, ")")
end

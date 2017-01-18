const ExternalReal = Union{subtypes(Real)...}

########
# Dual #
########

immutable Dual{N,T<:Real} <: Real
    value::T
    partials::Partials{N,T}
end

################
# Constructors #
################

Dual{N,T}(value::T, partials::Partials{N,T}) = Dual{N,T}(value, partials)

function Dual{N,A,B}(value::A, partials::Partials{N,B})
    T = promote_type(A, B)
    return Dual(convert(T, value), convert(Partials{N,T}, partials))
end

Dual(value::Real, partials::Tuple) = Dual(value, Partials(partials))
Dual(value::Real, partials::Tuple{}) = Dual(value, Partials{0,typeof(value)}(partials))
Dual(value::Real, partials::Real...) = Dual(value, partials)

##############################
# Utility/Accessor Functions #
##############################

@inline value(x::Real) = x
@inline value(n::Dual) = n.value

@inline partials(x::Real) = Partials{0,typeof(x)}(tuple())
@inline partials(n::Dual) = n.partials
@inline partials(x::Real, i...) = zero(x)
@inline partials(n::Dual, i) = n.partials[i]
@inline partials(n::Dual, i, j) = partials(n, i).partials[j]
@inline partials(n::Dual, i, j, k...) = partials(partials(n, i, j), k...)

@inline npartials{N}(::Dual{N}) = N
@inline npartials{N,T}(::Type{Dual{N,T}}) = N

@inline degree{T}(::T) = degree(T)
@inline degree{T}(::Type{T}) = 0
degree{N,T}(::Type{Dual{N,T}}) = 1 + degree(T)

@inline valtype{T}(::T) = T
@inline valtype{T}(::Type{T}) = T
@inline valtype{N,T}(::Dual{N,T}) = T
@inline valtype{N,T}(::Type{Dual{N,T}}) = T

#####################
# Generic Functions #
#####################

macro ambiguous(ex)
    def = ex.head == :macrocall ? ex.args[2] : ex
    sig = def.args[1]
    body = def.args[2]
    f = isa(sig.args[1], Expr) && sig.args[1].head == :curly ? sig.args[1].args[1] : sig.args[1]
    a, b = sig.args[2].args[1], sig.args[3].args[1]
    Ta, Tb = sig.args[2].args[2], sig.args[3].args[2]
    if isa(a, Symbol) && isa(b, Symbol) && isa(Ta, Symbol) && isa(Tb, Symbol)
        if Ta == :Real && Tb == :Dual
            return quote
                @inline $(f){A<:ExternalReal,B<:Dual}(a::Dual{0,A}, b::Dual{0,B}) = Dual($(f)(value(a), value(b)))
                @inline $(f){M,A<:ExternalReal,B<:Dual}(a::Dual{0,A}, b::Dual{M,B}) = $(f)(value(a), b)
                @inline $(f){N,A<:ExternalReal,B<:Dual}(a::Dual{N,A}, b::Dual{0,B}) = $(f)(a, value(b))
                @inline $(f){N,A<:ExternalReal,B<:Dual}($(a)::Dual{N,A}, $(b)::Dual{N,B}) = $(body)
                @inline $(f){N,M,A<:ExternalReal,B<:Dual}($(a)::Dual{N,A}, $(b)::Dual{M,B}) = $(body)
                $(esc(ex))
            end
        elseif Ta == :Dual && Tb == :Real
            return quote
                @inline $(f){A<:Dual,B<:ExternalReal}(a::Dual{0,A}, b::Dual{0,B}) = Dual($(f)(value(a), value(b)))
                @inline $(f){M,A<:Dual,B<:ExternalReal}(a::Dual{0,A}, b::Dual{M,B}) = $(f)(value(a), b)
                @inline $(f){N,A<:Dual,B<:ExternalReal}(a::Dual{N,A}, b::Dual{0,B}) = $(f)(a, value(b))
                @inline $(f){N,A<:Dual,B<:ExternalReal}($(a)::Dual{N,A}, $(b)::Dual{N,B}) = $(body)
                @inline $(f){N,M,A<:Dual,B<:ExternalReal}($(a)::Dual{N,A}, $(b)::Dual{M,B}) = $(body)
                $(esc(ex))
            end
        else
            return esc(ex)
        end
    end
    return quote
        @inline $(f){N,M,A<:Real,B<:Real}(a::Dual{N,A}, b::Dual{M,B}) = error("npartials($(typeof(a))) != npartials($(typeof(b)))")
        if !(in($f, (isequal, ==, isless, <, <=, <)))
            @inline $(f){A<:Real,B<:Real}(a::Dual{0,A}, b::Dual{0,B}) = Dual($(f)(value(a), value(b)))
            @inline $(f){M,A<:Real,B<:Real}(a::Dual{0,A}, b::Dual{M,B}) = $(f)(value(a), b)
            @inline $(f){N,A<:Real,B<:Real}(a::Dual{N,A}, b::Dual{0,B}) = $(f)(a, value(b))
        end
        $(esc(ex))
    end
end

Base.copy(n::Dual) = n

Base.eps(n::Dual) = eps(value(n))
Base.eps{D<:Dual}(::Type{D}) = eps(valtype(D))

Base.rtoldefault{N, T <: Real}(::Type{Dual{N,T}}) = Base.rtoldefault(T)

Base.floor{T<:Real}(::Type{T}, n::Dual) = floor(T, value(n))
Base.floor(n::Dual) = floor(value(n))

Base.ceil{T<:Real}(::Type{T}, n::Dual) = ceil(T, value(n))
Base.ceil(n::Dual) = ceil(value(n))

Base.trunc{T<:Real}(::Type{T}, n::Dual) = trunc(T, value(n))
Base.trunc(n::Dual) = trunc(value(n))

Base.round{T<:Real}(::Type{T}, n::Dual) = round(T, value(n))
Base.round(n::Dual) = round(value(n))

Base.hash(n::Dual) = hash(value(n))
Base.hash(n::Dual, hsh::UInt64) = hash(value(n), hsh)

function Base.read{N,T}(io::IO, ::Type{Dual{N,T}})
    value = read(io, T)
    partials = read(io, Partials{N,T})
    return Dual{N,T}(value, partials)
end

function Base.write(io::IO, n::Dual)
    write(io, value(n))
    write(io, partials(n))
end

@inline Base.zero(n::Dual) = zero(typeof(n))
@inline Base.zero{N,T}(::Type{Dual{N,T}}) = Dual(zero(T), zero(Partials{N,T}))

@inline Base.one(n::Dual) = one(typeof(n))
@inline Base.one{N,T}(::Type{Dual{N,T}}) = Dual(one(T), zero(Partials{N,T}))

@inline Base.rand(n::Dual) = rand(typeof(n))
@inline Base.rand{N,T}(::Type{Dual{N,T}}) = Dual(rand(T), zero(Partials{N,T}))
@inline Base.rand(rng::AbstractRNG, n::Dual) = rand(rng, typeof(n))
@inline Base.rand{N,T}(rng::AbstractRNG, ::Type{Dual{N,T}}) = Dual(rand(rng, T), zero(Partials{N,T}))

# Predicates #
#------------#

isconstant(n::Dual) = iszero(partials(n))

@ambiguous Base.isequal{N}(a::Dual{N}, b::Dual{N}) = isequal(value(a), value(b))
@ambiguous @compat(Base.:(==)){N}(a::Dual{N}, b::Dual{N}) = value(a) == value(b)
@ambiguous Base.isless{N}(a::Dual{N}, b::Dual{N}) = value(a) < value(b)
@ambiguous @compat(Base.:<){N}(a::Dual{N}, b::Dual{N}) = isless(a, b)
@ambiguous @compat(Base.:(<=)){N}(a::Dual{N}, b::Dual{N}) = <=(value(a), value(b))

for T in (AbstractFloat, Irrational, Real)
    Base.isequal(n::Dual, x::T) = isequal(value(n), x)
    Base.isequal(x::T, n::Dual) = isequal(n, x)

    @compat(Base.:(==))(n::Dual, x::T) = (value(n) == x)
    @compat(Base.:(==))(x::T, n::Dual) = ==(n, x)

    Base.isless(n::Dual, x::T) = value(n) < x
    Base.isless(x::T, n::Dual) = x < value(n)

    @compat(Base.:<)(n::Dual, x::T) = isless(n, x)
    @compat(Base.:<)(x::T, n::Dual) = isless(x, n)

    @compat(Base.:(<=))(n::Dual, x::T) = <=(value(n), x)
    @compat(Base.:(<=))(x::T, n::Dual) = <=(x, value(n))
end

Base.isnan(n::Dual) = isnan(value(n))
Base.isfinite(n::Dual) = isfinite(value(n))
Base.isinf(n::Dual) = isinf(value(n))
Base.isreal(n::Dual) = isreal(value(n))
Base.isinteger(n::Dual) = isinteger(value(n))
Base.iseven(n::Dual) = iseven(value(n))
Base.isodd(n::Dual) = isodd(value(n))

########################
# Promotion/Conversion #
########################

Base.promote_rule{N1,N2,A<:Real,B<:Real}(D1::Type{Dual{N1,A}}, D2::Type{Dual{N2,B}}) = error("can't promote $(D1) and $(D2)")
Base.promote_rule{N,A<:Real,B<:Real}(::Type{Dual{N,A}}, ::Type{Dual{N,B}}) = Dual{N,promote_type(A, B)}
Base.promote_rule{N,T<:Real}(::Type{Dual{N,T}}, ::Type{BigFloat}) = Dual{N,promote_type(T, BigFloat)}
Base.promote_rule{N,T<:Real}(::Type{BigFloat}, ::Type{Dual{N,T}}) = Dual{N,promote_type(BigFloat, T)}
Base.promote_rule{N,T<:Real}(::Type{Dual{N,T}}, ::Type{Bool}) = Dual{N,promote_type(T, Bool)}
Base.promote_rule{N,T<:Real}(::Type{Bool}, ::Type{Dual{N,T}}) = Dual{N,promote_type(Bool, T)}
Base.promote_rule{N,T<:Real,s}(::Type{Dual{N,T}}, ::Type{Irrational{s}}) = Dual{N,promote_type(T, Irrational{s})}
Base.promote_rule{N,s,T<:Real}(::Type{Irrational{s}}, ::Type{Dual{N,T}}) = Dual{N,promote_type(Irrational{s}, T)}
Base.promote_rule{N,A<:Real,B<:Real}(::Type{Dual{N,A}}, ::Type{B}) = Dual{N,promote_type(A, B)}
Base.promote_rule{N,A<:Real,B<:Real}(::Type{A}, ::Type{Dual{N,B}}) = Dual{N,promote_type(A, B)}

Base.convert(::Type{Dual}, n::Dual) = n
Base.convert{N,T<:Real}(::Type{Dual{N,T}}, n::Dual{N}) = Dual(convert(T, value(n)), convert(Partials{N,T}, partials(n)))
Base.convert{D<:Dual}(::Type{D}, n::D) = n
Base.convert{N,T<:Real}(::Type{Dual{N,T}}, x::Real) = Dual(convert(T, x), zero(Partials{N,T}))
Base.convert(::Type{Dual}, x::Real) = Dual(x)

Base.promote_array_type{D<:Dual, A<:AbstractFloat}(F, ::Type{D}, ::Type{A}) = promote_type(D, A)
Base.promote_array_type{D<:Dual, A<:AbstractFloat, P}(F, ::Type{D}, ::Type{A}, ::Type{P}) = P
Base.promote_array_type{A<:AbstractFloat, D<:Dual}(F, ::Type{A}, ::Type{D}) = promote_type(D, A)
Base.promote_array_type{A<:AbstractFloat, D<:Dual, P}(F, ::Type{A}, ::Type{D}, ::Type{P}) = P

Base.float{N,T}(n::Dual{N,T}) = Dual{N,promote_type(T, Float16)}(n)

########
# Math #
########

# Addition/Subtraction #
#----------------------#

@ambiguous @inline @compat(Base.:+){N}(n1::Dual{N}, n2::Dual{N}) = Dual(value(n1) + value(n2), partials(n1) + partials(n2))
@ambiguous @inline @compat(Base.:+)(n::Dual, x::Real) = Dual(value(n) + x, partials(n))
@ambiguous @inline @compat(Base.:+)(x::Real, n::Dual) = n + x

@ambiguous @inline @compat(Base.:-){N}(n1::Dual{N}, n2::Dual{N}) = Dual(value(n1) - value(n2), partials(n1) - partials(n2))
@ambiguous @inline @compat(Base.:-)(n::Dual, x::Real) = Dual(value(n) - x, partials(n))
@ambiguous @inline @compat(Base.:-)(x::Real, n::Dual) = Dual(x - value(n), -(partials(n)))
@inline @compat(Base.:-)(n::Dual) = Dual(-(value(n)), -(partials(n)))

# Multiplication #
#----------------#

@inline @compat(Base.:*)(n::Dual, x::Bool) = x ? n : (signbit(value(n))==0 ? zero(n) : -zero(n))
@inline @compat(Base.:*)(x::Bool, n::Dual) = n * x

@ambiguous @inline function @compat(Base.:*){N}(n1::Dual{N}, n2::Dual{N})
    v1, v2 = value(n1), value(n2)
    return Dual(v1 * v2, _mul_partials(partials(n1), partials(n2), v2, v1))
end

@ambiguous @inline @compat(Base.:*)(n::Dual, x::Real) = Dual(value(n) * x, partials(n) * x)
@ambiguous @inline @compat(Base.:*)(x::Real, n::Dual) = n * x

# Division #
#----------#

@ambiguous @inline function @compat(Base.:/){N}(n1::Dual{N}, n2::Dual{N})
    v1, v2 = value(n1), value(n2)
    return Dual(v1 / v2, _div_partials(partials(n1), partials(n2), v1, v2))
end

@ambiguous @inline function @compat(Base.:/)(x::Real, n::Dual)
    v = value(n)
    divv = x / v
    return Dual(divv, -(divv / v) * partials(n))
end

@ambiguous @inline @compat(Base.:/)(n::Dual, x::Real) = Dual(value(n) / x, partials(n) / x)

# Exponentiation #
#----------------#

for f in (macroexpand(:(@compat(Base.:^))), :(NaNMath.pow))
    @eval begin
        @ambiguous @inline function ($f){N}(n1::Dual{N}, n2::Dual{N})
            v1, v2 = value(n1), value(n2)
            expv = ($f)(v1, v2)
            powval = v2 * ($f)(v1, v2 - 1)
            logval = isconstant(n2) ? one(expv) : expv * log(v1)
            new_partials = _mul_partials(partials(n1), partials(n2), powval, logval)
            return Dual(expv, new_partials)
        end

        @inline ($f)(::Base.Irrational{:e}, n::Dual) = exp(n)
    end

    for T in (:Integer, :Rational, :Real)
        @eval begin
            @ambiguous @inline function ($f)(n::Dual, x::$(T))
                v = value(n)
                expv = ($f)(v, x)
                deriv = x * ($f)(v, x - 1)
                return Dual(expv, deriv * partials(n))
            end

            @ambiguous @inline function ($f)(x::$(T), n::Dual)
                v = value(n)
                expv = ($f)(x, v)
                deriv = expv*log(x)
                return Dual(expv, deriv * partials(n))
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

@inline Base.conj(n::Dual) = n
@inline Base.transpose(n::Dual) = n
@inline Base.ctranspose(n::Dual) = n
@inline Base.abs(n::Dual) = signbit(value(n)) ? -n : n

for fsym in AUTO_DEFINED_UNARY_FUNCS
    v = :v
    deriv = Calculus.differentiate(:($(fsym)($v)), v)

    # exp and sqrt are manually defined below
    if !(in(fsym, (:exp, :sqrt)))
        @eval begin
            @inline function Base.$(fsym)(n::Dual)
                $(v) = value(n)
                return Dual($(fsym)($v), $(deriv) * partials(n))
            end
        end
    end

    # extend corresponding NaNMath methods
    if fsym in NANMATH_FUNCS
        nan_deriv = to_nanmath(deriv)
        @eval begin
            @inline function NaNMath.$(fsym)(n::Dual)
                v = value(n)
                return Dual(NaNMath.$(fsym)($v), $(nan_deriv) * partials(n))
            end
        end
    end
end

#################
# Special Cases #
#################

# Manually Optimized Functions #
#------------------------------#

@inline function Base.exp{N}(n::Dual{N})
    expv = exp(value(n))
    return Dual(expv, expv * partials(n))
end

@inline function Base.sqrt{N}(n::Dual{N})
    sqrtv = sqrt(value(n))
    deriv = inv(sqrtv + sqrtv)
    return Dual(sqrtv, deriv * partials(n))
end

@inline function calc_hypot(x, y)
    vx = value(x)
    vy = value(y)
    h = hypot(vx, vy)
    return Dual(h, (vx/h) * partials(x) + (vy/h) * partials(y))
end

@inline function calc_hypot(x, y, z)
    vx = value(x)
    vy = value(y)
    vz = value(z)
    h = hypot(vx, vy, vz)
    return Dual(h, (vx/h) * partials(x) + (vy/h) * partials(y) + (vz/h) * partials(z))
end

@ambiguous @inline Base.hypot{N}(x::Dual{N}, y::Dual{N}) = calc_hypot(x, y)
@ambiguous @inline Base.hypot(x::Dual, y::Real) = calc_hypot(x, y)
@ambiguous @inline Base.hypot(x::Real, y::Dual) = calc_hypot(x, y)

@inline Base.hypot(x::Dual, y::Dual, z::Dual) = calc_hypot(x, y, z)

@inline Base.hypot(x::Real, y::Dual, z::Dual) = calc_hypot(x, y, z)
@inline Base.hypot(x::Dual, y::Real, z::Dual) = calc_hypot(x, y, z)
@inline Base.hypot(x::Dual, y::Dual, z::Real) = calc_hypot(x, y, z)

@inline Base.hypot(x::Dual, y::Real, z::Real) = calc_hypot(x, y, z)
@inline Base.hypot(x::Real, y::Dual, z::Real) = calc_hypot(x, y, z)
@inline Base.hypot(x::Real, y::Real, z::Dual) = calc_hypot(x, y, z)

@inline sincos(n) = (sin(n), cos(n))

@inline function sincos(n::Dual)
    sn, cn = sincos(value(n))
    return (Dual(sn, cn * partials(n)), Dual(cn, -sn * partials(n)))
end

# Other Functions #
#-----------------#

@inline function calc_atan2(y, x)
    z = y / x
    v = value(z)
    atan2v = atan2(value(y), value(x))
    deriv = inv(one(v) + v*v)
    return Dual(atan2v, deriv * partials(z))
end

@ambiguous @inline Base.atan2{N}(y::Dual{N}, x::Dual{N}) = calc_atan2(y, x)
@ambiguous @inline Base.atan2(y::Real, x::Dual) = calc_atan2(y, x)
@ambiguous @inline Base.atan2(y::Dual, x::Real) = calc_atan2(y, x)

###################
# Pretty Printing #
###################

function Base.show{N}(io::IO, n::Dual{N})
    print(io, "Dual(", value(n))
    for i in 1:N
        print(io, ",", partials(n, i))
    end
    print(io, ")")
end

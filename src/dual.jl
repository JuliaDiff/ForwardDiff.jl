########
# Dual #
########

immutable Dual{T,V<:Real,N} <: Real
    value::V
    partials::Partials{N,V}
end

####################
# TagMismatchError #
####################

immutable TagMismatchError{X,Y} <: Exception
    x::Dual{X}
    y::Dual{Y}
end

function Base.showerror{X,Y}(io::IO, e::TagMismatchError{X,Y})
    print(io, "potential perturbation confusion detected when computing binary operation ",
              "on $(e.x) and $(e.y) (tag $X != tag $Y). ForwardDiff cannot safely perform ",
              "differentiation in this context; see the following issue for details: ",
              "https://github.com/JuliaDiff/ForwardDiff.jl/issues/83")
end

################
# Constructors #
################

(::Type{Dual{T}}){T,N,V}(value::V, partials::Partials{N,V}) = Dual{T,V,N}(value, partials)

function (::Type{Dual{T}}){T,N,A,B}(value::A, partials::Partials{N,B})
    C = promote_type(A, B)
    return Dual{T}(convert(C, value), convert(Partials{N,C}, partials))
end

(::Type{Dual{T}}){T}(value::Real, partials::Tuple) = Dual{T}(value, Partials(partials))
(::Type{Dual{T}}){T}(value::Real, partials::Tuple{}) = Dual{T}(value, Partials{0,typeof(value)}(partials))
(::Type{Dual{T}}){T}(value::Real, partials::Real...) = Dual{T}(value, partials)
(::Type{Dual{T}}){T,V<:Real,N,i}(value::V, ::Type{Val{N}}, ::Type{Val{i}}) = Dual{T}(value, single_seed(Partials{N,V}, Val{i}))

Dual(args...) = Dual{Void}(args...)

##############################
# Utility/Accessor Functions #
##############################

@inline value(x::Real) = x
@inline value(d::Dual) = d.value

@inline partials(x::Real) = Partials{0,typeof(x)}(tuple())
@inline partials(d::Dual) = d.partials
@inline partials(x::Real, i...) = zero(x)
@inline partials(d::Dual, i) = d.partials[i]
@inline partials(d::Dual, i, j) = partials(d, i).partials[j]
@inline partials(d::Dual, i, j, k...) = partials(partials(d, i, j), k...)

@inline valtype{V}(::V) = V
@inline valtype{V}(::Type{V}) = V
@inline valtype{T,V,N}(::Dual{T,V,N}) = V
@inline valtype{T,V,N}(::Type{Dual{T,V,N}}) = V

#####################
# Generic Functions #
#####################

Base.copy(d::Dual) = d

Base.eps(d::Dual) = eps(value(d))
Base.eps{D<:Dual}(::Type{D}) = eps(valtype(D))

Base.rtoldefault{D<:Dual}(::Type{D}) = Base.rtoldefault(valtype(D))

Base.floor{R<:Real}(::Type{R}, d::Dual) = floor(R, value(d))
Base.floor(d::Dual) = floor(value(d))

Base.ceil{R<:Real}(::Type{R}, d::Dual) = ceil(R, value(d))
Base.ceil(d::Dual) = ceil(value(d))

Base.trunc{R<:Real}(::Type{R}, d::Dual) = trunc(R, value(d))
Base.trunc(d::Dual) = trunc(value(d))

Base.round{R<:Real}(::Type{R}, d::Dual) = round(R, value(d))
Base.round(d::Dual) = round(value(d))

Base.hash(d::Dual) = hash(value(d))
Base.hash(d::Dual, hsh::UInt64) = hash(value(d), hsh)

function Base.read{T,V,N}(io::IO, ::Type{Dual{T,V,N}})
    value = read(io, V)
    partials = read(io, Partials{N,V})
    return Dual{T,N,V}(value, partials)
end

function Base.write(io::IO, d::Dual)
    write(io, value(d))
    write(io, partials(d))
end

@inline Base.zero(d::Dual) = zero(typeof(d))
@inline Base.zero{T,V,N}(::Type{Dual{T,V,N}}) = Dual{T}(zero(V), zero(Partials{N,V}))

@inline Base.one(d::Dual) = one(typeof(d))
@inline Base.one{T,V,N}(::Type{Dual{T,V,N}}) = Dual{T}(one(V), zero(Partials{N,V}))

@inline Base.rand(d::Dual) = rand(typeof(d))
@inline Base.rand{T,V,N}(::Type{Dual{T,V,N}}) = Dual{T}(rand(V), zero(Partials{N,V}))
@inline Base.rand(rng::AbstractRNG, d::Dual) = rand(rng, typeof(d))
@inline Base.rand{T,V,N}(rng::AbstractRNG, ::Type{Dual{T,V,N}}) = Dual{T}(rand(rng, V), zero(Partials{N,V}))

# Predicates #
#------------#

isconstant(d::Dual) = iszero(partials(d))

for pred in (:isequal, :(==), :isless, :(<=), :<)
    @eval begin
        Base.$(pred)(x::Dual, y::Dual) = throw(TagMismatchError(x, y))
        Base.$(pred){T}(x::Dual{T}, y::Dual{T}) = $(pred)(value(x), value(y))
        Base.$(pred){X,Y,V,N}(x::Dual{X}, y::Dual{Y,Dual{X,V,N}}) = $(pred)(x, value(y))
        Base.$(pred){X,Y,V,N}(x::Dual{X,Dual{Y,V,N}}, y::Dual{Y}) = $(pred)(value(x), y)
    end
    for R in (:AbstractFloat, :Irrational, :Real)
        @eval begin
            Base.$(pred)(x::Dual, y::$R) = $(pred)(value(x), y)
            Base.$(pred)(x::$R, y::Dual) = $(pred)(x, value(y))
        end
    end
end

Base.isnan(d::Dual) = isnan(value(d))
Base.isfinite(d::Dual) = isfinite(value(d))
Base.isinf(d::Dual) = isinf(value(d))
Base.isreal(d::Dual) = isreal(value(d))
Base.isinteger(d::Dual) = isinteger(value(d))
Base.iseven(d::Dual) = iseven(value(d))
Base.isodd(d::Dual) = isodd(value(d))

########################
# Promotion/Conversion #
########################

Base.promote_rule{T,A<:Real,B<:Real,N}(::Type{Dual{T,A,N}}, ::Type{Dual{T,B,N}}) = Dual{T,promote_type(A, B),N}

for R in (:BigFloat, :Bool, :Irrational, :Real)
    @eval begin
        Base.promote_rule{R<:$R,T,V<:Real,N}(::Type{R}, ::Type{Dual{T,V,N}}) = Dual{T,promote_type(R, V),N}
        Base.promote_rule{T,V<:Real,N,R<:$R}(::Type{Dual{T,V,N}}, ::Type{R}) = Dual{T,promote_type(V, R),N}
    end
end

Base.convert(::Type{Dual}, d::Dual) = d
Base.convert{T,V<:Real,N}(::Type{Dual{T,V,N}}, d::Dual{T}) = Dual{T}(convert(V, value(d)), convert(Partials{N,V}, partials(d)))
Base.convert{D<:Dual}(::Type{D}, n::D) = d
Base.convert{T,V<:Real,N}(::Type{Dual{T,V,N}}, x::Real) = Dual{T}(V(x), zero(Partials{N,V}))
Base.convert(::Type{Dual}, x::Real) = Dual(x)

Base.promote_array_type{D<:Dual, A<:AbstractFloat}(F, ::Type{D}, ::Type{A}) = promote_type(D, A)
Base.promote_array_type{D<:Dual, A<:AbstractFloat, P}(F, ::Type{D}, ::Type{A}, ::Type{P}) = P
Base.promote_array_type{A<:AbstractFloat, D<:Dual}(F, ::Type{A}, ::Type{D}) = promote_type(D, A)
Base.promote_array_type{A<:AbstractFloat, D<:Dual, P}(F, ::Type{A}, ::Type{D}, ::Type{P}) = P

Base.float{T,V,N}(d::Dual{T,V,N}) = Dual{T,promote_type(V, Float16),N}(d)
Base.AbstractFloat{T,V,N}(d::Dual{T,V,N}) = Dual{T,promote_type(V, Float16),N}(d)

########
# Math #
########

# Addition/Subtraction #
#----------------------#

@inline Base.:+(x::Dual, y::Dual) = throw(TagMismatchError(x, y))
@inline Base.:+{T}(x::Dual{T}, y::Dual{T}) = Dual{T}(value(x) + value(y), partials(x) + partials(y))
@inline Base.:+{T}(x::Dual{T}, y::Real) = Dual{T}(value(x) + y, partials(x))
@inline Base.:+{X,Y,V,N}(x::Dual{X,Dual{Y,V,N}}, y::Dual{Y}) = Dual{X}(value(x) + y, partials(x))
@inline Base.:+{T}(x::Real, y::Dual{T}) = Dual{T}(x + value(y), partials(y))
@inline Base.:+{X,Y,V,N}(x::Dual{X}, y::Dual{Y,Dual{X,V,N}}) = Dual{Y}(x + value(y), partials(y))

@inline Base.:-(x::Dual, y::Dual) = throw(TagMismatchError(x, y))
@inline Base.:-{T}(x::Dual{T}, y::Dual{T}) = Dual{T}(value(x) - value(y), partials(x) - partials(y))
@inline Base.:-{T}(x::Dual{T}, y::Real) = Dual{T}(value(x) - y, partials(x))
@inline Base.:-{X,Y,V,N}(x::Dual{X,Dual{Y,V,N}}, y::Dual{Y}) = Dual{X}(value(x) - y, partials(x))
@inline Base.:-{T}(x::Real, y::Dual{T}) = Dual{T}(x - value(y), -partials(y))
@inline Base.:-{X,Y,V,N}(x::Dual{X}, y::Dual{Y,Dual{X,V,N}}) = Dual{Y}(x - value(y), -partials(y))

@inline Base.:-(d::Dual) = Dual(-value(d), -partials(d))

# Multiplication #
#----------------#

@inline Base.:*(x::Dual, y::Dual) = throw(TagMismatchError(x, y))

@inline function Base.:*{T}(x::Dual{T}, y::Dual{T})
    vx, vy = value(x), value(y)
    return Dual{T}(vx * vy, _mul_partials(partials(x), partials(y), vy, vx))
end

@inline Base.:*{T}(x::Dual{T}, y::Real) = Dual{T}(value(x) * y, partials(x) * y)
@inline Base.:*{X,Y,V,N}(x::Dual{X,Dual{Y,V,N}}, y::Dual{Y}) = Dual{X}(value(x) * y, partials(x) * y)

@inline Base.:*{T}(x::Real, y::Dual{T}) = Dual{T}(x * value(y), x * partials(y))
@inline Base.:*{X,Y,V,N}(x::Dual{X}, y::Dual{Y,Dual{X,V,N}}) = Dual{Y}(x * value(y), x * partials(y))

@inline Base.:*(d::Dual, x::Bool) = x ? d : (signbit(value(d))==0 ? zero(d) : -zero(d))
@inline Base.:*(x::Bool, d::Dual) = d * x

# Division #
#----------#

@inline Base.:/(x::Dual, y::Dual) = throw(TagMismatchError(x, y))

@inline function Base.:/{T}(x::Dual{T}, y::Dual{T})
    vx, vy = value(x), value(y)
    return Dual{T}(vx / vy, _div_partials(partials(x), partials(y), vx, vy))
end

@inline Base.:/{T}(x::Dual{T}, y::Real) = Dual{T}(value(x) / y, partials(x) / y)
@inline Base.:/{X,Y,V,N}(x::Dual{X,Dual{Y,V,N}}, y::Dual{Y}) = Dual{X}(value(x) / y, partials(x) / y)

@inline function Base.:/{T}(x::Real, y::Dual{T})
    v = value(y)
    divv = x / v
    return Dual{T}(divv, -(divv / v) * partials(y))
end

@inline function Base.:/{X,Y,V,N}(x::Dual{X}, y::Dual{Y,Dual{X,V,N}})
    v = value(y)
    divv = x / v
    return Dual{Y}(divv, -(divv / v) * partials(y))
end

# Exponentiation #
#----------------#

for f in (:(Base.:^), :(NaNMath.pow))
    @eval begin
        @inline ($f)(x::Dual, y::Dual) = throw(TagMismatchError(x, y))

        @inline function ($f){T}(x::Dual{T}, y::Dual{T})
            vx, vy = value(x), value(y)
            expv = ($f)(vx, vy)
            powval = vy * ($f)(vx, vy - 1)
            logval = isconstant(y) ? one(expv) : expv * log(vx)
            new_partials = _mul_partials(partials(x), partials(y), powval, logval)
            return Dual{T}(expv, new_partials)
        end

        @inline function ($f){X,Y,V,N}(x::Dual{X,Dual{Y,V,N}}, y::Dual{Y})
            v = value(x)
            expv = ($f)(v, y)
            deriv = y * ($f)(v, y - 1)
            return Dual{X}(expv, deriv * partials(x))
        end

        @inline function ($f){X,Y,V,N}(x::Dual{X}, y::Dual{Y,Dual{X,V,N}})
            v = value(y)
            expv = ($f)(x, v)
            deriv = expv*log(x)
            return Dual{Y}(expv, deriv * partials(y))
        end

        @inline ($f)(::Base.Irrational{:e}, d::Dual) = exp(d)
    end

    for R in (:Integer, :Rational, :Real)
        @eval begin
            @inline function ($f){T}(x::Dual{T}, y::$R)
                v = value(x)
                expv = ($f)(v, y)
                deriv = y * ($f)(v, y - 1)
                return Dual{T}(expv, deriv * partials(x))
            end

            @inline function ($f){T}(x::$R, y::Dual{T})
                v = value(y)
                expv = ($f)(x, v)
                deriv = expv*log(x)
                return Dual{T}(expv, deriv * partials(y))
            end
        end
    end
end

# Unary Math Functions #
#--------------------- #

function to_nanmath(x::Expr)
    if x.head == :call
        funsym = Expr(:., :NaNMath, Base.Meta.quot(x.args[1]))
        return Expr(:call, funsym, [to_nanmath(z) for z in x.args[2:end]]...)
    else
        return Expr(:call, [to_nanmath(z) for z in x.args]...)
    end
end

to_nanmath(x) = x

@inline Base.conj(d::Dual) = d
@inline Base.transpose(d::Dual) = d
@inline Base.ctranspose(d::Dual) = d
@inline Base.abs(d::Dual) = signbit(value(d)) ? -d : d

for fsym in AUTO_DEFINED_UNARY_FUNCS
    v = :v
    deriv = Calculus.differentiate(:($(fsym)($v)), v)

    # exp and sqrt are manually defined below
    if !(in(fsym, (:exp, :sqrt)))
        funcs = Vector{Expr}(0)
        is_special_function = in(fsym, SPECIAL_FUNCS)
        is_special_function && push!(funcs, :(SpecialFunctions.$(fsym)))
        (!(is_special_function) || VERSION < v"0.6.0-dev.2767") && push!(funcs, :(Base.$(fsym)))
        for func in funcs
            @eval begin
                @inline function $(func){T}(d::Dual{T})
                    $(v) = value(d)
                    return Dual{T}($(func)($v), $(deriv) * partials(d))
                end
            end
        end
    end

    # extend corresponding NaNMath methods
    if fsym in NANMATH_FUNCS
        nan_deriv = to_nanmath(deriv)
        @eval begin
            @inline function NaNMath.$(fsym){T}(d::Dual{T})
                v = value(d)
                return Dual{T}(NaNMath.$(fsym)($v), $(nan_deriv) * partials(d))
            end
        end
    end
end

#################
# Special Cases #
#################

# Manually Optimized Functions #
#------------------------------#

@inline function Base.exp{T}(d::Dual{T})
    expv = exp(value(d))
    return Dual{T}(expv, expv * partials(d))
end

@inline function Base.sqrt{T}(d::Dual{T})
    sqrtv = sqrt(value(d))
    deriv = inv(sqrtv + sqrtv)
    return Dual{T}(sqrtv, deriv * partials(d))
end

# Other Functions #
#-----------------#

@inline function calc_hypot{T}(x, y, ::Type{T})
    vx = value(x)
    vy = value(y)
    h = hypot(vx, vy)
    return Dual{T}(h, (vx/h) * partials(x) + (vy/h) * partials(y))
end

@inline Base.hypot(x::Dual, y::Dual) = throw(TagMismatchError(x, y))
@inline Base.hypot{T}(x::Dual{T}, y::Dual{T}) = calc_hypot(x, y, T)
@inline Base.hypot{T}(x::Dual{T}, y::Real) = calc_hypot(x, y, T)
@inline Base.hypot{X,Y,V,N}(x::Dual{X,Dual{Y,V,N}}, y::Dual{Y}) = calc_hypot(x, y, X)
@inline Base.hypot{T}(x::Real, y::Dual{T}) = calc_hypot(x, y, T)
@inline Base.hypot{X,Y,V,N}(x::Dual{X}, y::Dual{Y,Dual{X,V,N}}) = calc_hypot(x, y, Y)

@inline sincos(x) = (sin(x), cos(x))

@inline function sincos{T}(d::Dual{T})
    sd, cd = sincos(value(d))
    return (Dual{T}(sd, cd * partials(d)), Dual{T}(cd, -sd * partials(d)))
end

@inline function calc_atan2{T}(y, x, ::Type{T})
    z = y / x
    v = value(z)
    atan2v = atan2(value(y), value(x))
    deriv = inv(one(v) + v*v)
    return Dual{T}(atan2v, deriv * partials(z))
end

@inline Base.atan2(x::Dual, y::Dual) = throw(TagMismatchError(x, y))
@inline Base.atan2{T}(x::Dual{T}, y::Dual{T}) = calc_atan2(x, y, T)
@inline Base.atan2{T}(x::Dual{T}, y::Real) = calc_atan2(x, y, T)
@inline Base.atan2{X,Y,V,N}(x::Dual{X,Dual{Y,V,N}}, y::Dual{Y}) = calc_atan2(x, y, X)
@inline Base.atan2{T}(x::Real, y::Dual{T}) = calc_atan2(x, y, T)
@inline Base.atan2{X,Y,V,N}(x::Dual{X}, y::Dual{Y,Dual{X,V,N}}) = calc_atan2(x, y, Y)

@generated function Base.fma{N}(x::Dual{N}, y::Dual{N}, z::Dual{N})
    ex = Expr(:tuple, [:(fma(value(x), partials(y)[$i], fma(value(y), partials(x)[$i], partials(z)[$i]))) for i in 1:N]...)
    return quote
        $(Expr(:meta, :inline))
        v = fma(value(x), value(y), value(z))
        Dual(v, $ex)
    end
end

@inline function Base.fma(x::Dual, y::Dual, z::Real)
    vx, vy = value(x), value(y)
    result = fma(vx, vy, z)
    return Dual(result, _mul_partials(partials(x), partials(y), vy, vx))
end

@generated function Base.fma{N}(x::Dual{N}, y::Real, z::Dual{N})
    ex = Expr(:tuple, [:(fma(partials(x)[$i], y,  partials(z)[$i])) for i in 1:N]...)
    return quote
        $(Expr(:meta, :inline))
        v = fma(value(x), y, value(z))
        Dual(v, $ex)
    end
end

@inline Base.fma(x::Real, y::Dual, z::Dual) = fma(y, x, z)

@inline function Base.fma(x::Dual, y::Real, z::Real)
    vx = value(x)
    return Dual(fma(vx, y, value(z)), partials(x) * y)
end

@inline Base.fma(x::Real, y::Dual, z::Real) = fma(y, x, z)

@inline function Base.fma(x::Real, y::Real, z::Dual)
    Dual(fma(x, y, value(z)), partials(z))
end

###################
# Pretty Printing #
###################

function Base.show{T,V,N}(io::IO, d::Dual{T,V,N})
    print(io, "Dual{$T}(", value(d))
    for i in 1:N
        print(io, ",", partials(d, i))
    end
    print(io, ")")
end

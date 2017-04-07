########
# Dual #
########

struct Dual{T,V<:Real,N} <: Real
    value::V
    partials::Partials{N,V}
end

################
# Constructors #
################

@inline (::Type{Dual{T}}){T,N,V}(value::V, partials::Partials{N,V}) = Dual{T,V,N}(value, partials)

@inline function (::Type{Dual{T}}){T,N,A,B}(value::A, partials::Partials{N,B})
    C = promote_type(A, B)
    return Dual{T}(convert(C, value), convert(Partials{N,C}, partials))
end

@inline (::Type{Dual{T}}){T}(value::Real, partials::Tuple) = Dual{T}(value, Partials(partials))
@inline (::Type{Dual{T}}){T}(value::Real, partials::Tuple{}) = Dual{T}(value, Partials{0,typeof(value)}(partials))
@inline (::Type{Dual{T}}){T}(value::Real, partials::Real...) = Dual{T}(value, partials)
@inline (::Type{Dual{T}}){T,V<:Real,N,i}(value::V, ::Type{Val{N}}, ::Type{Val{i}}) = Dual{T}(value, single_seed(Partials{N,V}, Val{i}))

@inline Dual(args...) = Dual{Void}(args...)

####################
# TagMismatchError #
####################

struct TagMismatchError{X,Y} <: Exception
    x::Dual{X}
    y::Dual{Y}
end

function TagMismatchError(x, y, z)
    if isa(x, Dual) && isa(y, Dual) && tagtype(x) !== tagtype(y)
        return TagMismatchError(x, y)
    elseif isa(x, Dual) && isa(z, Dual) && tagtype(x) !== tagtype(z)
        return TagMismatchError(x, z)
    elseif isa(y, Dual) && isa(z, Dual) && tagtype(y) !== tagtype(z)
        return TagMismatchError(y, z)
    else
        error("the provided arguments have matching tags, or are not Duals")
    end
end

function Base.showerror{X,Y}(io::IO, e::TagMismatchError{X,Y})
    print(io, "potential perturbation confusion detected when computing binary operation ",
              "on $(e.x) and $(e.y) (tag mismatch: $X != $Y). ForwardDiff cannot safely ",
              "perform differentiation in this context; see the following issue for ",
              "details: https://github.com/JuliaDiff/jl/issues/83")
end

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

@inline npartials(::Dual{T,V,N}) where {T,V,N} = N
@inline npartials(::Type{Dual{T,V,N}}) where {T,V,N} = N

@inline order(::Type{V}) where {V} = 0
@inline order(::Type{Dual{T,V,N}}) where {T,V,N} = 1 + order(V)

@inline valtype(::V) where {V} = V
@inline valtype(::Type{V}) where {V} = V
@inline valtype(::Dual{T,V,N}) where {T,V,N} = V
@inline valtype(::Type{Dual{T,V,N}}) where {T,V,N} = V

@inline tagtype(::V) where {V} = Void
@inline tagtype(::Type{V}) where {V} = Void
@inline tagtype(::Dual{T,V,N}) where {T,V,N} = T
@inline tagtype(::Type{Dual{T,V,N}}) where {T,V,N} = T

#####################################
# N-ary Operation Definition Macros #
#####################################

macro define_binary_dual_op(f, xy_body, x_body, y_body)
    expr = quote
        @inline $(f)(x::Dual, y::Dual) = throw(TagMismatchError(x, y))
        @inline $(f){T}(x::Dual{T}, y::Dual{T}) = $xy_body
    end
    for R in REAL_TYPES
        real_defs = quote
            @inline $(f)(x::Dual{T}, y::$R) where {T} = $x_body
            @inline $(f)(x::$R, y::Dual{T}) where {T} = $y_body
        end
        append!(expr.args, real_defs.args)
    end
    nested_defs = quote
        @inline $(f)(x::Dual{T,Dual{S,<:Real,N},M}, y::Dual{T,Dual{S,<:Real,N},M}) where {T,S,N,M} = $xy_body
        @inline $(f)(x::Dual{T,Dual{S,<:Real,N},M}, y::Dual{S,<:Real,N}) where {T,S,N,M} = $x_body
        @inline $(f)(x::Dual{S,<:Real,N}, y::Dual{T,Dual{S,<:Real,N},M}) where {T,S,N,M} = $y_body
    end
    append!(expr.args, nested_defs.args)
    return esc(expr)
end

# macro define_ternary_dual_op(f, xyz_body, xy_body, xz_body, yz_body, x_body, y_body, z_body)
#     return esc(quote
#         @inline $(f){T}(x::Dual, y::Dual, z::Dual) = throw(TagMismatchError(x, y, z))
#         @inline $(f){T}(x::Dual{T}, y::Dual{T}, z::Dual{T}) = $xyz_body
#
#         @inline $(f){T,X,Y,Z,N}(x::Dual{T,Dual{T,X,N}}, y::Dual{T,Dual{T,Y,N}}, z::Dual{T,Dual{T,Z,N}}) = $xyz_body
#         @inline $(f){T,X,Y,Z,N}(x::Dual{T,Dual{T,X,N}}, y::Dual{T,Dual{T,Y,N}}, z::Dual{T,Z,N})         = $xy_body
#         @inline $(f){T,X,Y,Z,N}(x::Dual{T,Dual{T,X,N}}, y::Dual{T,Y,N},         z::Dual{T,Dual{T,Z,N}}) = $xz_body
#         @inline $(f){T,X,Y,Z,N}(x::Dual{T,X,N},         y::Dual{T,Dual{T,Y,N}}, z::Dual{T,Dual{T,Z,N}}) = $yz_body
#         @inline $(f){T,X,Y,Z,N}(x::Dual{T,Dual{T,X,N}}, y::Dual{T,Y,N},         z::Dual{T,Z,N})         = $x_body
#         @inline $(f){T,X,Y,Z,N}(x::Dual{T,X,N},         y::Dual{T,Dual{T,Y,N}}, z::Dual{T,Z,N})         = $y_body
#         @inline $(f){T,X,Y,Z,N}(x::Dual{T,X,N},         y::Dual{T,Y,N},         z::Dual{T,Dual{T,Z,N}}) = $z_body
#
#         @inline $(f){T,S,X,Y,Z,N}(x::Dual{T,Dual{S,X,N}}, y::Dual{T,Dual{S,Y,N}}, z::Dual{T,Dual{S,Z,N}}) = $xyz_body
#         @inline $(f){T,S,X,Y,Z,N}(x::Dual{T,Dual{S,X,N}}, y::Dual{T,Dual{S,Y,N}}, z::Dual{S,Z,N})         = $xy_body
#         @inline $(f){T,S,X,Y,Z,N}(x::Dual{T,Dual{S,X,N}}, y::Dual{S,Y,N},         z::Dual{T,Dual{S,Z,N}}) = $xz_body
#         @inline $(f){T,S,X,Y,Z,N}(x::Dual{S,X,N},         y::Dual{T,Dual{S,Y,N}}, z::Dual{T,Dual{S,Z,N}}) = $yz_body
#         @inline $(f){T,S,X,Y,Z,N}(x::Dual{T,Dual{S,X,N}}, y::Dual{S,Y,N},         z::Dual{S,Z,N})         = $x_body
#         @inline $(f){T,S,X,Y,Z,N}(x::Dual{S,X,N},         y::Dual{T,Dual{S,Y,N}}, z::Dual{S,Z,N})         = $y_body
#         @inline $(f){T,S,X,Y,Z,N}(x::Dual{S,X,N},         y::Dual{S,Y,N},         z::Dual{T,Dual{S,Z,N}}) = $z_body
#
#         for RT in REAL_TYPES
#             R = Expr(:$, :RT)
#             @eval begin
#                 println("here 1")
#                 @inline $(f)(x::Dual, y::Dual, z::$R) = throw(TagMismatchError(x, y, z))
#                 @inline $(f)(x::Dual, y::$R, z::Dual) = throw(TagMismatchError(x, y, z))
#                 @inline $(f)(x::$R, y::Dual, z::Dual) = throw(TagMismatchError(x, y, z))
#
#                 @inline $(f){T}(x::Dual{T}, y::Dual{T}, z::$R) = $xy_body
#                 @inline $(f){T}(x::Dual{T}, y::$R, z::Dual{T}) = $xz_body
#                 @inline $(f){T}(x::$R, y::Dual{T}, z::Dual{T}) = $yz_body
#
#                 @inline $(f){T,X,Y,N}(x::Dual{T,Dual{T,X,N}}, y::Dual{T,Dual{T,Y,N}}, z::$R)                  = $xy_body
#                 @inline $(f){T,X,Z,N}(x::Dual{T,Dual{T,X,N}}, y::$R,                  z::Dual{T,Dual{T,Z,N}}) = $xz_body
#                 @inline $(f){T,Y,Z,N}(x::$R,                  y::Dual{T,Dual{T,Y,N}}, z::Dual{T,Dual{T,Z,N}}) = $yz_body
#                 @inline $(f){T,X,Y,N}(x::Dual{T,Dual{T,X,N}}, y::Dual{T,Y,N}, z::$R)          = $x_body
#                 @inline $(f){T,X,Z,N}(x::Dual{T,Dual{T,X,N}}, y::$R,          z::Dual{T,Z,N}) = $x_body
#                 @inline $(f){T,X,Y,N}(x::Dual{T,X,N}, y::Dual{T,Dual{T,Y,N}}, z::$R)          = $y_body
#                 @inline $(f){T,Y,Z,N}(x::$R,          y::Dual{T,Dual{T,Y,N}}, z::Dual{T,Z,N}) = $y_body
#                 @inline $(f){T,X,Z,N}(x::Dual{T,X,N}, y::$R, z::Dual{T,Dual{T,Z,N}}) = $z_body
#                 @inline $(f){T,Y,Z,N}(x::$R, y::Dual{T,Y,N}, z::Dual{T,Dual{T,Z,N}}) = $z_body
#                 println("here 2")
#                 @inline $(f){T,S,X,Y,N}(x::Dual{T,Dual{S,X,N}}, y::Dual{T,Dual{S,Y,N}}, z::$R)                  = $xy_body
#                 @inline $(f){T,S,X,Z,N}(x::Dual{T,Dual{S,X,N}}, y::$R,                  z::Dual{T,Dual{S,Z,N}}) = $xz_body
#                 @inline $(f){T,S,Y,Z,N}(x::$R,                  y::Dual{T,Dual{S,Y,N}}, z::Dual{T,Dual{S,Z,N}}) = $yz_body
#                 @inline $(f){T,S,X,Y,N}(x::Dual{T,Dual{S,X,N}}, y::Dual{S,Y,N}, z::$R)          = $x_body
#                 @inline $(f){T,S,X,Z,N}(x::Dual{T,Dual{S,X,N}}, y::$R,          z::Dual{S,Z,N}) = $x_body
#                 @inline $(f){T,S,X,Y,N}(x::Dual{S,X,N}, y::Dual{T,Dual{S,Y,N}}, z::$R)          = $y_body
#                 @inline $(f){T,S,Y,Z,N}(x::$R,          y::Dual{T,Dual{S,Y,N}}, z::Dual{S,Z,N}) = $y_body
#                 @inline $(f){T,S,X,Z,N}(x::Dual{S,X,N}, y::$R, z::Dual{T,Dual{S,Z,N}}) = $z_body
#                 @inline $(f){T,S,Y,Z,N}(x::$R, y::Dual{S,Y,N}, z::Dual{T,Dual{S,Z,N}}) = $z_body
#             end
#             for QT in REAL_TYPES
#                 Q = Expr(:$, :QT)
#                 @eval begin
#                     println("here 3")
#                     @inline $(f)(x::Dual, y::$R, z::$Q) = throw(TagMismatchError(x, y, z))
#                     @inline $(f)(x::$R, y::Dual, z::$Q) = throw(TagMismatchError(x, y, z))
#                     @inline $(f)(x::$R, y::$Q, z::Dual) = throw(TagMismatchError(x, y, z))
#                     println("here 4")
#                     @inline $(f){T}(x::Dual{T}, y::$R, z::$Q) = $x_body
#                     @inline $(f){T}(x::$R, y::Dual{T}, z::$Q) = $y_body
#                     @inline $(f){T}(x::$R, y::$Q, z::Dual{T}) = $z_body
#                 end
#             end
#         end
#     end)
# end

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
    return Dual{T,V,N}(value, partials)
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
        @define_binary_dual_op(
            Base.$(pred),
            $(pred)(value(x), value(y)),
            $(pred)(value(x), y),
            $(pred)(x, value(y))
        )
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

Base.convert{T,V<:Real,N}(::Type{Dual{T,V,N}}, d::Dual{T}) = Dual{T}(convert(V, value(d)), convert(Partials{N,V}, partials(d)))
Base.convert{T,V<:Real,N}(::Type{Dual{T,V,N}}, x::Real) = Dual{T}(V(x), zero(Partials{N,V}))
Base.convert{D<:Dual}(::Type{D}, d::D) = d

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

@define_binary_dual_op(
    Base.:+,
    Dual{T}(value(x) + value(y), partials(x) + partials(y)),
    Dual{T}(value(x) + y, partials(x)),
    Dual{T}(x + value(y), partials(y))
)

@define_binary_dual_op(
    Base.:-,
    Dual{T}(value(x) - value(y), partials(x) - partials(y)),
    Dual{T}(value(x) - y, partials(x)),
    Dual{T}(x - value(y), -partials(y))
)

@inline Base.:-{T}(d::Dual{T}) = Dual{T}(-value(d), -partials(d))

# Multiplication #
#----------------#

@define_binary_dual_op(
    Base.:*,
    begin
        vx, vy = value(x), value(y)
        Dual{T}(vx * vy, _mul_partials(partials(x), partials(y), vy, vx))
    end,
    Dual{T}(value(x) * y, partials(x) * y),
    Dual{T}(x * value(y), x * partials(y))
)

@inline Base.:*(d::Dual, x::Bool) = x ? d : (signbit(value(d))==0 ? zero(d) : -zero(d))
@inline Base.:*(x::Bool, d::Dual) = d * x

# Division #
#----------#

@define_binary_dual_op(
    Base.:/,
    begin
        vx, vy = value(x), value(y)
        Dual{T}(vx / vy, _div_partials(partials(x), partials(y), vx, vy))
    end,
    Dual{T}(value(x) / y, partials(x) / y),
    begin
        v = value(y)
        divv = x / v
        Dual{T}(divv, -(divv / v) * partials(y))
    end
)

# Exponentiation #
#----------------#

for f in (:(Base.:^), :(NaNMath.pow))
    @eval begin
        @define_binary_dual_op(
            $f,
            begin
                vx, vy = value(x), value(y)
                expv = ($f)(vx, vy)
                powval = vy * ($f)(vx, vy - 1)
                logval = isconstant(y) ? one(expv) : expv * log(vx)
                new_partials = _mul_partials(partials(x), partials(y), powval, logval)
                return Dual{T}(expv, new_partials)
            end,
            begin
                v = value(x)
                expv = ($f)(v, y)
                deriv = y * ($f)(v, y - 1)
                return Dual{T}(expv, deriv * partials(x))
            end,
            begin
                v = value(y)
                expv = ($f)(x, v)
                deriv = expv*log(x)
                return Dual{T}(expv, deriv * partials(y))
            end
        )
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

# hypot

@inline function calc_hypot{T}(x, y, ::Type{T})
    vx = value(x)
    vy = value(y)
    h = hypot(vx, vy)
    return Dual{T}(h, (vx/h) * partials(x) + (vy/h) * partials(y))
end

@define_binary_dual_op(
    Base.hypot,
    calc_hypot(x, y, T),
    calc_hypot(x, y, T),
    calc_hypot(x, y, T)
)

@inline function calc_hypot{T}(x, y, z, ::Type{T})
    vx = value(x)
    vy = value(y)
    vz = value(z)
    h = hypot(vx, vy, vz)
    p = (vx / h) * partials(x) + (vy / h) * partials(y) + (vz / h) * partials(z)
    return Dual{T}(h, p)
end

# @define_ternary_dual_op(
#     Base.hypot,
#     calc_hypot(x, y, z, T),
#     calc_hypot(x, y, z, T),
#     calc_hypot(x, y, z, T),
#     calc_hypot(x, y, z, T),
#     calc_hypot(x, y, z, T),
#     calc_hypot(x, y, z, T),
#     calc_hypot(x, y, z, T),
# )

# atan2

@inline function calc_atan2{T}(y, x, ::Type{T})
    z = y / x
    v = value(z)
    atan2v = atan2(value(y), value(x))
    deriv = inv(one(v) + v*v)
    return Dual{T}(atan2v, deriv * partials(z))
end

@define_binary_dual_op(
    Base.atan2,
    calc_atan2(x, y, T),
    calc_atan2(x, y, T),
    calc_atan2(x, y, T)
)

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

# sincos

@inline sincos(x) = (sin(x), cos(x))

@inline function sincos{T}(d::Dual{T})
    sd, cd = sincos(value(d))
    return (Dual{T}(sd, cd * partials(d)), Dual{T}(cd, -sd * partials(d)))
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

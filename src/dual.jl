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

@inline (::Type{Dual{T}})(value::V, partials::Partials{N,V}) where {T,N,V} = Dual{T,V,N}(value, partials)

@inline function (::Type{Dual{T}})(value::A, partials::Partials{N,B}) where {T,N,A,B}
    C = promote_type(A, B)
    return Dual{T}(convert(C, value), convert(Partials{N,C}, partials))
end

@inline (::Type{Dual{T}})(value::Real, partials::Tuple) where {T} = Dual{T}(value, Partials(partials))
@inline (::Type{Dual{T}})(value::Real, partials::Tuple{}) where {T} = Dual{T}(value, Partials{0,typeof(value)}(partials))
@inline (::Type{Dual{T}})(value::Real, partials::Real...) where {T} = Dual{T}(value, partials)
@inline (::Type{Dual{T}})(value::V, ::Chunk{N}, p::Val{i}) where {T,V<:Real,N,i} = Dual{T}(value, single_seed(Partials{N,V}, p))

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

function Base.showerror(io::IO, e::TagMismatchError{X,Y}) where {X,Y}
    print(io, "potential perturbation confusion detected when computing binary operation ",
              "on $(e.x) and $(e.y) (tag mismatch: $X != $Y). ForwardDiff cannot safely ",
              "perform differentiation in this context; see the following issue for ",
              "details: https://github.com/JuliaDiff/ForwardDiff.jl/issues/83")
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

####################################
# N-ary Operation Definition Tools #
####################################

macro define_binary_dual_op(f, xy_body, x_body, y_body)
    defs = quote
        @inline $(f)(x::Dual, y::Dual) = throw(TagMismatchError(x, y))
        @inline $(f)(x::Dual{T}, y::Dual{T}) where {T} = $xy_body
        @inline $(f)(x::Dual{T,Dual{S,X,N},M}, y::Dual{T,Dual{S,Y,N},M}) where {T,S,X<:Real,Y<:Real,N,M} = $xy_body
        @inline $(f)(x::Dual{T,Dual{S,X,N},M}, y::Dual{S,Y,N})           where {T,S,X<:Real,Y<:Real,N,M} = $x_body
        @inline $(f)(x::Dual{S,X,N},           y::Dual{T,Dual{S,Y,N},M}) where {T,S,X<:Real,Y<:Real,N,M} = $y_body
    end
    for R in REAL_TYPES
        expr = quote
            @inline $(f)(x::Dual{T}, y::$R) where {T} = $x_body
            @inline $(f)(x::$R, y::Dual{T}) where {T} = $y_body
        end
        append!(defs.args, expr.args)
    end
    return esc(defs)
end

macro define_ternary_dual_op(f, xyz_body, xy_body, xz_body, yz_body, x_body, y_body, z_body)
    defs = quote
        @inline $(f)(x::Dual, y::Dual, z::Dual) = throw(TagMismatchError(x, y, z))
        @inline $(f)(x::Dual{T}, y::Dual{T}, z::Dual{T}) where {T} = $xyz_body
        @inline $(f)(x::Dual{T,Dual{S,X,N},M}, y::Dual{T,Dual{S,Y,N},M}, z::Dual{T,Dual{S,Z,N},M}) where {T,S,X<:Real,Y<:Real,Z<:Real,N,M} = $xyz_body
        @inline $(f)(x::Dual{T,Dual{S,X,N},M}, y::Dual{T,Dual{S,Y,N},M}, z::Dual{S,Z,N})           where {T,S,X<:Real,Y<:Real,Z<:Real,N,M} = $xy_body
        @inline $(f)(x::Dual{T,Dual{S,X,N},M}, y::Dual{S,Y,N},           z::Dual{T,Dual{S,Z,N},M}) where {T,S,X<:Real,Y<:Real,Z<:Real,N,M} = $xz_body
        @inline $(f)(x::Dual{S,X,N},           y::Dual{T,Dual{S,Y,N},M}, z::Dual{T,Dual{S,Z,N},M}) where {T,S,X<:Real,Y<:Real,Z<:Real,N,M} = $yz_body
        @inline $(f)(x::Dual{T,Dual{S,X,N},M}, y::Dual{S,Y,N},           z::Dual{S,Z,N})           where {T,S,X<:Real,Y<:Real,Z<:Real,N,M} = $x_body
        @inline $(f)(x::Dual{S,X,N},           y::Dual{T,Dual{S,Y,N},M}, z::Dual{S,Z,N})           where {T,S,X<:Real,Y<:Real,Z<:Real,N,M} = $y_body
        @inline $(f)(x::Dual{S,X,N},           y::Dual{S,Y,N},           z::Dual{T,Dual{S,Z,N},M}) where {T,S,X<:Real,Y<:Real,Z<:Real,N,M} = $z_body
    end
    for R in REAL_TYPES
        expr = quote
            @inline $(f)(x::Dual, y::Dual, z::$R) = throw(TagMismatchError(x, y, z))
            @inline $(f)(x::Dual, y::$R, z::Dual) = throw(TagMismatchError(x, y, z))
            @inline $(f)(x::$R, y::Dual, z::Dual) = throw(TagMismatchError(x, y, z))

            @inline $(f)(x::Dual{T}, y::Dual{T}, z::$R) where {T} = $xy_body
            @inline $(f)(x::Dual{T}, y::$R, z::Dual{T}) where {T} = $xz_body
            @inline $(f)(x::$R, y::Dual{T}, z::Dual{T}) where {T} = $yz_body

            @inline $(f)(x::Dual{T,Dual{S,X,N},M}, y::Dual{T,Dual{S,Y,N},M}, z::$R)                    where {T,S,X<:Real,Y<:Real,N,M} = $xy_body
            @inline $(f)(x::Dual{T,Dual{S,X,N},M}, y::$R,                    z::Dual{T,Dual{S,Z,N},M}) where {T,S,X<:Real,Z<:Real,N,M} = $xz_body
            @inline $(f)(x::$R,                    y::Dual{T,Dual{S,Y,N},M}, z::Dual{T,Dual{S,Z,N},M}) where {T,S,Y<:Real,Z<:Real,N,M} = $yz_body

            @inline $(f)(x::Dual{T,Dual{S,X,N},M}, y::Dual{S,Y,N},           z::$R)                    where {T,S,X<:Real,Y<:Real,N,M} = $x_body
            @inline $(f)(x::Dual{T,Dual{S,X,N},M}, y::$R,                    z::Dual{S,Z,N})           where {T,S,X<:Real,Z<:Real,N,M} = $x_body
            @inline $(f)(x::$R,                    y::Dual{T,Dual{S,Y,N},M}, z::Dual{S,Z,N})           where {T,S,Y<:Real,Z<:Real,N,M} = $y_body
            @inline $(f)(x::Dual{S,X,N},           y::Dual{T,Dual{S,Y,N},M}, z::$R)                    where {T,S,X<:Real,Y<:Real,N,M} = $y_body
            @inline $(f)(x::Dual{S,X,N},           y::$R,                    z::Dual{T,Dual{S,Z,N},M}) where {T,S,X<:Real,Z<:Real,N,M} = $z_body
            @inline $(f)(x::$R,                    y::Dual{S,Y,N},           z::Dual{T,Dual{S,Z,N},M}) where {T,S,Y<:Real,Z<:Real,N,M} = $z_body
        end
        append!(defs.args, expr.args)
        for Q in REAL_TYPES
            Q === R && continue
            expr = quote
                @inline $(f)(x::Dual{T}, y::$R, z::$Q) where {T} = $x_body
                @inline $(f)(x::$R, y::Dual{T}, z::$Q) where {T} = $y_body
                @inline $(f)(x::$R, y::$Q, z::Dual{T}) where {T} = $z_body
            end
            append!(defs.args, expr.args)
        end
        expr = quote
            @inline $(f)(x::Dual{T}, y::$R, z::$R) where {T} = $x_body
            @inline $(f)(x::$R, y::Dual{T}, z::$R) where {T} = $y_body
            @inline $(f)(x::$R, y::$R, z::Dual{T}) where {T} = $z_body
        end
        append!(defs.args, expr.args)
    end
    return esc(defs)
end

function unary_dual_definition(M, f)
    work = CommonSubexpressions.cse(quote
        val = $(f)(x)
        deriv = $(DiffBase.diffrule(f, :x))
    end)
    return quote
        @inline function ($M).$(f)(d::Dual{T}) where T
            x = value(d)
            $work
            return Dual{T}(val, deriv * partials(d))
        end
    end
end

function binary_dual_definition(M, f)
    dvx, dvy = DiffBase.diffrule(f, :vx, :vy)
    xy_work = CommonSubexpressions.cse(quote
        val = $(f)(vx, vy)
        dvx = $dvx
        dvy = $dvy
    end)
    x_work = CommonSubexpressions.cse(quote
        val = $(f)(vx, vy)
        dvx = $dvx
    end)
    y_work = CommonSubexpressions.cse(quote
        val = $(f)(vx, vy)
        dvy = $dvy
    end)
    return quote
        @define_binary_dual_op(
            ($M).$(f),
            begin
                vx, vy = value(x), value(y)
                $xy_work
                return Dual{T}(val, _mul_partials(partials(x), partials(y), dvx, dvy))
            end,
            begin
                vx, vy = value(x), value(y)
                $x_work
                return Dual{T}(val, dvx * partials(x))
            end,
            begin
                vx, vy = value(x), value(y)
                $y_work
                return Dual{T}(val, dvy * partials(y))
            end
        )
    end
end

#####################
# Generic Functions #
#####################

Base.copy(d::Dual) = d

Base.eps(d::Dual) = eps(value(d))
Base.eps(::Type{D}) where {D<:Dual} = eps(valtype(D))

Base.rtoldefault(::Type{D}) where {D<:Dual} = Base.rtoldefault(valtype(D))

Base.floor(::Type{R}, d::Dual) where {R<:Real} = floor(R, value(d))
Base.floor(d::Dual) = floor(value(d))

Base.ceil(::Type{R}, d::Dual) where {R<:Real} = ceil(R, value(d))
Base.ceil(d::Dual) = ceil(value(d))

Base.trunc(::Type{R}, d::Dual) where {R<:Real} = trunc(R, value(d))
Base.trunc(d::Dual) = trunc(value(d))

Base.round(::Type{R}, d::Dual) where {R<:Real} = round(R, value(d))
Base.round(d::Dual) = round(value(d))

Base.hash(d::Dual) = hash(value(d))
Base.hash(d::Dual, hsh::UInt) = hash(value(d), hsh)

function Base.read(io::IO, ::Type{Dual{T,V,N}}) where {T,V,N}
    value = read(io, V)
    partials = read(io, Partials{N,V})
    return Dual{T,V,N}(value, partials)
end

function Base.write(io::IO, d::Dual)
    write(io, value(d))
    write(io, partials(d))
end

@inline Base.zero(d::Dual) = zero(typeof(d))
@inline Base.zero(::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(zero(V), zero(Partials{N,V}))

@inline Base.one(d::Dual) = one(typeof(d))
@inline Base.one(::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(one(V), zero(Partials{N,V}))

@inline Base.rand(d::Dual) = rand(typeof(d))
@inline Base.rand(::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(rand(V), zero(Partials{N,V}))
@inline Base.rand(rng::AbstractRNG, d::Dual) = rand(rng, typeof(d))
@inline Base.rand(rng::AbstractRNG, ::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(rand(rng, V), zero(Partials{N,V}))

# Predicates #
#------------#

isconstant(d::Dual) = iszero(partials(d))

for pred in RealInterface.UNARY_PREDICATES
    @eval Base.$(pred)(d::Dual) = $(pred)(value(d))
end

for pred in RealInterface.BINARY_PREDICATES
    @eval begin
        @define_binary_dual_op(
            Base.$(pred),
            $(pred)(value(x), value(y)),
            $(pred)(value(x), y),
            $(pred)(x, value(y))
        )
    end
end

########################
# Promotion/Conversion #
########################

function Base.promote_rule(::Type{Dual{T,A,N}},
                           ::Type{Dual{T,B,N}}) where {T,A<:Real,B<:Real,N}
    return Dual{T,promote_type(A, B),N}
end

for R in (:BigFloat, :Bool, :Irrational, :Real)
    @eval begin
        Base.promote_rule(::Type{R}, ::Type{Dual{T,V,N}}) where {R<:$R,T,V<:Real,N} = Dual{T,promote_type(R, V),N}
        Base.promote_rule(::Type{Dual{T,V,N}}, ::Type{R}) where {T,V<:Real,N,R<:$R} = Dual{T,promote_type(V, R),N}
    end
end

Base.convert(::Type{Dual{T,V,N}}, d::Dual{T}) where {T,V<:Real,N} = Dual{T}(convert(V, value(d)), convert(Partials{N,V}, partials(d)))
Base.convert(::Type{Dual{T,V,N}}, x::Real) where {T,V<:Real,N} = Dual{T}(V(x), zero(Partials{N,V}))
Base.convert(::Type{D}, d::D) where {D<:Dual} = d

Base.promote_array_type(F, ::Type{D}, ::Type{A}) where {D<:Dual,A<:AbstractFloat} = promote_type(D, A)
Base.promote_array_type(F, ::Type{<:Dual}, ::Type{<:AbstractFloat}, ::Type{P}) where {P} = P
Base.promote_array_type(F, ::Type{A}, ::Type{D}) where {D<:Dual,A<:AbstractFloat} = promote_type(D, A)
Base.promote_array_type(F, ::Type{<:AbstractFloat}, ::Type{<:Dual}, ::Type{P}) where {P} = P

Base.float(d::Dual{T,V,N}) where {T,V,N} = Dual{T,promote_type(V, Float16),N}(d)
Base.AbstractFloat(d::Dual{T,V,N}) where {T,V,N} = Dual{T,promote_type(V, Float16),N}(d)

##############
# Arithmetic #
##############

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

@inline Base.:-(d::Dual{T}) where {T} = Dual{T}(-value(d), -partials(d))

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

###################################
# General Mathematical Operations #
###################################

@inline Base.conj(d::Dual) = d

@inline Base.transpose(d::Dual) = d

@inline Base.ctranspose(d::Dual) = d

@inline Base.abs(d::Dual) = signbit(value(d)) ? -d : d

for f in RealInterface.UNARY_MATH
    DiffBase.hasdiffrule(f, 1) && eval(unary_dual_definition(Base, f))
end

for f in RealInterface.UNARY_SPECIAL_MATH
    DiffBase.hasdiffrule(f, 1) && eval(unary_dual_definition(SpecialFunctions, f))
end

for f in RealInterface.BINARY_SPECIAL_MATH
    DiffBase.hasdiffrule(f, 2) && eval(binary_dual_definition(SpecialFunctions, f))
end

function to_nanmath!(x)
    if isa(x, Expr)
        if x.head == :call
            f = x.args[1]
            if in(f, RealInterface.UNARY_NAN_MATH) || in(f, RealInterface.BINARY_NAN_MATH)
                x.args[1] = :(NaNMath.$f)
            end
            foreach(to_nanmath!, x.args[2:end])
        else
            foreach(to_nanmath!, x.args)
        end
    end
    return x
end

for f in RealInterface.UNARY_NAN_MATH
    DiffBase.hasdiffrule(f, 1) && eval(to_nanmath!(unary_dual_definition(NaNMath, f)))
end

#################
# Special Cases #
#################

# hypot #
#-------#

@inline function calc_hypot(x, y, ::Type{T}) where T
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

@inline function calc_hypot(x, y, z, ::Type{T}) where T
    vx = value(x)
    vy = value(y)
    vz = value(z)
    h = hypot(vx, vy, vz)
    p = (vx / h) * partials(x) + (vy / h) * partials(y) + (vz / h) * partials(z)
    return Dual{T}(h, p)
end

@define_ternary_dual_op(
    Base.hypot,
    calc_hypot(x, y, z, T),
    calc_hypot(x, y, z, T),
    calc_hypot(x, y, z, T),
    calc_hypot(x, y, z, T),
    calc_hypot(x, y, z, T),
    calc_hypot(x, y, z, T),
    calc_hypot(x, y, z, T),
)

# atan2 #
#-------#

@inline function calc_atan2(y, x, ::Type{T}) where T
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

# fma #
#-----#

@generated function calc_fma_xyz(x::Dual{T,<:Real,N},
                                 y::Dual{T,<:Real,N},
                                 z::Dual{T,<:Real,N}) where {T,N}
    ex = Expr(:tuple, [:(fma(value(x), partials(y)[$i], fma(value(y), partials(x)[$i], partials(z)[$i]))) for i in 1:N]...)
    return quote
        $(Expr(:meta, :inline))
        v = fma(value(x), value(y), value(z))
        return Dual{T}(v, $ex)
    end
end

@inline function calc_fma_xy(x::Dual{T}, y::Dual{T}, z::Real) where T
    vx, vy = value(x), value(y)
    result = fma(vx, vy, z)
    return Dual{T}(result, _mul_partials(partials(x), partials(y), vy, vx))
end

@generated function calc_fma_xz(x::Dual{T,<:Real,N},
                                y::Real,
                                z::Dual{T,<:Real,N}) where {T,N}
    ex = Expr(:tuple, [:(fma(partials(x)[$i], y,  partials(z)[$i])) for i in 1:N]...)
    return quote
        $(Expr(:meta, :inline))
        v = fma(value(x), y, value(z))
        Dual(v, $ex)
    end
end

@define_ternary_dual_op(
    Base.fma,
    calc_fma_xyz(x, y, z),                         # xyz_body
    calc_fma_xy(x, y, z),                          # xy_body
    calc_fma_xz(x, y, z),                          # xz_body
    Base.fma(y, x, z),                             # yz_body
    Dual{T}(fma(value(x), y, z), partials(x) * y), # x_body
    Base.fma(y, x, z),                             # y_body
    Dual{T}(fma(x, y, value(z)), partials(z))      # z_body
)

# muladd #
#-----#

@generated function calc_muladd_xyz(x::Dual{T,<:Real,N},
                                 y::Dual{T,<:Real,N},
                                 z::Dual{T,<:Real,N}) where {T,N}
    ex = Expr(:tuple, [:(muladd(value(x), partials(y)[$i], muladd(value(y), partials(x)[$i], partials(z)[$i]))) for i in 1:N]...)
    return quote
        $(Expr(:meta, :inline))
        v = muladd(value(x), value(y), value(z))
        return Dual{T}(v, $ex)
    end
end

@inline function calc_muladd_xy(x::Dual{T}, y::Dual{T}, z::Real) where T
    vx, vy = value(x), value(y)
    result = muladd(vx, vy, z)
    return Dual{T}(result, _mul_partials(partials(x), partials(y), vy, vx))
end

@generated function calc_muladd_xz(x::Dual{T,<:Real,N},
                                y::Real,
                                z::Dual{T,<:Real,N}) where {T,N}
    ex = Expr(:tuple, [:(muladd(partials(x)[$i], y,  partials(z)[$i])) for i in 1:N]...)
    return quote
        $(Expr(:meta, :inline))
        v = muladd(value(x), y, value(z))
        Dual(v, $ex)
    end
end

@define_ternary_dual_op(
    Base.muladd,
    calc_muladd_xyz(x, y, z),                         # xyz_body
    calc_muladd_xy(x, y, z),                          # xy_body
    calc_muladd_xz(x, y, z),                          # xz_body
    Base.muladd(y, x, z),                             # yz_body
    Dual{T}(muladd(value(x), y, z), partials(x) * y), # x_body
    Base.muladd(y, x, z),                             # y_body
    Dual{T}(muladd(x, y, value(z)), partials(z))      # z_body
)

# sincos #
#--------#

@inline sincos(x) = (sin(x), cos(x))

@inline function sincos(d::Dual{T}) where T
    sd, cd = sincos(value(d))
    return (Dual{T}(sd, cd * partials(d)), Dual{T}(cd, -sd * partials(d)))
end

###################
# Pretty Printing #
###################

function Base.show(io::IO, d::Dual{T,V,N}) where {T,V,N}
    print(io, "Dual{$(repr(T))}(", value(d))
    for i in 1:N
        print(io, ",", partials(d, i))
    end
    print(io, ")")
end

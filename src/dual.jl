########
# Dual #
########

struct Dual{T,V<:Real,N} <: Real
    value::V
    partials::Partials{N,V}
end

##############
# Exceptions #
##############

struct DualMismatchError{A,B} <: Exception
    a::A
    b::B
end

Base.showerror(io::IO, e::DualMismatchError{A,B}) where {A,B} =
    print(io, "Cannot determine ordering of Dual tags $(e.a) and $(e.b)")

"""
    ForwardDiff.≺(a, b)::Bool

Determines the order in which tagged `Dual` objects are composed. If true, then `Dual{b}`
objects will appear outside `Dual{a}` objects.

This is important when working with nested differentiation: currently, only the outermost
tag can be extracted, so it should be used in the _innermost_ function.
"""
≺(a,b) = throw(DualMismatchError(a,b))

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

##############################
# Utility/Accessor Functions #
##############################

@inline value(x::Real) = x
@inline value(d::Dual) = d.value

@inline value(::Type{T}, x::Real) where T = x
@inline value(::Type{T}, d::Dual{T}) where T = value(d)
function value(::Type{T}, d::Dual{S}) where {T,S}
    # TODO: in the case of nested Duals, it may be possible to "transpose" the Dual objects
    throw(DualMismatchError(T,S))
end

@inline partials(x::Real) = Partials{0,typeof(x)}(tuple())
@inline partials(d::Dual) = d.partials
@inline partials(x::Real, i...) = zero(x)
@inline partials(d::Dual, i) = d.partials[i]
@inline partials(d::Dual, i, j) = partials(d, i).partials[j]
@inline partials(d::Dual, i, j, k...) = partials(partials(d, i, j), k...)

@inline partials(::Type{T}, x::Real, i...) where T = partials(x, i...)
@inline partials(::Type{T}, d::Dual{T}, i...) where T = partials(d, i...)
partials(::Type{T}, d::Dual{S}, i...) where {T,S} = throw(DualMismatchError(T,S))

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
        @inline $(f)(x::Dual{Txy}, y::Dual{Txy}) where {Txy} = $xy_body
        @inline $(f)(x::Dual{Tx}, y::Dual{Ty}) where {Tx,Ty} = Ty ≺ Tx ? $x_body : $y_body
    end
    for R in AMBIGUOUS_TYPES
        expr = quote
            @inline $(f)(x::Dual{Tx}, y::$R) where {Tx} = $x_body
            @inline $(f)(x::$R, y::Dual{Ty}) where {Ty} = $y_body
        end
        append!(defs.args, expr.args)
    end
    return esc(defs)
end

macro define_ternary_dual_op(f, xyz_body, xy_body, xz_body, yz_body, x_body, y_body, z_body)
    defs = quote
        @inline $(f)(x::Dual{Txyz}, y::Dual{Txyz}, z::Dual{Txyz}) where {Txyz} = $xyz_body
        @inline $(f)(x::Dual{Txy}, y::Dual{Txy}, z::Dual{Tz}) where {Txy,Tz} = Tz ≺ Txy ? $xy_body : $z_body
        @inline $(f)(x::Dual{Txz}, y::Dual{Ty}, z::Dual{Txz}) where {Txz,Ty} = Ty ≺ Txz ? $xz_body : $y_body
        @inline $(f)(x::Dual{Tx}, y::Dual{Tyz}, z::Dual{Tyz}) where {Tx,Tyz} = Tyz ≺ Tx ? $x_body  : $yz_body
        @inline function $(f)(x::Dual{Tx}, y::Dual{Ty}, z::Dual{Tz}) where {Tx,Ty,Tz}
            if Tz ≺ Tx && Ty ≺ Tx
                $x_body
            elseif Tz ≺ Ty
                $y_body
            else
                $z_body
            end
        end
    end
    for R in AMBIGUOUS_TYPES
        expr = quote
            @inline $(f)(x::Dual{Txy}, y::Dual{Txy}, z::$R) where {Txy} = $xy_body
            @inline $(f)(x::Dual{Tx}, y::Dual{Ty}, z::$R)  where {Tx, Ty} = Ty ≺ Tx ? $x_body : $y_body
            @inline $(f)(x::Dual{Txz}, y::$R, z::Dual{Txz}) where {Txz} = $xz_body
            @inline $(f)(x::Dual{Tx}, y::$R, z::Dual{Tz}) where {Tx,Tz} = Tz ≺ Tx ? $x_body : $z_body
            @inline $(f)(x::$R, y::Dual{Tyz}, z::Dual{Tyz}) where {Tyz} = $yz_body
            @inline $(f)(x::$R, y::Dual{Ty}, z::Dual{Tz}) where {Ty,Tz} = Tz ≺ Ty ? $y_body : $z_body
        end
        append!(defs.args, expr.args)
        for Q in AMBIGUOUS_TYPES
            Q === R && continue
            expr = quote
                @inline $(f)(x::Dual{Tx}, y::$R, z::$Q) where {Tx} = $x_body
                @inline $(f)(x::$R, y::Dual{Ty}, z::$Q) where {Ty} = $y_body
                @inline $(f)(x::$R, y::$Q, z::Dual{Tz}) where {Tz} = $z_body
            end
            append!(defs.args, expr.args)
        end
        expr = quote
            @inline $(f)(x::Dual{Tx}, y::$R, z::$R) where {Tx} = $x_body
            @inline $(f)(x::$R, y::Dual{Ty}, z::$R) where {Ty} = $y_body
            @inline $(f)(x::$R, y::$R, z::Dual{Tz}) where {Tz} = $z_body
        end
        append!(defs.args, expr.args)
    end
    return esc(defs)
end

function unary_dual_definition(M, f)
    work = qualified_cse!(quote
        val = $M.$f(x)
        deriv = $(DiffRules.diffrule(M, f, :x))
    end)
    return quote
        @inline function $M.$f(d::Dual{T}) where T
            x = value(d)
            $work
            return Dual{T}(val, deriv * partials(d))
        end
    end
end

function binary_dual_definition(M, f)
    dvx, dvy = DiffRules.diffrule(M, f, :vx, :vy)
    xy_work = qualified_cse!(quote
        val = $M.$f(vx, vy)
        dvx = $dvx
        dvy = $dvy
    end)
    dvx, _ = DiffRules.diffrule(M, f, :vx, :y)
    x_work = qualified_cse!(quote
        val = $M.$f(vx, y)
        dvx = $dvx
    end)
    _, dvy = DiffRules.diffrule(M, f, :x, :vy)
    y_work = qualified_cse!(quote
        val = $M.$f(x, vy)
        dvy = $dvy
    end)
    expr = quote
        @define_binary_dual_op(
            $M.$f,
            begin
                vx, vy = value(x), value(y)
                $xy_work
                return Dual{Txy}(val, _mul_partials(partials(x), partials(y), dvx, dvy))
            end,
            begin
                vx = value(x)
                $x_work
                return Dual{Tx}(val, dvx * partials(x))
            end,
            begin
                vy = value(y)
                $y_work
                return Dual{Ty}(val, dvy * partials(y))
            end
        )
    end
    return expr
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

for pred in UNARY_PREDICATES
    @eval Base.$(pred)(d::Dual) = $(pred)(value(d))
end

for pred in BINARY_PREDICATES
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

Base.@pure function Base.promote_rule(::Type{Dual{T1,V1,N1}},
                                      ::Type{Dual{T2,V2,N2}}) where {T1,V1<:Real,N1,T2,V2<:Real,N2}
    # V1 and V2 might themselves be Dual types
    if T2 ≺ T1
        Dual{T1,promote_type(V1,Dual{T2,V2,N2}),N1}
    else
        Dual{T2,promote_type(V2,Dual{T1,V1,N1}),N2}
    end
end

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

Base.float(d::Dual{T,V,N}) where {T,V,N} = Dual{T,promote_type(V, Float16),N}(d)
Base.AbstractFloat(d::Dual{T,V,N}) where {T,V,N} = Dual{T,promote_type(V, Float16),N}(d)

###################################
# General Mathematical Operations #
###################################

@inline Base.conj(d::Dual) = d

@inline Base.transpose(d::Dual) = d

@inline Base.ctranspose(d::Dual) = d

@inline Base.abs(d::Dual) = signbit(value(d)) ? -d : d

for (M, f, arity) in DiffRules.diffrules()
    in((M, f), ((:Base, :^), (:NaNMath, :pow), (:Base, :/))) && continue
    if arity == 1
        eval(unary_dual_definition(M, f))
    elseif arity == 2
        eval(binary_dual_definition(M, f))
    else
        error("ForwardDiff currently only knows how to autogenerate Dual definitions for unary and binary functions.")
    end
end

#################
# Special Cases #
#################

# * #
#---#

@inline Base.:*(d::Dual, x::Bool) = x ? d : (signbit(value(d))==0 ? zero(d) : -zero(d))
@inline Base.:*(x::Bool, d::Dual) = d * x

# / #
#---#

# We can't use the normal diffrule autogeneration for this because (x/y) === (x * (1/y))
# doesn't generally hold true for floating point; see issue #264
@define_binary_dual_op(
    Base.:/,
    begin
        vx, vy = value(x), value(y)
        Dual{Txy}(vx / vy, _div_partials(partials(x), partials(y), vx, vy))
    end,
    Dual{Tx}(value(x) / y, partials(x) / y),
    begin
        v = value(y)
        divv = x / v
        Dual{Ty}(divv, -(divv / v) * partials(y))
    end
)

# exponentiation #
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
                return Dual{Txy}(expv, new_partials)
            end,
            begin
                v = value(x)
                expv = ($f)(v, y)
                deriv = y * ($f)(v, y - 1)
                return Dual{Tx}(expv, deriv * partials(x))
            end,
            begin
                v = value(y)
                expv = ($f)(x, v)
                deriv = expv*log(x)
                return Dual{Ty}(expv, deriv * partials(y))
            end
        )
    end
end

# hypot #
#-------#

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
    calc_hypot(x, y, z, Txyz),
    calc_hypot(x, y, z, Txy),
    calc_hypot(x, y, z, Txz),
    calc_hypot(x, y, z, Tyz),
    calc_hypot(x, y, z, Tx),
    calc_hypot(x, y, z, Ty),
    calc_hypot(x, y, z, Tz),
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
        Dual{T}(v, $ex)
    end
end

@define_ternary_dual_op(
    Base.fma,
    calc_fma_xyz(x, y, z),                         # xyz_body
    calc_fma_xy(x, y, z),                          # xy_body
    calc_fma_xz(x, y, z),                          # xz_body
    Base.fma(y, x, z),                             # yz_body
    Dual{Tx}(fma(value(x), y, z), partials(x) * y), # x_body
    Base.fma(y, x, z),                              # y_body
    Dual{Tz}(fma(x, y, value(z)), partials(z))      # z_body
)

# muladd #
#--------#

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
        Dual{T}(v, $ex)
    end
end

@define_ternary_dual_op(
    Base.muladd,
    calc_muladd_xyz(x, y, z),                         # xyz_body
    calc_muladd_xy(x, y, z),                          # xy_body
    calc_muladd_xz(x, y, z),                          # xz_body
    Base.muladd(y, x, z),                             # yz_body
    Dual{Tx}(muladd(value(x), y, z), partials(x) * y), # x_body
    Base.muladd(y, x, z),                             # y_body
    Dual{Tz}(muladd(x, y, value(z)), partials(z))      # z_body
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

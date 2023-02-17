########
# Dual #
########

"""
    ForwardDiff.can_dual(V::Type)

Determines whether the type V is allowed as the scalar type in a
Dual. By default, only `<:Real` types are allowed.
"""
can_dual(::Type{<:Real}) = true
can_dual(::Type) = false

struct Dual{T,V,N} <: Real
    value::V
    partials::Partials{N,V}
    function Dual{T, V, N}(value::V, partials::Partials{N, V}) where {T, V, N}
        can_dual(V) || throw_cannot_dual(V)
        new{T, V, N}(value, partials)
    end
end

##########
# Traits #
##########
Base.ArithmeticStyle(::Type{<:Dual{T,V}}) where {T,V} = Base.ArithmeticStyle(V)

##############
# Exceptions #
##############

struct DualMismatchError{A,B} <: Exception
    a::A
    b::B
end

Base.showerror(io::IO, e::DualMismatchError{A,B}) where {A,B} =
    print(io, "Cannot determine ordering of Dual tags $(e.a) and $(e.b)")

@noinline function throw_cannot_dual(V::Type)
    throw(ArgumentError("Cannot create a dual over scalar type $V." *
        " If the type behaves as a scalar, define ForwardDiff.can_dual(::Type{$V}) = true."))
end

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

@inline Dual{T}(value::V, partials::Partials{N,V}) where {T,N,V} = Dual{T,V,N}(value, partials)

@inline function Dual{T}(value::A, partials::Partials{N,B}) where {T,N,A,B}
    C = promote_type(A, B)
    return Dual{T}(convert(C, value), convert(Partials{N,C}, partials))
end

@inline Dual{T}(value, partials::Tuple) where {T} = Dual{T}(value, Partials(partials))
@inline Dual{T}(value, partials::Tuple{}) where {T} = Dual{T}(value, Partials{0,typeof(value)}(partials))
@inline Dual{T}(value) where {T} = Dual{T}(value, ())
@inline Dual{T}(x::Dual{T}) where {T} = Dual{T}(x, ())
@inline Dual{T}(value, partial1, partials...) where {T} = Dual{T}(value, tuple(partial1, partials...))
@inline Dual{T}(value::V, ::Chunk{N}, p::Val{i}) where {T,V,N,i} = Dual{T}(value, single_seed(Partials{N,V}, p))
@inline Dual(args...) = Dual{Nothing}(args...)

# we define these special cases so that the "constructor <--> convert" pun holds for `Dual`
@inline Dual{T,V,N}(x::Dual{T,V,N}) where {T,V,N} = x
@inline Dual{T,V,N}(x) where {T,V,N} = convert(Dual{T,V,N}, x)
@inline Dual{T,V,N}(x::Number) where {T,V,N} = convert(Dual{T,V,N}, x)
@inline Dual{T,V}(x) where {T,V} = convert(Dual{T,V}, x)

##############################
# Utility/Accessor Functions #
##############################

@inline value(x) = x
@inline value(d::Dual) = d.value

@inline value(::Type{T}, x) where T = x
@inline value(::Type{T}, d::Dual{T}) where T = value(d)
@inline function value(::Type{T}, d::Dual{S}) where {T,S}
    if S ≺ T
        d
    else
        throw(DualMismatchError(T,S))
    end
end

@inline partials(x) = Partials{0,typeof(x)}(tuple())
@inline partials(d::Dual) = d.partials
@inline partials(x, i...) = zero(x)
@inline Base.@propagate_inbounds partials(d::Dual, i) = d.partials[i]
@inline Base.@propagate_inbounds partials(d::Dual, i, j) = partials(d, i).partials[j]
@inline Base.@propagate_inbounds partials(d::Dual, i, j, k...) = partials(partials(d, i, j), k...)

@inline Base.@propagate_inbounds partials(::Type{T}, x, i...) where T = partials(x, i...)
@inline Base.@propagate_inbounds partials(::Type{T}, d::Dual{T}, i...) where T = partials(d, i...)
@inline function partials(::Type{T}, d::Dual{S}, i...) where {T,S}
    if S ≺ T
        zero(d)
    else
        throw(DualMismatchError(T,S))
    end
end


@inline npartials(::Dual{T,V,N}) where {T,V,N} = N
@inline npartials(::Type{Dual{T,V,N}}) where {T,V,N} = N

@inline order(::Type{V}) where {V} = 0
@inline order(::Type{Dual{T,V,N}}) where {T,V,N} = 1 + order(V)

@inline valtype(::V) where {V} = V
@inline valtype(::Type{V}) where {V} = V
@inline valtype(::Dual{T,V,N}) where {T,V,N} = V
@inline valtype(::Type{Dual{T,V,N}}) where {T,V,N} = V

@inline tagtype(::V) where {V} = Nothing
@inline tagtype(::Type{V}) where {V} = Nothing
@inline tagtype(::Dual{T,V,N}) where {T,V,N} = T
@inline tagtype(::Type{Dual{T,V,N}}) where {T,V,N} = T

####################################
# N-ary Operation Definition Tools #
####################################

macro define_binary_dual_op(f, xy_body, x_body, y_body)
    FD = ForwardDiff
    defs = quote
        @inline $(f)(x::$FD.Dual{Txy}, y::$FD.Dual{Txy}) where {Txy} = $xy_body
        @inline $(f)(x::$FD.Dual{Tx}, y::$FD.Dual{Ty}) where {Tx,Ty} = Ty ≺ Tx ? $x_body : $y_body
    end
    for R in AMBIGUOUS_TYPES
        expr = quote
            @inline $(f)(x::$FD.Dual{Tx}, y::$R) where {Tx} = $x_body
            @inline $(f)(x::$R, y::$FD.Dual{Ty}) where {Ty} = $y_body
        end
        append!(defs.args, expr.args)
    end
    return esc(defs)
end

macro define_ternary_dual_op(f, xyz_body, xy_body, xz_body, yz_body, x_body, y_body, z_body)
    FD = ForwardDiff
    defs = quote
        @inline $(f)(x::$FD.Dual{Txyz}, y::$FD.Dual{Txyz}, z::$FD.Dual{Txyz}) where {Txyz} = $xyz_body
        @inline $(f)(x::$FD.Dual{Txy}, y::$FD.Dual{Txy}, z::$FD.Dual{Tz}) where {Txy,Tz} = Tz ≺ Txy ? $xy_body : $z_body
        @inline $(f)(x::$FD.Dual{Txz}, y::$FD.Dual{Ty}, z::$FD.Dual{Txz}) where {Txz,Ty} = Ty ≺ Txz ? $xz_body : $y_body
        @inline $(f)(x::$FD.Dual{Tx}, y::$FD.Dual{Tyz}, z::$FD.Dual{Tyz}) where {Tx,Tyz} = Tyz ≺ Tx ? $x_body  : $yz_body
        @inline function $(f)(x::$FD.Dual{Tx}, y::$FD.Dual{Ty}, z::$FD.Dual{Tz}) where {Tx,Ty,Tz}
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
            @inline $(f)(x::$FD.Dual{Txy}, y::$FD.Dual{Txy}, z::$R) where {Txy} = $xy_body
            @inline $(f)(x::$FD.Dual{Tx}, y::$FD.Dual{Ty}, z::$R)  where {Tx, Ty} = Ty ≺ Tx ? $x_body : $y_body
            @inline $(f)(x::$FD.Dual{Txz}, y::$R, z::$FD.Dual{Txz}) where {Txz} = $xz_body
            @inline $(f)(x::$FD.Dual{Tx}, y::$R, z::$FD.Dual{Tz}) where {Tx,Tz} = Tz ≺ Tx ? $x_body : $z_body
            @inline $(f)(x::$R, y::$FD.Dual{Tyz}, z::$FD.Dual{Tyz}) where {Tyz} = $yz_body
            @inline $(f)(x::$R, y::$FD.Dual{Ty}, z::$FD.Dual{Tz}) where {Ty,Tz} = Tz ≺ Ty ? $y_body : $z_body
        end
        append!(defs.args, expr.args)
        for Q in AMBIGUOUS_TYPES
            Q === R && continue
            expr = quote
                @inline $(f)(x::$FD.Dual{Tx}, y::$R, z::$Q) where {Tx} = $x_body
                @inline $(f)(x::$R, y::$FD.Dual{Ty}, z::$Q) where {Ty} = $y_body
                @inline $(f)(x::$R, y::$Q, z::$FD.Dual{Tz}) where {Tz} = $z_body
            end
            append!(defs.args, expr.args)
        end
        expr = quote
            @inline $(f)(x::$FD.Dual{Tx}, y::$R, z::$R) where {Tx} = $x_body
            @inline $(f)(x::$R, y::$FD.Dual{Ty}, z::$R) where {Ty} = $y_body
            @inline $(f)(x::$R, y::$R, z::$FD.Dual{Tz}) where {Tz} = $z_body
        end
        append!(defs.args, expr.args)
    end
    return esc(defs)
end

# Support complex-valued functions such as `hankelh1`
function dual_definition_retval(::Val{T}, val::Real, deriv::Real, partial::Partials) where {T}
    return Dual{T}(val, deriv * partial)
end
function dual_definition_retval(::Val{T}, val::Real, deriv1::Real, partial1::Partials, deriv2::Real, partial2::Partials) where {T}
    return Dual{T}(val, _mul_partials(partial1, partial2, deriv1, deriv2))
end
function dual_definition_retval(::Val{T}, val::Complex, deriv::Union{Real,Complex}, partial::Partials) where {T}
    reval, imval = reim(val)
    if deriv isa Real
        p = deriv * partial
        return Complex(Dual{T}(reval, p), Dual{T}(imval, zero(p)))
    else
        rederiv, imderiv = reim(deriv)
        return Complex(Dual{T}(reval, rederiv * partial), Dual{T}(imval, imderiv * partial))
    end
end
function dual_definition_retval(::Val{T}, val::Complex, deriv1::Union{Real,Complex}, partial1::Partials, deriv2::Union{Real,Complex}, partial2::Partials) where {T}
    reval, imval = reim(val)
    if deriv1 isa Real && deriv2 isa Real
        p = _mul_partials(partial1, partial2, deriv1, deriv2)
        return Complex(Dual{T}(reval, p), Dual{T}(imval, zero(p)))
    else
        rederiv1, imderiv1 = reim(deriv1)
        rederiv2, imderiv2 = reim(deriv2)
        return Complex(
            Dual{T}(reval, _mul_partials(partial1, partial2, rederiv1, rederiv2)),
            Dual{T}(imval, _mul_partials(partial1, partial2, imderiv1, imderiv2)),
        )
    end
end

function unary_dual_definition(M, f)
    FD = ForwardDiff
    Mf = M == :Base ? f : :($M.$f)
    work = qualified_cse!(quote
        val = $Mf(x)
        deriv = $(DiffRules.diffrule(M, f, :x))
    end)
    return quote
        @inline function $M.$f(d::$FD.Dual{T}) where T
            x = $FD.value(d)
            $work
            return $FD.dual_definition_retval(Val{T}(), val, deriv, $FD.partials(d))
        end
    end
end

function binary_dual_definition(M, f)
    FD = ForwardDiff
    dvx, dvy = DiffRules.diffrule(M, f, :vx, :vy)
    Mf = M == :Base ? f : :($M.$f)
    xy_work = qualified_cse!(quote
        val = $Mf(vx, vy)
        dvx = $dvx
        dvy = $dvy
    end)
    dvx, _ = DiffRules.diffrule(M, f, :vx, :y)
    x_work = qualified_cse!(quote
        val = $Mf(vx, y)
        dvx = $dvx
    end)
    _, dvy = DiffRules.diffrule(M, f, :x, :vy)
    y_work = qualified_cse!(quote
        val = $Mf(x, vy)
        dvy = $dvy
    end)
    expr = quote
        $FD.@define_binary_dual_op(
            $M.$f,
            begin
                vx, vy = $FD.value(x), $FD.value(y)
                $xy_work
                return $FD.dual_definition_retval(Val{Txy}(), val, dvx, $FD.partials(x), dvy, $FD.partials(y))
            end,
            begin
                vx = $FD.value(x)
                $x_work
                return $FD.dual_definition_retval(Val{Tx}(), val, dvx, $FD.partials(x))
            end,
            begin
                vy = $FD.value(y)
                $y_work
                return $FD.dual_definition_retval(Val{Ty}(), val, dvy, $FD.partials(y))
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

# The `base` keyword was added in Julia 1.8:
# https://github.com/JuliaLang/julia/pull/42428
if VERSION < v"1.8.0-DEV.725"
    Base.precision(d::Dual) = precision(value(d))
    Base.precision(::Type{D}) where {D<:Dual} = precision(valtype(D))
else
    Base.precision(d::Dual; base::Integer=2) = precision(value(d); base=base)
    function Base.precision(::Type{D}; base::Integer=2) where {D<:Dual}
        precision(valtype(D); base=base)
    end
end

function Base.nextfloat(d::ForwardDiff.Dual{T,V,N}) where {T,V,N}
    ForwardDiff.Dual{T}(nextfloat(d.value), d.partials)
end

function Base.prevfloat(d::ForwardDiff.Dual{T,V,N}) where {T,V,N}
    ForwardDiff.Dual{T}(prevfloat(d.value), d.partials)
end

Base.rtoldefault(::Type{D}) where {D<:Dual} = Base.rtoldefault(valtype(D))

Base.floor(::Type{R}, d::Dual) where {R<:Real} = floor(R, value(d))
Base.floor(d::Dual) = floor(value(d))

Base.ceil(::Type{R}, d::Dual) where {R<:Real} = ceil(R, value(d))
Base.ceil(d::Dual) = ceil(value(d))

Base.trunc(::Type{R}, d::Dual) where {R<:Real} = trunc(R, value(d))
Base.trunc(d::Dual) = trunc(value(d))

Base.round(::Type{R}, d::Dual) where {R<:Real} = round(R, value(d))
Base.round(d::Dual) = round(value(d))

Base.fld(x::Dual, y::Dual) = fld(value(x), value(y))

Base.cld(x::Dual, y::Dual) = cld(value(x), value(y))

Base.exponent(x::Dual) = exponent(value(x))

if VERSION ≥ v"1.4"
    Base.div(x::Dual, y::Dual, r::RoundingMode) = div(value(x), value(y), r)
else
    Base.div(x::Dual, y::Dual) = div(value(x), value(y))
end

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

@inline function Base.Int(d::Dual)
    all(iszero, partials(d)) || throw(InexactError(:Int, Int, d))
    Int(value(d))
end
@inline function Base.Integer(d::Dual)
    all(iszero, partials(d)) || throw(InexactError(:Integer, Integer, d))
    Integer(value(d))
end

@inline Random.rand(rng::AbstractRNG, d::Dual) = rand(rng, value(d))
@inline Random.rand(::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(rand(V), zero(Partials{N,V}))
@inline Random.rand(rng::AbstractRNG, ::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(rand(rng, V), zero(Partials{N,V}))
@inline Random.randn(::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(randn(V), zero(Partials{N,V}))
@inline Random.randn(rng::AbstractRNG, ::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(randn(rng, V), zero(Partials{N,V}))
@inline Random.randexp(::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(randexp(V), zero(Partials{N,V}))
@inline Random.randexp(rng::AbstractRNG, ::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(randexp(rng, V), zero(Partials{N,V}))

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

function Base.promote_rule(::Type{Dual{T1,V1,N1}},
                                      ::Type{Dual{T2,V2,N2}}) where {T1,V1,N1,T2,V2,N2}
    # V1 and V2 might themselves be Dual types
    if T2 ≺ T1
        Dual{T1,promote_type(V1,Dual{T2,V2,N2}),N1}
    else
        Dual{T2,promote_type(V2,Dual{T1,V1,N1}),N2}
    end
end

function Base.promote_rule(::Type{Dual{T,A,N}},
                           ::Type{Dual{T,B,N}}) where {T,A,B,N}
    return Dual{T,promote_type(A, B),N}
end

for R in (Irrational, Real, BigFloat, Bool)
    if isconcretetype(R) # issue #322
        @eval begin
            Base.promote_rule(::Type{$R}, ::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T,promote_type($R, V),N}
            Base.promote_rule(::Type{Dual{T,V,N}}, ::Type{$R}) where {T,V,N} = Dual{T,promote_type(V, $R),N}
        end
    else
        @eval begin
            Base.promote_rule(::Type{R}, ::Type{Dual{T,V,N}}) where {R<:$R,T,V,N} = Dual{T,promote_type(R, V),N}
            Base.promote_rule(::Type{Dual{T,V,N}}, ::Type{R}) where {T,V,N,R<:$R} = Dual{T,promote_type(V, R),N}
        end
    end
end

@inline Base.convert(::Type{Dual{T,V,N}}, d::Dual{T}) where {T,V,N} = Dual{T}(V(value(d)), convert(Partials{N,V}, partials(d)))
@inline Base.convert(::Type{Dual{T,V,N}}, x) where {T,V,N} = Dual{T}(V(x), zero(Partials{N,V}))
@inline Base.convert(::Type{Dual{T,V,N}}, x::Number) where {T,V,N} = Dual{T}(V(x), zero(Partials{N,V}))
Base.convert(::Type{D}, d::D) where {D<:Dual} = d

Base.float(::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T,float(V),N}
Base.float(d::Dual) = convert(float(typeof(d)), d)

###################################
# General Mathematical Operations #
###################################

for (M, f, arity) in DiffRules.diffrules(filter_modules = nothing)
    if (M, f) in ((:Base, :^), (:NaNMath, :pow), (:Base, :/), (:Base, :+), (:Base, :-), (:Base, :sin), (:Base, :cos))
        continue  # Skip methods which we define elsewhere.
    elseif !(isdefined(@__MODULE__, M) && isdefined(getfield(@__MODULE__, M), f))
        continue  # Skip rules for methods not defined in the current scope
    end
    if arity == 1
        eval(unary_dual_definition(M, f))
    elseif arity == 2
        eval(binary_dual_definition(M, f))
    else
        # error("ForwardDiff currently only knows how to autogenerate Dual definitions for unary and binary functions.")
        # However, the presence of N-ary rules need not cause any problems here, they can simply be ignored.
    end
end

#################
# Special Cases #
#################

# +/- #
#-----#

@define_binary_dual_op(
    Base.:+,
    begin
        vx, vy = value(x), value(y)
        Dual{Txy}(vx + vy, partials(x) + partials(y))
    end,
    Dual{Tx}(value(x) + y, partials(x)),
    Dual{Ty}(x + value(y), partials(y))
)

@define_binary_dual_op(
    Base.:-,
    begin
        vx, vy = value(x), value(y)
        Dual{Txy}(vx - vy, partials(x) - partials(y))
    end,
    Dual{Tx}(value(x) - y, partials(x)),
    Dual{Ty}(x - value(y), -partials(y))
)

@inline Base.:-(d::Dual{T}) where {T} = Dual{T}(-value(d), -partials(d))

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
                if isconstant(y)
                    logval = one(expv)
                elseif iszero(vx) && vy > 0
                    logval = zero(vx)
                else
                    logval = expv * log(vx)
                end
                new_partials = _mul_partials(partials(x), partials(y), powval, logval)
                return Dual{Txy}(expv, new_partials)
            end,
            begin
                v = value(x)
                expv = ($f)(v, y)
                if y == zero(y) || iszero(partials(x))
                    new_partials = zero(partials(x))
                else
                    new_partials = partials(x) * y * ($f)(v, y - 1)
                end
                return Dual{Tx}(expv, new_partials)
            end,
            begin
                v = value(y)
                expv = ($f)(x, v)
                deriv = (iszero(x) && v > 0) ? zero(expv) : expv*log(x)
                return Dual{Ty}(expv, deriv * partials(y))
            end
        )
    end
end

@inline Base.literal_pow(::typeof(^), x::Dual{T}, ::Val{0}) where {T} =
    Dual{T}(one(value(x)), zero(partials(x)))

for y in 1:3
    @eval @inline function Base.literal_pow(::typeof(^), x::Dual{T}, ::Val{$y}) where {T}
        v = value(x)
        expv = v^$y
        deriv = $y * v^$(y - 1)
        return Dual{T}(expv, deriv * partials(x))
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

@generated function calc_fma_xyz(x::Dual{T,<:Any,N},
                                 y::Dual{T,<:Any,N},
                                 z::Dual{T,<:Any,N}) where {T,N}
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

@generated function calc_fma_xz(x::Dual{T,<:Any,N},
                                y::Real,
                                z::Dual{T,<:Any,N}) where {T,N}
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

@generated function calc_muladd_xyz(x::Dual{T,<:Any,N},
                                    y::Dual{T,<:Any,N},
                                    z::Dual{T,<:Any,N}) where {T,N}
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

@generated function calc_muladd_xz(x::Dual{T,<:Any,N},
                                   y::Real,
                                   z::Dual{T,<:Any,N}) where {T,N}
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

# sin/cos #
#--------#
function Base.sin(d::Dual{T}) where T
    s, c = sincos(value(d))
    return Dual{T}(s, c * partials(d))
end

function Base.cos(d::Dual{T}) where T
    s, c = sincos(value(d))
    return Dual{T}(c, -s * partials(d))
end

@inline function Base.sincos(d::Dual{T}) where T
    sd, cd = sincos(value(d))
    return (Dual{T}(sd, cd * partials(d)), Dual{T}(cd, -sd * partials(d)))
end

# sincospi #
#----------#

if VERSION >= v"1.6.0-DEV.292"
    @inline function Base.sincospi(d::Dual{T}) where T
        sd, cd = sincospi(value(d))
        return (Dual{T}(sd, cd * π * partials(d)), Dual{T}(cd, -sd * π * partials(d)))
    end
end

# Symmetric eigvals #
#-------------------#

function LinearAlgebra.eigvals(A::Symmetric{<:Dual{Tg,T,N}}) where {Tg,T<:Real,N}
    λ,Q = eigen(Symmetric(value.(parent(A))))
    parts = ntuple(j -> diag(Q' * getindex.(partials.(A), j) * Q), N)
    Dual{Tg}.(λ, tuple.(parts...))
end

function LinearAlgebra.eigvals(A::Hermitian{<:Complex{<:Dual{Tg,T,N}}}) where {Tg,T<:Real,N}
    λ,Q = eigen(Hermitian(value.(real.(parent(A))) .+ im .* value.(imag.(parent(A)))))
    parts = ntuple(j -> diag(real.(Q' * (getindex.(partials.(real.(A)) .+ im .* partials.(imag.(A)), j)) * Q)), N)
    Dual{Tg}.(λ, tuple.(parts...))
end

function LinearAlgebra.eigvals(A::SymTridiagonal{<:Dual{Tg,T,N}}) where {Tg,T<:Real,N}
    λ,Q = eigen(SymTridiagonal(value.(parent(A).dv),value.(parent(A).ev)))
    parts = ntuple(j -> diag(Q' * getindex.(partials.(A), j) * Q), N)
    Dual{Tg}.(λ, tuple.(parts...))
end

# A ./ (λ - λ') but with diag special cased
function _lyap_div!(A, λ)
    for (j,μ) in enumerate(λ), (k,λ) in enumerate(λ)
        if k ≠ j
            A[k,j] /= μ - λ
        end
    end
    A
end

function LinearAlgebra.eigen(A::Symmetric{<:Dual{Tg,T,N}}) where {Tg,T<:Real,N}
    λ = eigvals(A)
    _,Q = eigen(Symmetric(value.(parent(A))))
    parts = ntuple(j -> Q*_lyap_div!(Q' * getindex.(partials.(A), j) * Q - Diagonal(getindex.(partials.(λ), j)), value.(λ)), N)
    Eigen(λ,Dual{Tg}.(Q, tuple.(parts...)))
end

function LinearAlgebra.eigen(A::SymTridiagonal{<:Dual{Tg,T,N}}) where {Tg,T<:Real,N}
    λ = eigvals(A)
    _,Q = eigen(SymTridiagonal(value.(parent(A))))
    parts = ntuple(j -> Q*_lyap_div!(Q' * getindex.(partials.(A), j) * Q - Diagonal(getindex.(partials.(λ), j)), value.(λ)), N)
    Eigen(λ,Dual{Tg}.(Q, tuple.(parts...)))
end

# Functions in SpecialFunctions which return tuples #
# Their derivatives are not defined in DiffRules    #
#---------------------------------------------------#

function SpecialFunctions.logabsgamma(d::Dual{T,<:Real}) where {T}
    x = value(d)
    y, s = SpecialFunctions.logabsgamma(x)
    return (Dual{T}(y, SpecialFunctions.digamma(x) * partials(d)), s)
end

# Derivatives wrt to first parameter and precision setting are not supported
function SpecialFunctions.gamma_inc(a::Real, d::Dual{T,<:Real}, ind::Integer) where {T}
    x = value(d)
    p, q = SpecialFunctions.gamma_inc(a, x, ind)
    ∂p = exp(-x) * x^(a - 1) / SpecialFunctions.gamma(a) * partials(d)
    return (Dual{T}(p, ∂p), Dual{T}(q, -∂p))
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

for op in (:(Base.typemin), :(Base.typemax), :(Base.floatmin), :(Base.floatmax))
    @eval function $op(::Type{ForwardDiff.Dual{T,V,N}}) where {T,V,N}
        ForwardDiff.Dual{T,V,N}($op(V))
    end
end

if VERSION >= v"1.6.0-rc1"
    Printf.tofloat(d::Dual) = Printf.tofloat(value(d))
end

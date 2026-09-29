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
        T isa Type || throw_invalid_tag(T)
        check_tag_order(T, value)
        foreach(p -> check_tag_order(T, p), partials.values)
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

@noinline function throw_invalid_tag(T)
    throw(ArgumentError(lazy"The tag of a Dual must be a type, got $(repr(T))."))
end

@noinline function throw_cannot_dual(V::Type)
    throw(ArgumentError(lazy"Cannot create a dual over scalar type $V. If the type behaves as a scalar, define ForwardDiff.can_dual(::Type{$V}) = true."))
end

"""
    ForwardDiff.≺(a, b)::Bool

Determines the order in which tagged `Dual` objects are stored. If true, then `Dual{b}`
objects will appear outside `Dual{a}` objects.

Values and partials with respect to a tag can be extracted irrespective of this order.
"""
function ≺ end

@noinline function throw_tag_order(T, S)
    throw(ArgumentError(lazy"Cannot store a Dual with tag $T outside a Dual with tag $S, since $T ≺ $S."))
end

@noinline function throw_same_tag(T)
    throw(ArgumentError(lazy"Cannot store a Dual with tag $T inside a Dual with the same tag."))
end

# Tags are strictly decreasing inwards, hence unique, so it suffices to check the next layer
@inline check_tag_order(::Type{T}, x) where {T} = nothing
@inline function check_tag_order(::Type{T}, ::Dual{S}) where {T,S}
    if S === T
        throw_same_tag(T)
    elseif T ≺ S
        throw_tag_order(T, S)
    end
    return nothing
end

################
# Constructors #
################

# Converts the arguments of the constructors to a value and partials of the same type
@inline dual_args(value::V, partials::Partials{N,V}) where {N,V} = (value, partials)
@inline function dual_args(value::A, partials::Partials{N,B}) where {N,A,B}
    C = promote_type(A, B)
    return (convert(C, value), convert(Partials{N,C}, partials))
end
@inline dual_args(value, partials::Tuple) = dual_args(value, Partials(partials))
@inline dual_args(value, partials::Tuple{}) = dual_args(value, Partials{0,typeof(value)}(partials))
@inline dual_args(value) = dual_args(value, ())
@inline dual_args(value, partial1, partials...) = dual_args(value, tuple(partial1, partials...))
@inline dual_args(value::V, ::Chunk{N}, p::Val{i}) where {V,N,i} = dual_args(value, single_seed(Partials{N,V}, p))

@inline function Dual{T}(args...) where {T}
    value, partials = dual_args(args...)
    return Dual{T,eltype(partials),length(partials)}(value, partials)
end

@inline function Dual(args...)
    value, partials = dual_args(args...)
    return Dual{Tag{Nothing,eltype(partials)}}(value, partials)
end

# we define these special cases so that the "constructor <--> convert" pun holds for `Dual`
@inline Dual{T,V,N}(x::Dual{T,V,N}) where {T,V,N} = x
@inline Dual{T,V,N}(x) where {T,V,N} = convert(Dual{T,V,N}, x)
@inline Dual{T,V,N}(x::Number) where {T,V,N} = convert(Dual{T,V,N}, x)
@inline Dual{T,V}(x) where {T,V} = convert(Dual{T,V}, x)

# Fix method ambiguity issue by adapting the definition in Base to `Dual`s
Dual{T,V,N}(x::Base.TwicePrecision) where {T,V,N} =
    (Dual{T,V,N}(x.hi) + Dual{T,V,N}(x.lo))::Dual{T,V,N}

##############################
# Utility/Accessor Functions #
##############################

# Whether a `Dual` with tag `T` occurs anywhere in the nesting of `D`
@inline hastag(::Type{T}, ::Type) where {T} = false
@inline hastag(::Type{T}, ::Type{Dual{S,V,N}}) where {T,S,V,N} = S === T || hastag(T, V)

@inline map_partials(f::F, ::Type{W}, p::Partials{N}) where {F,W,N} = Partials{N,W}(map(f, p.values))

# Extraction w.r.t. `T` descends through layers with other tags, so it does not depend on
# the order in which the layers are nested. Without a layer with tag `T` the value is the
# number itself and the partials are zero.
@inline value(::Type{T}, x) where T = x
@inline value(::Type{T}, d::Dual{T}) where T = d.value
@inline function value(::Type{T}, d::Dual{S,V}) where {T,S,V}
    if hastag(T, typeof(d))
        Dual{S}(value(T, value(S, d)), map_partials(p -> value(T, p), valtype(T, V), partials(S, d)))
    else
        d
    end
end

@inline partials(::Type{T}, x) where T = Partials{0,typeof(x)}(tuple())
@inline partials(::Type{T}, x, i) where T = zero(x)
@inline partials(::Type{T}, d::Dual{T}) where T = d.partials
@inline Base.@propagate_inbounds partials(::Type{T}, d::Dual{T}, i) where T = d.partials[i]
@inline Base.@propagate_inbounds function partials(::Type{T}, d::Dual{S,V}, i) where {T,S,V}
    if hastag(T, typeof(d))
        Dual{S}(partials(T, value(S, d), i), map_partials(p -> partials(T, p, i), valtype(T, V), partials(S, d)))
    else
        zero(d)
    end
end
@inline function partials(::Type{T}, d::Dual) where {T}
    if hastag(T, typeof(d))
        Partials{npartials(T, typeof(d)),valtype(T, typeof(d))}(ntuple(i -> partials(T, d, i), Val(npartials(T, typeof(d)))))
    else
        Partials{0,typeof(d)}(tuple())
    end
end

@inline npartials(::Type{T}, ::Type) where {T} = 0
@inline npartials(::Type{T}, ::Type{Dual{T,V,N}}) where {T,V,N} = N
@inline npartials(::Type{T}, ::Type{Dual{S,V,N}}) where {T,S,V,N} = npartials(T, V)

@inline order(::Type{V}) where {V} = 0
@inline order(::Type{Dual{T,V,N}}) where {T,V,N} = 1 + order(V)

@inline valtype(::V) where {V} = V
@inline valtype(::Type{V}) where {V} = V
@inline valtype(::Dual{T,V,N}) where {T,V,N} = V
@inline valtype(::Type{Dual{T,V,N}}) where {T,V,N} = V

@inline valtype(::Type{T}, ::V) where {T,V} = valtype(T, V)
@inline valtype(::Type, ::Type{V}) where {V} = V
@inline valtype(::Type{T}, ::Type{Dual{T,V,N}}) where {T,V,N} = V
@inline valtype(::Type{T}, ::Type{Dual{S,V,N}}) where {T,S,V,N} = Dual{S,valtype(T, V),N}

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
@inline function dual_definition_retval(::Val{T}, val::Real, deriv::Real, partial::Partials) where {T}
    return Dual{T}(val, deriv * partial)
end
@inline function dual_definition_retval(::Val{T}, val::Real, deriv1::Real, partial1::Partials, deriv2::Real, partial2::Partials) where {T}
    return Dual{T}(val, _mul_partials(partial1, partial2, deriv1, deriv2))
end
@inline function dual_definition_retval(::Val{T}, val::Complex, deriv::Union{Real,Complex}, partial::Partials) where {T}
    reval, imval = reim(val)
    if deriv isa Real
        p = deriv * partial
        return Complex(Dual{T}(reval, p), Dual{T}(imval, zero(p)))
    else
        rederiv, imderiv = reim(deriv)
        return Complex(Dual{T}(reval, rederiv * partial), Dual{T}(imval, imderiv * partial))
    end
end
@inline function dual_definition_retval(::Val{T}, val::Complex, deriv1::Union{Real,Complex}, partial1::Partials, deriv2::Union{Real,Complex}, partial2::Partials) where {T}
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
            x = $FD.value(T, d)
            $work
            return $FD.dual_definition_retval(Val{T}(), val, deriv, $FD.partials(T, d))
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
                vx, vy = $FD.value(Txy, x), $FD.value(Txy, y)
                $xy_work
                return $FD.dual_definition_retval(Val{Txy}(), val, dvx, $FD.partials(Txy, x), dvy, $FD.partials(Txy, y))
            end,
            begin
                vx = $FD.value(Tx, x)
                $x_work
                return $FD.dual_definition_retval(Val{Tx}(), val, dvx, $FD.partials(Tx, x))
            end,
            begin
                vy = $FD.value(Ty, y)
                $y_work
                return $FD.dual_definition_retval(Val{Ty}(), val, dvy, $FD.partials(Ty, y))
            end
        )
    end
    return expr
end

#####################
# Generic Functions #
#####################

Base.copy(d::Dual) = d

Base.eps(d::Dual{T}) where {T} = eps(value(T, d))
Base.eps(::Type{D}) where {D<:Dual} = eps(valtype(D))

# The `base` keyword was added in Julia 1.8:
# https://github.com/JuliaLang/julia/pull/42428
Base.precision(d::Dual{T}; base::Integer=2) where {T} = precision(value(T, d); base=base)
function Base.precision(::Type{D}; base::Integer=2) where {D<:Dual}
    precision(valtype(D); base=base)
end

function Base.nextfloat(d::ForwardDiff.Dual{T,V,N}) where {T,V,N}
    ForwardDiff.Dual{T}(nextfloat(d.value), d.partials)
end

function Base.prevfloat(d::ForwardDiff.Dual{T,V,N}) where {T,V,N}
    ForwardDiff.Dual{T}(prevfloat(d.value), d.partials)
end

Base.rtoldefault(::Type{D}) where {D<:Dual} = Base.rtoldefault(valtype(D))

# Base derives floor/ceil/trunc/round from `round(x, ::RoundingMode)`:
# https://docs.julialang.org/en/v1/manual/interfaces/#man-rounding-interface
Base.round(d::Dual{T}, r::RoundingMode) where {T} = round(value(T, d), r)

# Julia 1.11 added the generic `f(::Type{T}, x)` fallbacks, so these can be
# dropped once 1.11 is the minimum supported version.
if VERSION < v"1.11"
    Base.floor(::Type{R}, d::Dual{T}) where {R<:Real,T} = floor(R, value(T, d))
    Base.ceil(::Type{R}, d::Dual{T}) where {R<:Real,T} = ceil(R, value(T, d))
    Base.trunc(::Type{R}, d::Dual{T}) where {R<:Real,T} = trunc(R, value(T, d))
    Base.round(::Type{R}, d::Dual{T}) where {R<:Real,T} = round(R, value(T, d))
end

Base.fld(x::Dual{Tx}, y::Dual{Ty}) where {Tx,Ty} = fld(value(Tx, x), value(Ty, y))

Base.cld(x::Dual{Tx}, y::Dual{Ty}) where {Tx,Ty} = cld(value(Tx, x), value(Ty, y))

Base.exponent(x::Dual{T}) where {T} = exponent(value(T, x))

Base.div(x::Dual{Tx}, y::Dual{Ty}, r::RoundingMode) where {Tx,Ty} = div(value(Tx, x), value(Ty, y), r)

Base.hash(d::Dual{T}, hsh::UInt) where {T} = hash(value(T, d), hsh)

function Base.read(io::IO, ::Type{Dual{T,V,N}}) where {T,V,N}
    value = read(io, V)
    partials = read(io, Partials{N,V})
    return Dual{T,V,N}(value, partials)
end

function Base.write(io::IO, d::Dual{T}) where {T}
    write(io, value(T, d))
    write(io, partials(T, d))
end

@inline Base.zero(d::Dual) = zero(typeof(d))
@inline Base.zero(::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(zero(V), zero(Partials{N,V}))

@inline Base.one(d::Dual) = one(typeof(d))
@inline Base.one(::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(one(V), zero(Partials{N,V}))

@inline function Base.Int(d::Dual{T}) where {T}
    all(iszero, partials(T, d)) || throw(InexactError(:Int, Int, d))
    Int(value(T, d))
end
@inline function Base.Integer(d::Dual{T}) where {T}
    all(iszero, partials(T, d)) || throw(InexactError(:Integer, Integer, d))
    Integer(value(T, d))
end

@inline Random.rand(rng::AbstractRNG, d::Dual{T}) where {T} = rand(rng, value(T, d))
@inline Random.rand(::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(rand(V), zero(Partials{N,V}))
@inline Random.rand(rng::AbstractRNG, ::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(rand(rng, V), zero(Partials{N,V}))
@inline Random.randn(::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(randn(V), zero(Partials{N,V}))
@inline Random.randn(rng::AbstractRNG, ::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(randn(rng, V), zero(Partials{N,V}))
@inline Random.randexp(::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(randexp(V), zero(Partials{N,V}))
@inline Random.randexp(rng::AbstractRNG, ::Type{Dual{T,V,N}}) where {T,V,N} = Dual{T}(randexp(rng, V), zero(Partials{N,V}))

# Predicates #
#------------#

isconstant(d::Dual{T}) where {T} = iszero(partials(T, d))

for pred in UNARY_PREDICATES
    @eval Base.$(pred)(d::Dual{T}) where {T} = $(pred)(value(T, d))
end

# Before PR#481 this loop ran over this list:
# BINARY_PREDICATES = Symbol[:isequal, :isless, :<, :>, :(==), :(<=), :(>=)]
# Not a minimal set, as Base defines some in terms of others.
@define_binary_dual_op(
    Base.:(<),
    (value(Txy, x) < value(Txy, y)) || (value(Txy, x) == value(Txy, y) && (partials(Txy, x) < partials(Txy, y))),
    (value(Tx, x) < y) || (value(Tx, x) == y && (partials(Tx, x) < zero(partials(Tx, x)))),
    (x < value(Ty, y)) || (x == value(Ty, y) && (zero(partials(Ty, y)) < partials(Ty, y))),
)
@define_binary_dual_op(
    Base.:(<=),
    (value(Txy, x) < value(Txy, y)) || (value(Txy, x) == value(Txy, y) && (partials(Txy, x) <= partials(Txy, y))),
    (value(Tx, x) < y) || (value(Tx, x) == y && (partials(Tx, x) <= zero(partials(Tx, x)))),
    (x < value(Ty, y)) || (x == value(Ty, y) && (zero(partials(Ty, y)) <= partials(Ty, y))),
)

@define_binary_dual_op(
    Base.isless,
    isless(value(Txy, x), value(Txy, y)) || (isequal(value(Txy, x), value(Txy, y)) && isless(partials(Txy, x), partials(Txy, y))),
    isless(value(Tx, x), y)        || (isequal(value(Tx, x), y) && isless(partials(Tx, x), zero(partials(Tx, x)))),
    isless(x, value(Ty, y))        || (isequal(x, value(Ty, y)) && isless(zero(partials(Ty, y)), partials(Ty, y))),
)

Base.iszero(x::Dual{T}) where {T} = iszero(value(T, x)) && iszero(partials(T, x))  # shortcut, equivalent to x == zero(x)

for pred in [:isequal, :(==)]
    @eval begin
        @define_binary_dual_op(
            Base.$(pred),
            $(pred)(value(Txy, x), value(Txy, y)) && $(pred)(partials(Txy, x), partials(Txy, y)),
            $(pred)(value(Tx, x), y)        && iszero(partials(Tx, x)),
            $(pred)(x, value(Ty, y))        && iszero(partials(Ty, y)),
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

# no common type for different numbers of partials, `promote_type` falls back to `typejoin`
Base.promote_rule(::Type{Dual{T,A,M}}, ::Type{Dual{T,B,N}}) where {T,A,B,M,N} = Union{}

for R in (AbstractIrrational, Real, BigFloat, Bool)
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

@inline Base.convert(::Type{Dual{T,V,N}}, d::Dual{T}) where {T,V,N} = Dual{T}(V(value(T, d)), convert(Partials{N,V}, partials(T, d)))
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
        vx, vy = value(Txy, x), value(Txy, y)
        Dual{Txy}(vx + vy, partials(Txy, x) + partials(Txy, y))
    end,
    Dual{Tx}(value(Tx, x) + y, partials(Tx, x)),
    Dual{Ty}(x + value(Ty, y), partials(Ty, y))
)

@define_binary_dual_op(
    Base.:-,
    begin
        vx, vy = value(Txy, x), value(Txy, y)
        Dual{Txy}(vx - vy, partials(Txy, x) - partials(Txy, y))
    end,
    Dual{Tx}(value(Tx, x) - y, partials(Tx, x)),
    Dual{Ty}(x - value(Ty, y), -partials(Ty, y))
)

@inline Base.:-(d::Dual{T}) where {T} = Dual{T}(-value(T, d), -partials(T, d))

# * #
#---#

@inline Base.:*(d::Dual{T}, x::Bool) where {T} = x ? d : (signbit(value(T, d))==0 ? zero(d) : -zero(d))
@inline Base.:*(x::Bool, d::Dual) = d * x

# / #
#---#

# We can't use the normal diffrule autogeneration for this because (x/y) === (x * (1/y))
# doesn't generally hold true for floating point; see issue #264
@define_binary_dual_op(
    Base.:/,
    begin
        vx, vy = value(Txy, x), value(Txy, y)
        Dual{Txy}(vx / vy, _div_partials(partials(Txy, x), partials(Txy, y), vx, vy))
    end,
    Dual{Tx}(value(Tx, x) / y, partials(Tx, x) / y),
    begin
        v = value(Ty, y)
        divv = x / v
        Dual{Ty}(divv, -(divv / v) * partials(Ty, y))
    end
)

# exponentiation #
#----------------#

for (f, log) in ((:(Base.:^), :(Base.log)), (:(NaNMath.pow), :(NaNMath.log)))
    @eval begin
        @define_binary_dual_op(
            $f,
            begin
                vx, vy = value(Txy, x), value(Txy, y)
                expv = ($f)(vx, vy)
                powval = vy * ($f)(vx, vy - 1)
                if isconstant(y)
                    logval = one(expv)
                elseif iszero(vx) && vy > 0
                    logval = zero(vx)
                else
                    logval = expv * ($log)(vx)
                end
                new_partials = _mul_partials(partials(Txy, x), partials(Txy, y), powval, logval)
                return Dual{Txy}(expv, new_partials)
            end,
            begin
                v = value(Tx, x)
                expv = ($f)(v, y)
                if y == zero(y) || iszero(partials(Tx, x))
                    new_partials = zero(partials(Tx, x))
                else
                    new_partials = partials(Tx, x) * y * ($f)(v, y - 1)
                end
                return Dual{Tx}(expv, new_partials)
            end,
            begin
                v = value(Ty, y)
                expv = ($f)(x, v)
                deriv = (iszero(x) && v > 0) ? zero(expv) : expv*($log)(oftype(expv, x))
                return Dual{Ty}(expv, deriv * partials(Ty, y))
            end
        )
    end
end

@inline Base.literal_pow(::typeof(^), x::Dual{T}, ::Val{0}) where {T} =
    Dual{T}(one(value(T, x)), zero(partials(T, x)))

for y in 1:3
    @eval @inline function Base.literal_pow(::typeof(^), x::Dual{T}, ::Val{$y}) where {T}
        v = value(T, x)
        expv = v^$y
        deriv = $y * v^$(y - 1)
        return Dual{T}(expv, deriv * partials(T, x))
    end
end

# hypot #
#-------#

@inline function calc_hypot(x, y, z, ::Type{T}) where T
    vx = value(T, x)
    vy = value(T, y)
    vz = value(T, z)
    h = hypot(vx, vy, vz)
    p = (vx / h) * partials(T, x) + (vy / h) * partials(T, y) + (vz / h) * partials(T, z)
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
    ex = Expr(:tuple, [:(fma(value(T, x), partials(T, y)[$i], fma(value(T, y), partials(T, x)[$i], partials(T, z)[$i]))) for i in 1:N]...)
    return quote
        $(Expr(:meta, :inline))
        v = fma(value(T, x), value(T, y), value(T, z))
        return Dual{T}(v, $ex)
    end
end

@inline function calc_fma_xy(x::Dual{T}, y::Dual{T}, z::Real) where T
    vx, vy = value(T, x), value(T, y)
    result = fma(vx, vy, z)
    return Dual{T}(result, _mul_partials(partials(T, x), partials(T, y), vy, vx))
end

@generated function calc_fma_xz(x::Dual{T,<:Any,N},
                                y::Real,
                                z::Dual{T,<:Any,N}) where {T,N}
    ex = Expr(:tuple, [:(fma(partials(T, x)[$i], y,  partials(T, z)[$i])) for i in 1:N]...)
    return quote
        $(Expr(:meta, :inline))
        v = fma(value(T, x), y, value(T, z))
        Dual{T}(v, $ex)
    end
end

@define_ternary_dual_op(
    Base.fma,
    calc_fma_xyz(x, y, z),                         # xyz_body
    calc_fma_xy(x, y, z),                          # xy_body
    calc_fma_xz(x, y, z),                          # xz_body
    Base.fma(y, x, z),                             # yz_body
    Dual{Tx}(fma(value(Tx, x), y, z), partials(Tx, x) * y), # x_body
    Base.fma(y, x, z),                              # y_body
    Dual{Tz}(fma(x, y, value(Tz, z)), partials(Tz, z))      # z_body
)

# muladd #
#--------#

@generated function calc_muladd_xyz(x::Dual{T,<:Any,N},
                                    y::Dual{T,<:Any,N},
                                    z::Dual{T,<:Any,N}) where {T,N}
    ex = Expr(:tuple, [:(muladd(value(T, x), partials(T, y)[$i], muladd(value(T, y), partials(T, x)[$i], partials(T, z)[$i]))) for i in 1:N]...)
    return quote
        $(Expr(:meta, :inline))
        v = muladd(value(T, x), value(T, y), value(T, z))
        return Dual{T}(v, $ex)
    end
end

@inline function calc_muladd_xy(x::Dual{T}, y::Dual{T}, z::Real) where T
    vx, vy = value(T, x), value(T, y)
    result = muladd(vx, vy, z)
    return Dual{T}(result, _mul_partials(partials(T, x), partials(T, y), vy, vx))
end

@generated function calc_muladd_xz(x::Dual{T,<:Any,N},
                                   y::Real,
                                   z::Dual{T,<:Any,N}) where {T,N}
    ex = Expr(:tuple, [:(muladd(partials(T, x)[$i], y,  partials(T, z)[$i])) for i in 1:N]...)
    return quote
        $(Expr(:meta, :inline))
        v = muladd(value(T, x), y, value(T, z))
        Dual{T}(v, $ex)
    end
end

@define_ternary_dual_op(
    Base.muladd,
    calc_muladd_xyz(x, y, z),                         # xyz_body
    calc_muladd_xy(x, y, z),                          # xy_body
    calc_muladd_xz(x, y, z),                          # xz_body
    Base.muladd(y, x, z),                             # yz_body
    Dual{Tx}(muladd(value(Tx, x), y, z), partials(Tx, x) * y), # x_body
    Base.muladd(y, x, z),                             # y_body
    Dual{Tz}(muladd(x, y, value(Tz, z)), partials(Tz, z))      # z_body
)

# sin/cos #
#--------#

function Base.sin(d::Dual{T}) where T
    s, c = sincos(value(T, d))
    return Dual{T}(s, c * partials(T, d))
end

function Base.cos(d::Dual{T}) where T
    s, c = sincos(value(T, d))
    return Dual{T}(c, -s * partials(T, d))
end

@inline function Base.sincos(d::Dual{T}) where T
    sd, cd = sincos(value(T, d))
    return (Dual{T}(sd, cd * partials(T, d)), Dual{T}(cd, -sd * partials(T, d)))
end

# sincospi #
#----------#

@inline function Base.sincospi(d::Dual{T}) where T
    sd, cd = sincospi(value(T, d))
    return (Dual{T}(sd, cd * π * partials(T, d)), Dual{T}(cd, -sd * π * partials(T, d)))
end

# LinearAlgebra.givensAlgorithm #
#-------------------------------#

# This definition ensures that we match `LinearAlgebra.givensAlgorithm`
# for non-dual numbers (i.e., `ForwardDiff.Dual` with zero partials)
# `LinearAlgebra.givensAlgorithm` is derived from LAPACK's dlartg
# which is [documented](https://netlib.org/lapack/explore-html/da/dd3/group__lartg_ga86f8f877eaea0386cdc2c3c175d9ea88.html) to return
# three values c, s, u for two arguments x and y with
# u = sgn(x) sqrt(x^2 + y^2)
# c = x/u
# s = y/u
# The function is discontinuous in u at x=0
@define_binary_dual_op(
    LinearAlgebra.givensAlgorithm,
    begin
        vx, vy = value(Txy, x), value(Txy, y)
        c, s, u = LinearAlgebra.givensAlgorithm(vx, vy)
        ∂c∂x = s^2 / u
        ∂c∂y = ∂s∂x = -(c * s / u)
        ∂s∂y = c^2 / u
        ∂x = partials(Txy, x)
        ∂y = partials(Txy, y)
        ∂c = _mul_partials(∂x, ∂y, ∂c∂x, ∂c∂y)
        ∂s = _mul_partials(∂x, ∂y, ∂s∂x, ∂s∂y)
        ∂u = _mul_partials(∂x, ∂y, c, s)
        return Dual{Txy}(c, ∂c), Dual{Txy}(s, ∂s), Dual{Txy}(u, ∂u)
    end,
    begin
        vx = value(Tx, x)
        c, s, u = LinearAlgebra.givensAlgorithm(vx, y)
        ∂c∂x = s^2 / u
        ∂s∂x = -(c * s / u)
        ∂x = partials(Tx, x)
        ∂c = ∂c∂x * ∂x
        ∂s = ∂s∂x * ∂x
        ∂u = c * ∂x
        return Dual{Tx}(c, ∂c), Dual{Tx}(s, ∂s), Dual{Tx}(u, ∂u)
    end,
    begin
        vy = value(Ty, y)
        c, s, u = LinearAlgebra.givensAlgorithm(x, vy)
        ∂c∂y = -(c * s / u)
        ∂s∂y = c^2 / u
        ∂y = partials(Ty, y)
        ∂c = ∂c∂y * ∂y
        ∂s = ∂s∂y * ∂y
        ∂u = s * ∂y
        return Dual{Ty}(c, ∂c), Dual{Ty}(s, ∂s), Dual{Ty}(u, ∂u)
    end,
)

# eigen values and vectors of Hermitian matrices #
#------------------------------------------------#

# Extract structured matrices of primal values and partials
_structured_value(A::Symmetric{Dual{T,V,N}}) where {T,V,N} = Symmetric(map(Base.Fix1(value, T), parent(A)), A.uplo === 'U' ? :U : :L)
_structured_value(A::Hermitian{Dual{T,V,N}}) where {T,V,N} = Hermitian(map(Base.Fix1(value, T), parent(A)), A.uplo === 'U' ? :U : :L)
_structured_value(A::Hermitian{Complex{Dual{T,V,N}}}) where {T,V,N} = Hermitian(map(z -> splat(complex)(map(Base.Fix1(value, T), reim(z))), parent(A)), A.uplo === 'U' ? :U : :L)
_structured_value(A::SymTridiagonal{Dual{T,V,N}}) where {T,V,N} = SymTridiagonal(map(Base.Fix1(value, T), A.dv), map(Base.Fix1(value, T), A.ev))

_structured_partials(A::Symmetric{Dual{T,V,N}}, j::Int) where {T,V,N} = Symmetric(partials.(T, parent(A), j), A.uplo === 'U' ? :U : :L)
_structured_partials(A::Hermitian{Dual{T,V,N}}, j::Int) where {T,V,N} = Hermitian(partials.(T, parent(A), j), A.uplo === 'U' ? :U : :L)
function _structured_partials(A::Hermitian{Complex{Dual{T,V,N}}}, j::Int) where {T,V,N}
    return Hermitian(complex.(partials.(T, real.(parent(A)), j), partials.(T, imag.(parent(A)), j)), A.uplo === 'U' ? :U : :L)
end
_structured_partials(A::SymTridiagonal{Dual{T,V,N}}, j::Int) where {T,V,N} = SymTridiagonal(partials.(T, A.dv, j), partials.(T, A.ev, j))

# Convert arrays of primal values and partials to arrays of Duals
function _to_duals(::Val{T}, values::AbstractArray{<:Real}, partials::Tuple{Vararg{AbstractArray{<:Real}}}) where {T}
    return Dual{T}.(values, tuple.(partials...))
end
function _to_duals(::Val{T}, values::AbstractArray{<:Complex}, partials::Tuple{Vararg{AbstractArray{<:Complex}}}) where {T}
    return complex.(
        Dual{T}.(real.(values), Base.Fix1(map, real).(tuple.(partials...))),
        Dual{T}.(imag.(values), Base.Fix1(map, imag).(tuple.(partials...))),
    )
end

# We forward the call to an internal method that can be shared and reused
LinearAlgebra.eigvals(A::Symmetric{Dual{T,V,N}}) where {T,V<:Real,N} = _eigvals_hermitian(A)
LinearAlgebra.eigvals(A::Hermitian{Dual{T,V,N}}) where {T,V<:Real,N} = _eigvals_hermitian(A)
LinearAlgebra.eigvals(A::Hermitian{Complex{Dual{T,V,N}}}) where {T,V<:Real,N} = _eigvals_hermitian(A)
LinearAlgebra.eigvals(A::SymTridiagonal{Dual{T,V,N}}) where {T,V<:Real,N} = _eigvals_hermitian(A)

# Eigenvalues of Hermitian-structured matrices
const DualMatrixRealComplex{T,V<:Real,N} = Union{AbstractMatrix{Dual{T,V,N}}, AbstractMatrix{Complex{Dual{T,V,N}}}}
function _eigvals_hermitian(A::DualMatrixRealComplex{T,<:Real,N}) where {T,N}
    F = eigen(_structured_value(A))
    λ = F.values
    Q = F.vectors
    parts = ntuple(j -> real(diag(Q' * (_structured_partials(A, j) * Q))), N)
    return _to_duals(Val(T), λ, parts)
end

# A ./ (λ' .- λ) but with diagonal elements zeroed out
# Default out-of-place method
function _lyap_div_zero_diag!!(A::AbstractMatrix, λ::AbstractVector)
    return map(
        (a, b, idx) -> idx[1] == idx[2] ? zero(a) / oneunit(b) : a / b,
        A,
        λ' .- λ,
        CartesianIndices(A),
    )
end
# For `Matrix` (and e.g. `StaticArrays.MMatrix`) we can use an in-place method
_lyap_div_zero_diag!!(A::Matrix, λ::AbstractVector) = _lyap_div_zero_diag!(A, λ)
function _lyap_div_zero_diag!(A::AbstractMatrix, λ::AbstractVector)
    for (j,μ) in enumerate(λ), (k,λ) in enumerate(λ)
        if k == j
            A[k, j] = zero(A[k, j])
        else
            A[k,j] /= μ - λ
        end
    end
    A
end

# We forward the call to an internal method that can be shared and reused
LinearAlgebra.eigen(A::Symmetric{Dual{T,V,N}}) where {T,V<:Real,N} = _eigen_hermitian(A)
LinearAlgebra.eigen(A::Hermitian{Dual{T,V,N}}) where {T,V<:Real,N} = _eigen_hermitian(A)
LinearAlgebra.eigen(A::Hermitian{Complex{Dual{T,V,N}}}) where {T,V<:Real,N} = _eigen_hermitian(A)
LinearAlgebra.eigen(A::SymTridiagonal{Dual{T,V,N}}) where {T,V<:Real,N} = _eigen_hermitian(A)

function _eigen_hermitian(A::DualMatrixRealComplex{T,<:Real,N}) where {T,N}
    F = eigen(_structured_value(A))
    λ = F.values
    Q = F.vectors
    # `Q' * (∂A * Q)`, not `(Q' * ∂A) * Q`: the latter hits `Adjoint * Symmetric`, which has no BLAS
    # specialization and so allocates an extra temporary and skips `symm`/`hemm`
    Qt_∂A_Q = ntuple(j -> Q' * (_structured_partials(A, j) * Q), N)
    λ_partials = map(real ∘ diag, Qt_∂A_Q)
    Q_partials = map(Qt_∂Aj_Q -> Q*_lyap_div_zero_diag!!(Qt_∂Aj_Q, λ), Qt_∂A_Q)
    return Eigen(_to_duals(Val(T), λ, λ_partials), _to_duals(Val(T), Q, Q_partials))
end

# Functions in SpecialFunctions which return tuples #
# Their derivatives are not defined in DiffRules    #
#---------------------------------------------------#

function SpecialFunctions.logabsgamma(d::Dual{T,<:Real}) where {T}
    x = value(T, d)
    y, s = SpecialFunctions.logabsgamma(x)
    return (Dual{T}(y, SpecialFunctions.digamma(x) * partials(T, d)), s)
end

# Derivatives wrt to first parameter and precision setting are not supported
function SpecialFunctions.gamma_inc(a::Real, d::Dual{T,<:Real}, ind::Integer) where {T}
    x = value(T, d)
    p, q = SpecialFunctions.gamma_inc(a, x, ind)
    ∂p = exp(-x) * x^(a - 1) / SpecialFunctions.gamma(a) * partials(T, d)
    return (Dual{T}(p, ∂p), Dual{T}(q, -∂p))
end

###################
# Pretty Printing #
###################

function Base.show(io::IO, d::Dual{T,V,N}) where {T,V,N}
    print(io, "Dual{$(repr(T))}(", value(T, d))
    for i in 1:N
        print(io, ",", partials(T, d, i))
    end
    print(io, ")")
end

for op in (:(Base.typemin), :(Base.typemax), :(Base.floatmin), :(Base.floatmax))
    @eval function $op(::Type{ForwardDiff.Dual{T,V,N}}) where {T,V,N}
        ForwardDiff.Dual{T,V,N}($op(V))
    end
end

Printf.tofloat(d::Dual{T}) where {T} = Printf.tofloat(value(T, d))

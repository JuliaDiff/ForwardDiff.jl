struct Partials{N,V} <: AbstractVector{V}
    values::NTuple{N,V}
end

##############################
# Utility/Accessor Functions #
##############################

@generated function single_seed(::Type{Partials{N,V}}, ::Val{i}) where {N,V,i}
    ex = Expr(:tuple, [ifelse(i === j, :(one(V)), :(zero(V))) for j in 1:N]...)
    return :(Partials($(ex)))
end

@inline valtype(::Partials{N,V}) where {N,V} = V
@inline valtype(::Type{Partials{N,V}}) where {N,V} = V

@inline npartials(::Partials{N}) where {N} = N
@inline npartials(::Type{Partials{N,V}}) where {N,V} = N

@inline Base.length(::Partials{N}) where {N} = N
@inline Base.size(::Partials{N}) where {N} = (N,)

@inline Base.getindex(partials::Partials, i::Int) = partials.values[i]

Base.start(partials::Partials) = start(partials.values)
Base.next(partials::Partials, i) = next(partials.values, i)
Base.done(partials::Partials, i) = done(partials.values, i)

Base.IndexStyle(::Type{<:Partials}) = IndexLinear()

#####################
# Generic Functions #
#####################

@inline iszero(partials::Partials) = iszero_tuple(partials.values)

@inline Base.zero(partials::Partials) = zero(typeof(partials))
@inline Base.zero(::Type{Partials{N,V}}) where {N,V} = Partials{N,V}(zero_tuple(NTuple{N,V}))

@inline Base.one(partials::Partials) = one(typeof(partials))
@inline Base.one(::Type{Partials{N,V}}) where {N,V} = Partials{N,V}(one_tuple(NTuple{N,V}))

@inline Base.rand(partials::Partials) = rand(typeof(partials))
@inline Base.rand(::Type{Partials{N,V}}) where {N,V} = Partials{N,V}(rand_tuple(NTuple{N,V}))
@inline Base.rand(rng::AbstractRNG, partials::Partials) = rand(rng, typeof(partials))
@inline Base.rand(rng::AbstractRNG, ::Type{Partials{N,V}}) where {N,V} = Partials{N,V}(rand_tuple(rng, NTuple{N,V}))

Base.isequal(a::Partials{N}, b::Partials{N}) where {N} = isequal(a.values, b.values)
Base.:(==)(a::Partials{N}, b::Partials{N}) where {N} = a.values == b.values

const PARTIALS_HASH = hash(Partials)

Base.hash(partials::Partials) = hash(partials.values, PARTIALS_HASH)
Base.hash(partials::Partials, hsh::UInt64) = hash(hash(partials), hsh)

@inline Base.copy(partials::Partials) = partials

Base.read(io::IO, ::Type{Partials{N,V}}) where {N,V} = Partials{N,V}(ntuple(i->read(io, V), N))

function Base.write(io::IO, partials::Partials)
    for p in partials
        write(io, p)
    end
end

########################
# Conversion/Promotion #
########################

Base.promote_rule(::Type{Partials{N,A}}, ::Type{Partials{N,B}}) where {N,A,B} = Partials{N,promote_type(A, B)}

Base.convert(::Type{Partials{N,V}}, partials::Partials) where {N,V} = Partials{N,V}(partials.values)
Base.convert(::Type{Partials{N,V}}, partials::Partials{N,V}) where {N,V} = partials

########################
# Arithmetic Functions #
########################

@inline Base.:+(a::Partials{N}, b::Partials{N}) where {N} = Partials(add_tuples(a.values, b.values))
@inline Base.:-(a::Partials{N}, b::Partials{N}) where {N} = Partials(sub_tuples(a.values, b.values))
@inline Base.:-(partials::Partials) = Partials(minus_tuple(partials.values))
@inline Base.:*(x::Real, partials::Partials) = partials*x

@inline function _div_partials(a::Partials, b::Partials, aval, bval)
    return _mul_partials(a, b, inv(bval), -(aval / (bval*bval)))
end

# NaN/Inf-safe methods #
#----------------------#

if NANSAFE_MODE_ENABLED
    @inline function Base.:*(partials::Partials, x::Real)
        x = ifelse(!isfinite(x) && iszero(partials), one(x), x)
        return Partials(scale_tuple(partials.values, x))
    end

    @inline function Base.:/(partials::Partials, x::Real)
        x = ifelse(x == zero(x) && iszero(partials), one(x), x)
        return Partials(div_tuple_by_scalar(partials.values, x))
    end

    @inline function _mul_partials(a::Partials{N}, b::Partials{N}, x_a, x_b) where N
        x_a = ifelse(!isfinite(x_a) && iszero(a), one(x_a), x_a)
        x_b = ifelse(!isfinite(x_b) && iszero(b), one(x_b), x_b)
        return Partials(mul_tuples(a.values, b.values, x_a, x_b))
    end
else
    @inline function Base.:*(partials::Partials, x::Real)
        return Partials(scale_tuple(partials.values, x))
    end

    @inline function Base.:/(partials::Partials, x::Real)
        return Partials(div_tuple_by_scalar(partials.values, x))
    end

    @inline function _mul_partials(a::Partials{N}, b::Partials{N}, x_a, x_b) where N
        return Partials(mul_tuples(a.values, b.values, x_a, x_b))
    end
end

# edge cases where N == 0 #
#-------------------------#

@inline Base.:+(a::Partials{0,A}, b::Partials{0,B}) where {A,B} = Partials{0,promote_type(A,B)}(tuple())
@inline Base.:+(a::Partials{0,A}, b::Partials{N,B}) where {N,A,B} = convert(Partials{N,promote_type(A,B)}, b)
@inline Base.:+(a::Partials{N,A}, b::Partials{0,B}) where {N,A,B} = convert(Partials{N,promote_type(A,B)}, a)

@inline Base.:-(a::Partials{0,A}, b::Partials{0,B}) where {A,B} = Partials{0,promote_type(A,B)}(tuple())
@inline Base.:-(a::Partials{0,A}, b::Partials{N,B}) where {N,A,B} = -(convert(Partials{N,promote_type(A,B)}, b))
@inline Base.:-(a::Partials{N,A}, b::Partials{0,B}) where {N,A,B} = convert(Partials{N,promote_type(A,B)}, a)
@inline Base.:-(partials::Partials{0,V}) where {V} = partials

@inline Base.:*(partials::Partials{0,V}, x::Real) where {V} = Partials{0,promote_type(V,typeof(x))}(tuple())
@inline Base.:*(x::Real, partials::Partials{0,V}) where {V} = Partials{0,promote_type(V,typeof(x))}(tuple())

@inline Base.:/(partials::Partials{0,V}, x::Real) where {V} = Partials{0,promote_type(V,typeof(x))}(tuple())

@inline _mul_partials(a::Partials{0,A}, b::Partials{0,B}, afactor, bfactor) where {A,B} = Partials{0,promote_type(A,B)}(tuple())
@inline _mul_partials(a::Partials{0,A}, b::Partials{N,B}, afactor, bfactor) where {N,A,B} = bfactor * b
@inline _mul_partials(a::Partials{N,A}, b::Partials{0,B}, afactor, bfactor) where {N,A,B} = afactor * a

##################################
# Generated Functions on NTuples #
##################################
# The below functions are generally
# equivalent to directly mapping over
# tuples using `map`, but run a bit
# faster since they generate inline code
# that doesn't rely on closures.

function tupexpr(f, N)
    ex = Expr(:tuple, [f(i) for i=1:N]...)
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $ex
    end
end

@inline iszero_tuple(::Tuple{}) = true
@inline zero_tuple(::Type{Tuple{}}) = tuple()
@inline one_tuple(::Type{Tuple{}}) = tuple()
@inline rand_tuple(::AbstractRNG, ::Type{Tuple{}}) = tuple()
@inline rand_tuple(::Type{Tuple{}}) = tuple()

@generated function iszero_tuple(tup::NTuple{N,V}) where {N,V}
    ex = Expr(:&&, [:(z == tup[$i]) for i=1:N]...)
    return quote
        z = zero(V)
        $(Expr(:meta, :inline))
        @inbounds return $ex
    end
end

@generated function zero_tuple(::Type{NTuple{N,V}}) where {N,V}
    ex = tupexpr(i -> :(z), N)
    return quote
        z = zero(V)
        return $ex
    end
end

@generated function one_tuple(::Type{NTuple{N,V}}) where {N,V}
    ex = tupexpr(i -> :(z), N)
    return quote
        z = one(V)
        return $ex
    end
end

@generated function rand_tuple(rng::AbstractRNG, ::Type{NTuple{N,V}}) where {N,V}
    return tupexpr(i -> :(rand(rng, V)), N)
end

@generated function rand_tuple(::Type{NTuple{N,V}}) where {N,V}
    return tupexpr(i -> :(rand(V)), N)
end

@generated function scale_tuple(tup::NTuple{N}, x) where N
    return tupexpr(i -> :(tup[$i] * x), N)
end

@generated function div_tuple_by_scalar(tup::NTuple{N}, x) where N
    return tupexpr(i -> :(tup[$i] / x), N)
end

@generated function add_tuples(a::NTuple{N}, b::NTuple{N})  where N
    return tupexpr(i -> :(a[$i] + b[$i]), N)
end

@generated function sub_tuples(a::NTuple{N}, b::NTuple{N})  where N
    return tupexpr(i -> :(a[$i] - b[$i]), N)
end

@generated function minus_tuple(tup::NTuple{N}) where N
    return tupexpr(i -> :(-tup[$i]), N)
end

@generated function mul_tuples(a::NTuple{N}, b::NTuple{N}, afactor, bfactor) where N
    return tupexpr(i -> :((afactor * a[$i]) + (bfactor * b[$i])), N)
end

###################
# Pretty Printing #
###################

Base.show(io::IO, p::Partials{N}) where {N} = print(io, "Partials", p.values)

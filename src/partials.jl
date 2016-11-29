immutable Partials{N,T} <: AbstractArray{N,T}
    values::NTuple{N,T}
end

##############################
# Utility/Accessor Functions #
##############################

@inline valtype{N,T}(::Partials{N,T}) = T
@inline valtype{N,T}(::Type{Partials{N,T}}) = T

@inline npartials{N}(::Partials{N}) = N
@inline npartials{N,T}(::Type{Partials{N,T}}) = N

@inline Base.length{N}(::Partials{N}) = N
@inline Base.size{N}(::Partials{N}) = (N,)

@inline Base.getindex(partials::Partials, i) = partials.values[i]
setindex{N,T}(partials::Partials{N,T}, v, i) = Partials{N,T}((partials[1:i-1]..., v, partials[i+1:N]...))

Base.start(partials::Partials) = start(partials.values)
Base.next(partials::Partials, i) = next(partials.values, i)
Base.done(partials::Partials, i) = done(partials.values, i)

#####################
# Generic Functions #
#####################

@inline iszero(partials::Partials) = iszero_tuple(partials.values)

@inline Base.zero(partials::Partials) = zero(typeof(partials))
@inline Base.zero{N,T}(::Type{Partials{N,T}}) = Partials{N,T}(zero_tuple(NTuple{N,T}))

@inline Base.one(partials::Partials) = one(typeof(partials))
@inline Base.one{N,T}(::Type{Partials{N,T}}) = Partials{N,T}(one_tuple(NTuple{N,T}))

@inline Base.rand(partials::Partials) = rand(typeof(partials))
@inline Base.rand{N,T}(::Type{Partials{N,T}}) = Partials{N,T}(rand_tuple(NTuple{N,T}))
@inline Base.rand(rng::AbstractRNG, partials::Partials) = rand(rng, typeof(partials))
@inline Base.rand{N,T}(rng::AbstractRNG, ::Type{Partials{N,T}}) = Partials{N,T}(rand_tuple(rng, NTuple{N,T}))

Base.isequal{N}(a::Partials{N}, b::Partials{N}) = isequal(a.values, b.values)
@compat(Base.:(==)){N}(a::Partials{N}, b::Partials{N}) = a.values == b.values

const PARTIALS_HASH = hash(Partials)

Base.hash(partials::Partials) = hash(partials.values, PARTIALS_HASH)
Base.hash(partials::Partials, hsh::UInt64) = hash(hash(partials), hsh)

@inline Base.copy(partials::Partials) = partials

Base.read{N,T}(io::IO, ::Type{Partials{N,T}}) = Partials{N,T}(ntuple(i->read(io, T), Val{N}))

function Base.write(io::IO, partials::Partials)
    for p in partials
        write(io, p)
    end
end

########################
# Conversion/Promotion #
########################

Base.promote_rule{N,A,B}(::Type{Partials{N,A}}, ::Type{Partials{N,B}}) = Partials{N,promote_type(A, B)}

Base.convert{N,T}(::Type{Partials{N,T}}, partials::Partials) = Partials{N,T}(partials.values)
Base.convert{N,T}(::Type{Partials{N,T}}, partials::Partials{N,T}) = partials

########################
# Arithmetic Functions #
########################

@inline @compat(Base.:+){N}(a::Partials{N}, b::Partials{N}) = Partials(add_tuples(a.values, b.values))
@inline @compat(Base.:-){N}(a::Partials{N}, b::Partials{N}) = Partials(sub_tuples(a.values, b.values))
@inline @compat(Base.:-)(partials::Partials) = Partials(minus_tuple(partials.values))
@inline @compat(Base.:*)(partials::Partials, x::Real) = Partials(scale_tuple(partials.values, x))
@inline @compat(Base.:*)(x::Real, partials::Partials) = partials*x
@inline @compat(Base.:/)(partials::Partials, x::Real) = Partials(div_tuple_by_scalar(partials.values, x))

@inline function _mul_partials{N}(a::Partials{N}, b::Partials{N}, afactor, bfactor)
    return Partials(mul_tuples(a.values, b.values, afactor, bfactor))
end

@inline function _div_partials(a::Partials, b::Partials, aval, bval)
    afactor = inv(bval)
    bfactor = -aval/(bval*bval)
    return _mul_partials(a, b, afactor, bfactor)
end

# edge cases where N == 0 #
#-------------------------#

@inline @compat(Base.:+){A,B}(a::Partials{0,A}, b::Partials{0,B}) = Partials{0,promote_type(A,B)}(tuple())
@inline @compat(Base.:-){A,B}(a::Partials{0,A}, b::Partials{0,B}) = Partials{0,promote_type(A,B)}(tuple())
@inline @compat(Base.:-){T}(partials::Partials{0,T}) = partials
@inline @compat(Base.:*){T}(partials::Partials{0,T}, x::Real) = Partials{0,promote_type(T,typeof(x))}(tuple())
@inline @compat(Base.:*){T}(x::Real, partials::Partials{0,T}) = Partials{0,promote_type(T,typeof(x))}(tuple())
@inline @compat(Base.:/){T}(partials::Partials{0,T}, x::Real) = Partials{0,promote_type(T,typeof(x))}(tuple())

@inline _mul_partials{A,B}(a::Partials{0,A}, b::Partials{0,B}, afactor, bfactor) = Partials{0,promote_type(A,B)}(tuple())
@inline _div_partials{A,B}(a::Partials{0,A}, b::Partials{0,B}, afactor, bfactor) = Partials{0,promote_type(A,B)}(tuple())

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

@generated function iszero_tuple{N,T}(tup::NTuple{N,T})
    ex = Expr(:&&, [:(z == tup[$i]) for i=1:N]...)
    return quote
        z = zero(T)
        $(Expr(:meta, :inline))
        @inbounds return $ex
    end
end

@generated function zero_tuple{N,T}(::Type{NTuple{N,T}})
    ex = tupexpr(i -> :(z), N)
    return quote
        z = zero(T)
        return $ex
    end
end

@generated function one_tuple{N,T}(::Type{NTuple{N,T}})
    ex = tupexpr(i -> :(z), N)
    return quote
        z = one(T)
        return $ex
    end
end

@generated function rand_tuple{N,T}(rng::AbstractRNG, ::Type{NTuple{N,T}})
    return tupexpr(i -> :(rand(rng, T)), N)
end

@generated function rand_tuple{N,T}(::Type{NTuple{N,T}})
    return tupexpr(i -> :(rand(T)), N)
end

@generated function scale_tuple{N}(tup::NTuple{N}, x)
    return tupexpr(i -> :(tup[$i] * x), N)
end

@generated function div_tuple_by_scalar{N}(tup::NTuple{N}, x)
    return tupexpr(i -> :(tup[$i] / x), N)
end

@generated function add_tuples{N}(a::NTuple{N}, b::NTuple{N})
    return tupexpr(i -> :(a[$i] + b[$i]), N)
end

@generated function sub_tuples{N}(a::NTuple{N}, b::NTuple{N})
    return tupexpr(i -> :(a[$i] - b[$i]), N)
end

@generated function minus_tuple{N}(tup::NTuple{N})
    return tupexpr(i -> :(-tup[$i]), N)
end

@generated function mul_tuples{N}(a::NTuple{N}, b::NTuple{N}, afactor, bfactor)
    return tupexpr(i -> :((afactor * a[$i]) + (bfactor * b[$i])), N)
end

###################
# Pretty Printing #
###################

Base.show{N}(io::IO, p::Partials{N}) = print(io, "Partials", p.values)

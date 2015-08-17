immutable Partials{T,C}
    data::C
    Partials{N}(data::NTuple{N,T}) = new(data)
    Partials(data::Vector{T}) = new(data)
end

typealias PartialsTup{N,T} Partials{T,NTuple{N,T}}
typealias PartialsVec{T} Partials{T,Vector{T}}

Partials(data) = Partials{eltype(data),typeof(data)}(data)

##############################
# Utility/Accessor Functions #
##############################
data(partials::Partials) = partials.data

eltype{T,C}(::Type{Partials{T,C}}) = T
eltype{T}(::Partials{T}) = T

getindex(partials::Partials, i) = data(partials)[i]

length(partials::Partials) = length(data(partials))

start(partials) = start(data(partials))
next(partials, i) = next(data(partials), i)
done(partials, i) = done(data(partials), i)

################
# Constructors #
################
zero_partials{N,T}(::Type{NTuple{N,T}}, n::Int) = Partials(zero_tuple(NTuple{N,T}))
zero_partials{T}(::Type{Vector{T}}, n) = Partials(zeros(T, n))

rand_partials{N,T}(::Type{NTuple{N,T}}, n::Int) = Partials(rand_tuple(NTuple{N,T}))
rand_partials{T}(::Type{Vector{T}}, n::Int) = Partials(rand(T, n))

#####################
# Generic Functions #
#####################
function iszero{T}(partials::Partials{T})
    z = zero(T)
    return any(x -> x == z, data(partials))
end

==(a::Partials, b::Partials) = data(a) == data(b)
isequal(a::Partials, b::Partials) = isequal(data(a), data(b))

hash(partials::Partials) = hash(data(partials))
hash(partials::Partials, hsh::Uint64) = hash(hash(partials), hsh)

copy(partials::Partials) = partials

function read{N,T}(io::IO, ::Type{PartialsTup{N,T}}, n::Int)
    return Partials(ntuple(i->read(io, T), Val{N}))
end

function read{T}(io::IO, ::Type{PartialsVec{T}}, n::Int)
    return Partials([read(io, T) for i in 1:n])
end

function write(io::IO, partials::Partials)
    for partial in data(partials)
        write(io, partial)
    end
end

convert{T,C}(::Type{Partials{T,C}}, partials::Partials) = Partials(convert(C, data(partials)))
convert{T,C}(::Type{Partials{T,C}}, partials::Partials{T,C}) = partials

##################
# Math Functions #
##################

# Addition/Subtraction #
#----------------------#
function +{N,A,B}(a::PartialsTup{N,A}, b::PartialsTup{N,B})
    return Partials(add_tuples(data(a), data(b)))
end

function +{A,B}(a::PartialsVec{A}, b::PartialsVec{B})
    return Partials(data(a) + data(b))
end

function -{N,A,B}(a::PartialsTup{N,A}, b::PartialsTup{N,B})
    return Partials(subtract_tuples(data(a), data(b)))
end

function -{A,B}(a::PartialsVec{A}, b::PartialsVec{B})
    return Partials(data(a) - data(b))
end

-{N,T}(partials::PartialsTup{N,T}) = Partials(minus_tuple(data(partials)))
-{T}(partials::PartialsVec{T}) = Partials(-data(partials))

# Multiplication #
#----------------#
function *{N,T}(partials::PartialsTup{N,T}, x::Number)
    return Partials(scale_tuple(data(partials), x))
end

function *{T}(partials::PartialsVec{T}, x::Number)
    return Partials(data(partials)*x)
end

*(x::Number, partials::Partials) = partials*x

@generated function _mul_partials{A,B,C,D}(a::PartialsVec{A}, b::PartialsVec{B}, afactor::C, bfactor::D)
    T = promote_type(A, B, C, D)
    return quote
        result = Vector{$T}(length(a))
        @simd for i in eachindex(result)
            @inbounds result[i] = (afactor * a[i]) + (bfactor * b[i])
        end
        return Partials(result)
    end
end

function _mul_partials{N,A,B}(a::PartialsTup{N,A}, b::PartialsTup{N,B}, afactor, bfactor)
    return Partials(mul_tuples(data(a), data(b), afactor, bfactor))
end

# Division #
#----------#
function /{N,T}(partials::PartialsTup{N,T}, x::Number)
    return Partials(div_tuple_by_scalar(data(partials), x))
end

function /{T}(partials::PartialsVec{T}, x::Number)
    return Partials(data(partials) / x)
end

function _div_partials(x, partials::Partials, val)
    return ((-x) / (val^2)) * partials
end

function _div_partials(a::Partials, b::Partials, aval, bval)
    afactor = inv(bval)
    bfactor = -aval/(bval^2)
    return _mul_partials(a, b, afactor, bfactor)
end

##################################
# Generated Functions on NTuples #
##################################
# The below functions are generally
# equivalent to directly mapping over
# tuples using `map`, but run a bit
# faster since they generate inline code
# that doesn't rely on closures.

function tupexpr(f,N)
    ex = Expr(:tuple, [f(i) for i=1:N]...)
    return quote
        @inbounds return $ex
    end
end

@generated function zero_tuple{N,T}(::Type{NTuple{N,T}})
    result = tupexpr(i -> :z, N)
    return quote
        z = zero($T)
        return $result
    end
end

@generated function rand_tuple{N,T}(::Type{NTuple{N,T}}) 
    return tupexpr(i -> :(rand($T)), N)
end

@generated function scale_tuple{N}(tup::NTuple{N}, x)
    return tupexpr(i -> :(tup[$i] * x), N)
end

@generated function div_tuple_by_scalar{N}(tup::NTuple{N}, x)
    return tupexpr(i -> :(tup[$i]/x), N)
end

@generated function minus_tuple{N}(tup::NTuple{N})
    return tupexpr(i -> :(-tup[$i]), N)
end

@generated function subtract_tuples{N}(a::NTuple{N}, b::NTuple{N})
    return tupexpr(i -> :(a[$i]-b[$i]), N)
end

@generated function add_tuples{N}(a::NTuple{N}, b::NTuple{N})
    return tupexpr(i -> :(a[$i]+b[$i]), N)
end

@generated function mul_tuples{N}(a::NTuple{N}, b::NTuple{N}, afactor, bfactor)
    return tupexpr(i -> :((afactor * a[$i]) + (bfactor * b[$i])), N)
end
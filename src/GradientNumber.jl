include("gradnum_math/tuple_funcs.jl")

immutable GradientNumber{N,T,C} <: ForwardDiffNumber{N,T,C}
    value::T
    grad::C
    GradientNumber(value::T, grad::NTuple{N,T}) = new(value, grad)
    GradientNumber(value::T, grad::Vector{T}) = new(value, grad)
    GradientNumber(value, grad::Tuple) = GradientNumber{N,T,C}(convert(T,value), convert(NTuple{N,T}, grad))
    GradientNumber(value, grad::Vector) = GradientNumber{N,T,C}(convert(T,value), convert(Vector{T}, grad))
end

typealias GradNumTup{N,T} GradientNumber{N,T,NTuple{N,T}}
typealias GradNumVec{N,T} GradientNumber{N,T,Vector{T}}

function GradientNumber{N,T}(value, grad::NTuple{N,T})
    S = promote_type(typeof(value), T)
    return GradientNumber{N,S,NTuple{N,S}}(convert(S, value), convert(NTuple{N,S},grad))
end

GradientNumber{N,T}(value::T, grad::NTuple{N,T}) = GradientNumber{N,T,NTuple{N,T}}(value, grad)
GradientNumber{T}(value::T, grad::T...) = GradientNumber(value, grad)

##############################
# Utility/Accessor Functions #
##############################
value(g::GradientNumber) = g.value

grad(g::GradientNumber) = g.grad
hess(g::GradientNumber) = error("GradientNumbers do not store Hessian values")
tens(g::GradientNumber) = error("GradientNumbers do not store tensor values")

eltype{N,T,C}(::Type{GradientNumber{N,T,C}}) = T
eltype{N,T}(::GradientNumber{N,T}) = T

npartials{N,T,C}(::Type{GradientNumber{N,T,C}}) = N
npartials{N}(::GradientNumber{N}) = N

zero_partials{N,T,C<:Tuple}(::Type{GradientNumber{N,T,C}}) = zero_tuple(C)
zero_partials{N,T,C<:Vector}(::Type{GradientNumber{N,T,C}}) = zeros(T, N)
rand_partials{N,T,C<:Tuple}(::Type{GradientNumber{N,T,C}}) = rand_tuple(C)
rand_partials{N,T,C<:Vector}(::Type{GradientNumber{N,T,C}}) = rand(T, N)

#####################
# Generic Functions #
#####################
isconstant(g::GradientNumber) = any(x -> x == 0, grad(g))
isconstant(::GradientNumber{0}) = true

==(a::GradientNumber, b::GradientNumber) = value(a) == value(b) && grad(a) == grad(b)

isequal(a::GradientNumber, b::GradientNumber) = isequal(value(a), value(b)) && isequal(grad(a), grad(b))

hash(g::GradientNumber) = isconstant(g) ? hash(value(g)) : hash(value(g), hash(grad(g)))
hash(g::GradientNumber, hsh::Uint64) = hash(hash(g), hsh)

read_partials{N,T}(io::IO, n::Int, ::Type{NTuple{N,T}}) = ntuple(n->read(io, T), Val{N})
read_partials{T}(io::IO, n::Int, ::Type{Vector{T}}) = [read(io, T) for i in 1:n]

function read{N,T,C}(io::IO, ::Type{GradientNumber{N,T,C}})
    value = read(io, T)
    partials = read_partials(io, N, C)
    return GradientNumber{N,T,C}(value, partials)
end

function write(io::IO, g::GradientNumber)
    write(io, value(g))
    for du in grad(g)
        write(io, du)
    end
end

########################
# Conversion/Promotion #
########################
zero{N,T,C}(G::Type{GradientNumber{N,T,C}}) = G(zero(T), zero_partials(G))
one{N,T,C}(G::Type{GradientNumber{N,T,C}}) = G(one(T), zero_partials(G))
rand{N,T,C}(G::Type{GradientNumber{N,T,C}}) = G(rand(T), rand_partials(G))

for F in (:GradNumVec, :GradNumTup)
    @eval begin
        convert{N,T}(::Type{$(F){N,T}}, x::Real) = $(F){N,T}(x, zero_partials($(F){N,T}))
        convert{N,A,B}(::Type{$(F){N,A}}, g::$(F){N,B}) = $(F){N,A}(value(g), grad(g))
        convert{N,T}(::Type{$(F){N,T}}, g::$(F){N,T}) = g

        promote_rule{N,A,B}(::Type{$(F){N,A}}, ::Type{$(F){N,B}}) = $(F){N,promote_type(A,B)}
        promote_rule{N,A,B}(::Type{$(F){N,A}}, ::Type{B}) = $(F){N,promote_type(A,B)}
    end
end

convert{T<:Real}(::Type{T}, g::GradientNumber{0}) = convert(T, value(g))
convert{T<:Real}(::Type{T}, g::GradientNumber) = isconstant(g) ? convert(T, value(g)) : throw(InexactError())
convert(::Type{GradientNumber}, g::GradientNumber) = g
convert(::Type{GradientNumber}, x::Real) = GradientNumber(x)

#########################
# Math with GradientNumber #
#########################
# helper function to force use of NaNMath 
# functions in derivative calculations
function to_nanmath(x::Expr)
    if x.head == :call
        funsym = Expr(:.,:NaNMath,Base.Meta.quot(x.args[1]))
        return Expr(:call,funsym,[to_nanmath(z) for z in x.args[2:end]]...)
    else
        return Expr(:call,[to_nanmath(z) for z in x.args]...)
    end
end
to_nanmath(x) = x

abs(g::GradientNumber) = (value(g) >= 0) ? g : -g
abs2(g::GradientNumber) = g*g

include("gradnum_math/gradtup_math.jl")
include("gradnum_math/gradvec_math.jl")

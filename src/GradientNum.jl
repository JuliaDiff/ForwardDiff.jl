immutable GradientNum{N,T,C} <: AutoDiffNum{N,T,C}
    value::T
    partials::C
    GradientNum(value::T, partials::NTuple{N,T}) = new(value, partials)
    GradientNum(value::T, partials::Vector{T}) = new(value, partials)
    GradientNum(value::T) = new(value, zero_partials(GradientNum{N,T,C}))
end

typealias GradNumTup{N,T} GradientNum{N,T,NTuple{N,T}}
typealias GradNumVec{N,T} GradientNum{N,T,Vector{T}}

GradientNum{N,T}(value::T, partials::NTuple{N,T}) = GradientNum{N,T,NTuple{N,T}}(value, partials)
GradientNum{T}(value::T) = GradientNum{0,T,NTuple{0,T}}(value, tuple())
GradientNum{T}(value::T, partials::T...) = GradientNum(value, partials)
GradientNum(value::Real, partials::Real...) = GradientNum(promote(value, partials...)...)

##############################
# Utility/Accessor Functions #
##############################
value(g::GradientNum) = g.value
partials(g::GradientNum) = g.partials
partials(g::GradientNum, i) = g.partials[i]

grad(g::GradientNum) = g
hess(g::GradientNum) = error("GradientNums do not store Hessian values")
tens(g::GradientNum) = error("GradientNums do not store tensor values")

eltype{N,T,C}(::Type{GradientNum{N,T,C}}) = T
npartials{N,T,C}(::Type{GradientNum{N,T,C}}) = N

include("grad/tuple_funcs.jl")

zero_partials{N,T,C<:Tuple}(::Type{GradientNum{N,T,C}}) = zero_tuple(C)
zero_partials{N,T,C<:Vector}(::Type{GradientNum{N,T,C}}) = zeros(N, T)

#####################
# Generic Functions #
#####################
isconstant(g::GradientNum) = any(x -> x == 0, partials(g))
isconstant(::GradientNum{0}) = true
isreal(g::GradientNum) = isconstant(g)

==(a::GradientNum, b::GradientNum) = value(a) == value(b) && partials(a) == partials(b)

isequal(a::GradientNum, b::GradientNum) = isequal(value(a), value(b)) && isequal(partials(a), partials(b))

hash(g::GradientNum) = isconstant(g) ? hash(value(g)) : hash(value(g), hash(partials(g)))

function read{G<:GradientNum}(io::IO, ::Type{G})
    T = eltype(G)
    value = read(io, T)
    partials = ntuple(n->read(io, T), npartials(G))
    return G(value, partials)
end

function write(io::IO, g::GradientNum)
    write(io, value(g))
    for du in partials(g)
        write(io, du)
    end
end

########################
# Conversion/Promotion #
########################
zero{N,T,C}(g::GradientNum{N,T,C}) = GradientNum{N,T,C}(zero(value(g)), zero_partials(typeof(g)))
zero{N,T,C}(::Type{GradientNum{N,T,C}}) = GradientNum{N,T,C}(zero(T), zero_partials(GradientNum{N,T,C}))

one{N,T,C}(g::GradientNum{N,T,C}) = GradientNum{N,T,C}(one(value(g)), zero_partials(typeof(g)))
one{N,T,C}(::Type{GradientNum{N,T,C}}) = GradientNum{N,T,C}(one(T), zero_partials(GradientNum{N,T,C}))

for F in (:GradNumVec, :GradNumTup)
    @eval begin
        convert{N,T}(::Type{$(F){N,T}}, x::Real) = $(F){N,T}(x)
        convert{N,T}(::Type{$(F){N,T}}, g::$(F){N}) = $(F){N,T}(value(g), partials(g))
        convert{N,T}(::Type{$(F){N,T}}, g::$(F){N,T}) = g

        promote_rule{N,A,B}(::Type{$(F){N,A}}, ::Type{$(F){N,B}}) = $(F){N,promote_type(A,B)}
        promote_rule{N,A,B}(::Type{$(F){N,A}}, ::Type{B}) = $(F){N,promote_type(A,B)}
    end
end

convert{T<:Real}(::Type{T}, g::GradientNum{0}) = convert(T, value(g))
convert{T<:Real}(::Type{T}, g::GradientNum) = isconstant(g) ? convert(T, value(g)) : throw(InexactError())
convert(::Type{GradientNum}, g::GradientNum) = g
convert(::Type{GradientNum}, x::Real) = GradientNum(x)

#########################
# Math with GradientNum #
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

abs(g::GradientNum) = (value(g) >= 0) ? g : -g
abs2(g::GradientNum) = g*g

include("grad/gradtup_math.jl")
include("grad/gradvec_math.jl")

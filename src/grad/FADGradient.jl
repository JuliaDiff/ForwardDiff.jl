immutable FADGradient{N,T,C} <: FADNumber{N,T,C}
    value::T
    partials::C
    FADGradient(value::T, partials::NTuple{N,T}) = new(value, partials)
    FADGradient(value::T, partials::Vector{T}) = new(value, partials)
    FADGradient(value::T) = new(value, zero_partials(FADGradient{N,T,C}))
end

typealias GradTup{N,T} FADGradient{N,T,NTuple{N,T}}
typealias GradVec{N,T} FADGradient{N,T,Vector{T}}

FADGradient{N,T}(value::T, partials::NTuple{N,T}) = FADGradient{N,T,NTuple{N,T}}(value, partials)
FADGradient{T}(value::T) = FADGradient{0,T,NTuple{0,T}}(value, tuple())
FADGradient{T}(value::T, partials::T...) = FADGradient(value, partials)
FADGradient(value::Real, partials::Real...) = FADGradient(promote(value, partials...)...)

##############################
# Utility/Accessor Functions #
##############################
value(g::FADGradient) = g.value
partials(g::FADGradient) = g.partials
partials(g::FADGradient, i) = g.partials[i]

grad(g::FADGradient) = g
hess(g::FADGradient) = error("FADGradients do not store Hessian values")
tens(g::FADGradient) = error("FADGradients do not store tensor values")

eltype{N,T,C}(::Type{FADGradient{N,T,C}}) = T
npartials{N,T,C}(::Type{FADGradient{N,T,C}}) = N

include("tuple_funcs.jl")

zero_partials{N,T,C<:Tuple}(::Type{FADGradient{N,T,C}}) = zero_tuple(C)
zero_partials{N,T,C<:Vector}(::Type{FADGradient{N,T,C}}) = zeros(N, T)

#####################
# Generic Functions #
#####################
isconstant(g::FADGradient) = any(x -> x == 0, partials(g))
isconstant(::FADGradient{0}) = true
isreal(g::FADGradient) = isconstant(g)

==(a::FADGradient, b::FADGradient) = value(a) == value(b) && partials(a) == partials(b)

isequal(a::FADGradient, b::FADGradient) = isequal(value(a), value(b)) && isequal(partials(a), partials(b))

hash(g::FADGradient) = isconstant(g) ? hash(value(g)) : hash(value(g), hash(partials(g)))

function read{F<:FADGradient}(io::IO, ::Type{F})
    T = eltype(F)
    value = read(io, T)
    partials = ntuple(n->read(io, T), npartials(F))
    return F(value, partials)
end

function write(io::IO, g::FADGradient)
    write(io, value(g))
    for du in partials(g)
        write(io, du)
    end
end

########################
# Conversion/Promotion #
########################
zero{N,T,C}(g::FADGradient{N,T,C}) = FADGradient{N,T,C}(zero(value(g)), zero_partials(typeof(g)))
zero{N,T,C}(::Type{FADGradient{N,T,C}}) = FADGradient{N,T,C}(zero(T), zero_partials(FADGradient{N,T,C}))

one{N,T,C}(g::FADGradient{N,T,C}) = FADGradient{N,T,C}(one(value(g)), zero_partials(typeof(g)))
one{N,T,C}(::Type{FADGradient{N,T,C}}) = FADGradient{N,T,C}(one(T), zero_partials(FADGradient{N,T,C}))

for F in (:GradVec, :GradTup)
    @eval begin
        convert{N,T}(::Type{$(F){N,T}}, x::Real) = $(F){N,T}(x)
        convert{N,T}(::Type{$(F){N,T}}, g::$(F){N}) = $(F){N,T}(value(g), partials(g))
        convert{N,T}(::Type{$(F){N,T}}, g::$(F){N,T}) = g

        promote_rule{N,A,B}(::Type{$(F){N,A}}, ::Type{$(F){N,B}}) = $(F){N,promote_type(A,B)}
        promote_rule{N,A,B}(::Type{$(F){N,A}}, ::Type{B}) = $(F){N,promote_type(A,B)}
    end
end

convert{T<:Real}(::Type{T}, g::FADGradient{0}) = convert(T, value(g))
convert{T<:Real}(::Type{T}, g::FADGradient) = isconstant(g) ? convert(T, value(g)) : throw(InexactError())
convert(::Type{FADGradient}, g::FADGradient) = g
convert(::Type{FADGradient}, x::Real) = FADGradient(x)

#########################
# Math with FADGradient #
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

abs(g::FADGradient) = (value(g) >= 0) ? g : -g
abs2(g::FADGradient) = g*g

include("gradtup_math.jl")
include("gradvec_math.jl")

immutable GradientNumber{N,T,C} <: ForwardDiffNumber{N,T,C}
    value::T
    partials::Partials{T,C}
    GradientNumber(value, partials::Partials) = new(value, partials)
    GradientNumber(value, partials::Tuple) = new(value, Partials(partials))
    GradientNumber(value, partials::Vector) = new(value, Partials(partials))
end

typealias GradNumTup{N,T} GradientNumber{N,T,NTuple{N,T}}
typealias GradNumVec{N,T} GradientNumber{N,T,Vector{T}}

GradientNumber{N,T}(value::T, grad::NTuple{N,T}) = GradientNumber{N,T,NTuple{N,T}}(value, Partials(grad))
GradientNumber{T}(value::T, grad::T...) = GradientNumber(value, grad)

##############################
# Utility/Accessor Functions #
##############################
@inline partials(g::GradientNumber) = g.partials

@inline value(g::GradientNumber) = g.value
@inline grad(g::GradientNumber) = data(partials(g))
@inline hess(g::GradientNumber) = error("GradientNumbers do not store Hessian values")
@inline tens(g::GradientNumber) = error("GradientNumbers do not store tensor values")

@inline eltype{N,T,C}(::Type{GradientNumber{N,T,C}}) = T
@inline npartials{N,T,C}(::Type{GradientNumber{N,T,C}}) = N

#####################
# Generic Functions #
#####################
isconstant(g::GradientNumber) = iszero(partials(g))

==(a::GradientNumber, b::GradientNumber) = value(a) == value(b) && partials(a) == partials(b)

isequal(a::GradientNumber, b::GradientNumber) = isequal(value(a), value(b)) && isequal(partials(a), partials(b))

hash(g::GradientNumber) = isconstant(g) ? hash(value(g)) : hash(value(g), hash(partials(g)))
hash(g::GradientNumber, hsh::Uint64) = hash(hash(g), hsh)

function read{N,T,C}(io::IO, ::Type{GradientNumber{N,T,C}})
    value = read(io, T)
    partials = read(io, Partials{T,C}, N)
    return GradientNumber{N,T,C}(value, partials)
end

function write(io::IO, g::GradientNumber)
    write(io, value(g))
    write(io, partials(g))
end

########################
# Conversion/Promotion #
########################
@inline zero{N,T,C}(G::Type{GradientNumber{N,T,C}}) = G(zero(T), zero_partials(C, N))
@inline one{N,T,C}(G::Type{GradientNumber{N,T,C}}) = G(one(T), zero_partials(C, N))
@inline rand{N,T,C}(G::Type{GradientNumber{N,T,C}}) = G(rand(T), rand_partials(C, N))

for G in (:GradNumVec, :GradNumTup)
    @eval begin
        convert{N,A,B}(::Type{($G){N,A}}, g::($G){N,B}) = ($G){N,A}(value(g), partials(g))
        convert{N,T}(::Type{($G){N,T}}, g::($G){N,T}) = g

        promote_rule{N,A,B}(::Type{($G){N,A}}, ::Type{($G){N,B}}) = ($G){N,promote_type(A,B)}
        promote_rule{N,A,B<:Number}(::Type{($G){N,A}}, ::Type{B}) = ($G){N,promote_type(A,B)}
    end
end

convert{T<:Real}(::Type{T}, g::GradientNumber) = isconstant(g) ? convert(T, value(g)) : throw(InexactError())
convert(::Type{GradientNumber}, g::GradientNumber) = g
convert{N,T,C}(::Type{GradientNumber{N,T,C}}, x::Real) = GradientNumber{N,T,C}(x, zero_partials(C, N))
convert(::Type{GradientNumber}, x::Real) = GradientNumber(x)

############################
# Math with GradientNumber #
############################

# Addition/Subtraction #
#----------------------#
@inline +{N}(g1::GradientNumber{N}, g2::GradientNumber{N}) = promote_typeof(g1, g2)(value(g1)+value(g2), partials(g1)+partials(g2))
@inline +(g::GradientNumber, x::Real) = promote_typeof(g, x)(value(g)+x, partials(g))
@inline +(x::Real, g::GradientNumber) = g+x

@inline -(g::GradientNumber) = typeof(g)(-value(g), -partials(g))
@inline -{N}(g1::GradientNumber{N}, g2::GradientNumber{N}) = promote_typeof(g1, g2)(value(g1)-value(g2), partials(g1)-partials(g2))
@inline -(g::GradientNumber, x::Real) = promote_typeof(g, x)(value(g)-x, partials(g))
@inline -(x::Real, g::GradientNumber) = promote_typeof(g, x)(x-value(g), -partials(g))

# Multiplication #
#----------------#
@inline *(g::GradientNumber, x::Bool) = x ? g : (signbit(value(g))==0 ? zero(g) : -zero(g))
@inline *(x::Bool, g::GradientNumber) = g*x

@inline function *{N}(g1::GradientNumber{N}, g2::GradientNumber{N})
    g1_a, g2_a = value(g1), value(g2)
    return promote_typeof(g1, g2)(g1_a*g2_a, _mul_partials(partials(g1), partials(g2), g2_a, g1_a))
end

@inline *(g::GradientNumber, x::Real) = promote_typeof(g, x)(value(g)*x, partials(g)*x)
@inline *(x::Real, g::GradientNumber) = g*x

# Division #
#----------#
@inline function /{N}(g1::GradientNumber{N}, g2::GradientNumber{N})
    g1_a, g2_a = value(g1), value(g2)
    new_a = g1_a/g2_a
    return promote_typeof(g1, g2, new_a)(new_a, _div_partials(partials(g1), partials(g2), g1_a, g2_a))
end

@inline function /(x::Real, g::GradientNumber)
    a = value(g)
    new_a = x/a
    return promote_typeof(g, new_a)(new_a, _div_partials(x, partials(g), a))
end

@inline function /(g::GradientNumber, x::Real)
    new_a = value(g)/x
    return promote_typeof(g, new_a)(new_a, partials(g)/x)
end

# Exponentiation #
#----------------#
for f in (:^, :(NaNMath.pow))
    @eval begin
        @inline function ($f){N}(g1::GradientNumber{N}, g2::GradientNumber{N})
            g1_a, g2_a = value(g1), value(g2)    
            new_a = ($f)(g1_a, g2_a)
            powval = g2_a * ($f)(g1_a, g2_a-1)
            logval = new_a * log(g1_a)
            new_bs = _mul_partials(partials(g1), partials(g2), powval, logval)
            return promote_typeof(g1, g2, new_a)(new_a, new_bs)
        end

        @inline ($f)(::Base.MathConst{:e}, g::GradientNumber) = exp(g)
    end
    # generate redundant definitions to resolve ambiguity warnings
    for R in (:Integer, :Rational, :Real)
        @eval begin
            @inline function ($f)(g::GradientNumber, x::$R)
                a = value(g)
                powval = x*($f)(a, x-1)
                return promote_typeof(g, powval)(($f)(a, x), powval*partials(g))
            end

            @inline function ($f)(x::$R, g::GradientNumber)
                a = value(g)
                new_a = ($f)(x, a)
                logval = new_a*log(x)
                return promote_typeof(logval, g)(new_a, logval*partials(g))
            end
        end
    end
end

# Unary functions on GradientNumbers #
#------------------------------------#

# helper function to allow use of NaNMath
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

for fsym in fad_supported_univar_funcs
    a = :a
    new_a = :($(fsym)($a))
    deriv = Calculus.differentiate(new_a, a)

    @eval begin
        @inline function $(fsym)(g::GradientNumber)
            a = value(g)
            deriv = $deriv
            return promote_typeof(g, deriv)($new_a, deriv*partials(g))
        end
    end

    # extend corresponding NaNMath methods
    if fsym in (:sin, :cos, :tan,
                :asin, :acos, :acosh,
                :atanh, :log, :log2,
                :log10, :lgamma, :log1p)

        nan_fsym = Expr(:.,:NaNMath,Base.Meta.quot(fsym))
        nan_new_a = :($(nan_fsym)($a))
        nan_deriv = to_nanmath(deriv)

        @eval begin
            @inline function $(nan_fsym)(g::GradientNumber)
                a = value(g)
                deriv = $nan_deriv
                return promote_typeof(g, deriv)($nan_new_a, deriv*partials(g))
            end
        end
    end
end

# Special Cases #
#---------------#
@inline function exp(g::GradientNumber)
    deriv = exp(value(g))
    return promote_typeof(g, deriv)(deriv, deriv*partials(g))
end

@inline abs(g::GradientNumber) = (value(g) >= 0) ? g : -g
@inline abs2(g::GradientNumber) = g*g

@inline calc_atan2(y::GradientNumber, x::GradientNumber) = atan2(value(y), value(x))
@inline calc_atan2(y::Real, x::GradientNumber) = atan2(y, value(x))
@inline calc_atan2(y::GradientNumber, x::Real) = atan2(value(y), x)

for Y in (:Real, :GradientNumber), X in (:Real, :GradientNumber)
    if !(Y == :Real && X == :Real)
        @eval begin
            @inline function atan2(y::$Y, x::$X)
                z = y/x
                a = value(z)
                deriv = inv(one(a) + a*a)
                return promote_typeof(z, a, deriv)(calc_atan2(y, x), deriv*partials(z))
            end
        end
    end
end

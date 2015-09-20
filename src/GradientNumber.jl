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
@inline function gradnum_from_deriv(g::GradientNumber, new_a, deriv)
    G = promote_typeof(g, new_a, deriv)
    return G(new_a, deriv*partials(g))
end

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
    a1, a2 = value(g1), value(g2)
    return promote_typeof(g1, g2)(a1*a2, _mul_partials(partials(g1), partials(g2), a2, a1))
end

@inline *(g::GradientNumber, x::Real) = promote_typeof(g, x)(value(g)*x, partials(g)*x)
@inline *(x::Real, g::GradientNumber) = g*x

# Division #
#----------#
@inline function /{N}(g1::GradientNumber{N}, g2::GradientNumber{N})
    a1, a2 = value(g1), value(g2)
    div_a = a1/a2
    return promote_typeof(g1, g2, div_a)(div_a, _div_partials(partials(g1), partials(g2), a1, a2))
end

@inline function /(x::Real, g::GradientNumber)
    a = value(g)
    div_a = x/a
    deriv = -(div_a/a)
    return gradnum_from_deriv(g, div_a, deriv)
end

@inline function /(g::GradientNumber, x::Real)
    div_a = value(g)/x
    return promote_typeof(g, div_a)(div_a, partials(g)/x)
end

# Exponentiation #
#----------------#
for f in (:^, :(NaNMath.pow))
    @eval begin
        @inline function ($f){N}(g1::GradientNumber{N}, g2::GradientNumber{N})
            a1, a2 = value(g1), value(g2)    
            exp_a = ($f)(a1, a2)
            powval = a2 * exp_a/a1
            logval = exp_a * log(a1)
            new_bs = _mul_partials(partials(g1), partials(g2), powval, logval)
            return promote_typeof(g1, g2, exp_a)(exp_a, new_bs)
        end

        @inline ($f)(::Base.Irrational{:e}, g::GradientNumber) = exp(g)
    end

    # generate redundant definitions to resolve ambiguity warnings
    for R in (:Integer, :Rational, :Real)
        @eval begin
            @inline function ($f)(g::GradientNumber, x::$R)
                a = value(g)
                exp_a = ($f)(a, x)
                deriv = x*(exp_a/a)
                return gradnum_from_deriv(g, exp_a, deriv)
            end

            @inline function ($f)(x::$R, g::GradientNumber)
                a = value(g)
                exp_a = ($f)(x, a)
                deriv = exp_a*log(x)
                return gradnum_from_deriv(g, exp_a, deriv)
            end
        end
    end
end

# Unary functions on GradientNumbers #
#------------------------------------#
for fsym in auto_defined_unary_funcs
    a = :a
    new_a = :($(fsym)($a))
    deriv = Calculus.differentiate(new_a, a)

    @eval begin
        @inline function $(fsym)(g::GradientNumber)
            a = value(g)
            return gradnum_from_deriv(g, $new_a, $deriv)
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
                return gradnum_from_deriv(g, $nan_new_a, $nan_deriv)
            end
        end
    end
end

#################
# Special Cases #
#################

# Manually Optimized Functions #
#------------------------------#
@inline function exp(g::GradientNumber)
    exp_a = exp(value(g))
    return gradnum_from_deriv(g, exp_a, exp_a)
end

@inline function sqrt(g::GradientNumber)
    sqrt_a = sqrt(value(g))
    deriv = 0.5 / sqrt_a
    return gradnum_from_deriv(g, sqrt_a, deriv)
end

# Other Functions #
#-----------------#
@inline calc_atan2(y::GradientNumber, x::GradientNumber) = atan2(value(y), value(x))
@inline calc_atan2(y::Real, x::GradientNumber) = atan2(y, value(x))
@inline calc_atan2(y::GradientNumber, x::Real) = atan2(value(y), x)

for Y in (:Real, :GradientNumber), X in (:Real, :GradientNumber)
    if !(Y == :Real && X == :Real)
        @eval begin
            @inline function atan2(y::$Y, x::$X)
                z = y/x
                a = value(z)
                atan2_a = calc_atan2(y, x)
                deriv = inv(one(a) + a*a)
                return gradnum_from_deriv(z, atan2_a, deriv)
            end
        end
    end
end

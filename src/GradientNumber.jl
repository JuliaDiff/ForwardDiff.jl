immutable GradientNumber{N,T,C} <: ForwardDiffNumber{N,T,C}
    value::T
    grad::Partials{T,C}
    function GradientNumber(value::T, grad::Partials{T,C})
        @assert N > 0 "Number of partials used must be greater than 0"
        return new(value, grad)
    end
    function GradientNumber(value, grad)
        new_value = convert(T, value)
        new_grad = Partials{T,C}(grad)
        return GradientNumber{N,T,C}(new_value, new_grad)
    end
    function GradientNumber(value)
        grad = zero_partials(C, N)
        return GradientNumber{N,T,C}(value, grad)
    end
end

typealias GradNumTup{N,T} GradientNumber{N,T,NTuple{N,T}}
typealias GradNumVec{N,T} GradientNumber{N,T,Vector{T}}

GradientNumber{N}(::Type{Val{N}}, value, grad) = GradientNumber(Val{N}, value, Partials(grad))

function GradientNumber{N,T}(::Type{Val{N}}, value::T)
    return GradientNumber{N,T,NTuple{N}}(value)
end

@generated function GradientNumber{N,A,B,C}(::Type{Val{N}}, value::A, grad::Partials{B,C})
    T = promote_type(A, B)
    C2 = switch_eltype(C, T)
    return :(GradientNumber{N,$T,$C2}(value, grad))
end

GradientNumber{N}(value::Number, grad::NTuple{N}) = GradientNumber(Val{N}, value, grad)
GradientNumber(value::Number, grad...) = GradientNumber(value, grad)

##############################
# Utility/Accessor Functions #
##############################
value(g::GradientNumber) = g.value
grad(g::GradientNumber) = g.grad
hess(g::GradientNumber) = error("GradientNumbers do not store Hessian values")
tens(g::GradientNumber) = error("GradientNumbers do not store tensor values")

eltype{N,T,C}(::Type{GradientNumber{N,T,C}}) = T
npartials{N,T,C}(::Type{GradientNumber{N,T,C}}) = N

#####################
# Generic Functions #
#####################
isconstant(g::GradientNumber) = iszero(grad(g))

==(a::GradientNumber, b::GradientNumber) = value(a) == value(b) && grad(a) == grad(b)

isequal(a::GradientNumber, b::GradientNumber) = isequal(value(a), value(b)) && isequal(grad(a), grad(b))

hash(g::GradientNumber) = isconstant(g) ? hash(value(g)) : hash(value(g), hash(grad(g)))
hash(g::GradientNumber, hsh::Uint64) = hash(hash(g), hsh)

function read{N,T,C}(io::IO, ::Type{GradientNumber{N,T,C}})
    value = read(io, T)
    grad = read(io, Partials{T,C}, N)
    return GradientNumber{N,T,C}(value, grad)
end

function write(io::IO, g::GradientNumber)
    write(io, value(g))
    write(io, grad(g))
end

########################
# Conversion/Promotion #
########################
zero{N,T,C}(G::Type{GradientNumber{N,T,C}}) = G(zero(T))
one{N,T,C}(G::Type{GradientNumber{N,T,C}}) = G(one(T))
rand{N,T,C}(G::Type{GradientNumber{N,T,C}}) = G(rand(T), rand_partials(C, N))

for G in (:GradNumVec, :GradNumTup)
    @eval begin
        convert{N,A,B}(::Type{($G){N,A}}, g::($G){N,B}) = ($G){N,A}(value(g), grad(g))
        convert{N,T}(::Type{($G){N,T}}, g::($G){N,T}) = g

        promote_rule{N,A,B}(::Type{($G){N,A}}, ::Type{($G){N,B}}) = ($G){N,promote_type(A,B)}
        promote_rule{N,A,B}(::Type{($G){N,A}}, ::Type{B}) = ($G){N,promote_type(A,B)}
    end
end

convert{T<:Real}(::Type{T}, g::GradientNumber) = isconstant(g) ? convert(T, value(g)) : throw(InexactError())
convert(::Type{GradientNumber}, g::GradientNumber) = g
convert{N,T,C}(::Type{GradientNumber{N,T,C}}, x::Real) = GradientNumber{N,T,C}(x)
convert(::Type{GradientNumber}, x::Real) = GradientNumber(x)

############################
# Math with GradientNumber #
############################

# Addition/Subtraction #
#----------------------#
+{N}(a::GradientNumber{N}, b::GradientNumber{N}) = GradientNumber(Val{N}, value(a)+value(b), grad(a)+grad(b))
+{N}(g::GradientNumber{N}, x::Real) = GradientNumber(Val{N}, value(g)+x, grad(g))
+{N}(x::Real, g::GradientNumber{N}) = g+x

-{N}(g::GradientNumber{N}) = GradientNumber(Val{N}, -value(g), -grad(g))
-{N}(a::GradientNumber{N}, b::GradientNumber{N}) = GradientNumber(Val{N}, value(a)-value(b), grad(a)-grad(b))
-{N}(g::GradientNumber{N}, x::Real) = GradientNumber(Val{N}, value(g)-x, grad(g))
-{N}(x::Real, g::GradientNumber{N}) = GradientNumber(Val{N}, x-value(g), -grad(g))

# Multiplication #
#----------------#
*(g::GradientNumber, x::Bool) = x ? g : (signbit(value(g))==0 ? zero(g) : -zero(g))
*(x::Bool, g::GradientNumber) = g*x

function *{N}(a::GradientNumber{N}, b::GradientNumber{N})
    aval, bval = value(a), value(b)
    return GradientNumber(Val{N}, aval*bval, _mul_partials(grad(a), grad(b), bval, aval))
end

*{N}(g::GradientNumber{N}, x::Real) = GradientNumber(Val{N}, value(g)*x, grad(g)*x)
*{N}(x::Real, g::GradientNumber{N}) = g*x

# Division #
#----------#
function /{N}(a::GradientNumber{N}, b::GradientNumber{N})
    aval, bval = value(a), value(b)
    return GradientNumber(Val{N}, aval/bval, _div_partials(grad(a), grad(b), aval, bval))
end

function /{N}(x::Real, g::GradientNumber{N})
    gval = value(g)
    return GradientNumber(Val{N}, x/gval, _div_partials(x, grad(g), gval))   
end

/{N}(g::GradientNumber{N}, x::Real) = GradientNumber(Val{N}, value(g)/x, grad(g)/x)

# Exponentiation #
#----------------#
for f in (:^, :(NaNMath.pow))
    @eval function ($f){N}(a::GradientNumber{N}, b::GradientNumber{N})
        aval, bval = value(a), value(b)    
        result_val = ($f)(aval, bval)
        powval = bval * ($f)(aval, bval-1)
        logval = result_val * log(aval)
        result_grad = _mul_partials(grad(a), grad(b), powval, logval)
        return GradientNumber(Val{N}, result_val, result_grad)
    end

    @eval ($f)(::Base.MathConst{:e}, g::GradientNumber) = exp(g)

    # generate redundant definitions to resolve ambiguity warnings
    for R in (:Integer, :Rational, :Real)
        @eval function ($f){N}(g::GradientNumber{N}, x::$R)
            gval = value(g)
            powval = x*($f)(gval, x-1)
            return GradientNumber(Val{N}, ($f)(gval, x), powval*grad(g))
        end

        @eval function ($f){N}(x::$R, g::GradientNumber{N})
            gval = value(g)
            result_val = ($f)(x, gval)
            logval = result_val*log(x)
            return GradientNumber(Val{N}, result_val, logval*grad(g))
        end
    end
end

# from Calculus.jl #
#------------------#
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

for fsym in fad_supported_univar_funcs
    fsym == :exp && continue
    
    valexpr = :($(fsym)(gval))
    dfexpr = Calculus.differentiate(valexpr, :gval)

    @eval function $(fsym){N}(g::GradientNumber{N})
        gval = value(g)
        df = $dfexpr
        return GradientNumber(Val{N}, $valexpr, df*grad(g))
    end

    # extend corresponding NaNMath methods
    if fsym in (:sin, :cos, :tan,
                :asin, :acos, :acosh,
                :atanh, :log, :log2,
                :log10, :lgamma, :log1p)

        nan_fsym = Expr(:.,:NaNMath,Base.Meta.quot(fsym))
        nan_valexpr = :($(nan_fsym)(gval))
        nan_dfexpr = to_nanmath(dfexpr)

        @eval function $(nan_fsym){N}(g::GradientNumber{N})
            gval = value(g)
            df = $nan_dfexpr
            return GradientNumber(Val{N}, $nan_valexpr, df*grad(g))
        end
    end
end

function exp{N}(g::GradientNumber{N})
    df = exp(value(g))
    return GradientNumber(Val{N}, df, df*grad(g))
end

abs(g::GradientNumber) = (value(g) >= 0) ? g : -g
abs2(g::GradientNumber) = g*g

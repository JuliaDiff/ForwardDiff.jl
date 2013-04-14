importall Base

export ad, 
    ADForward, 
    value,
    deriv,
    gradient,
    show,
    float,
    convert,
    promote_rule,
    /,
    +,
    -,
    *,
    ^,
    isless,
    #<,
    ==,
    #<=,
    abs,
    max,
    min,
    sqrt,
    cbrt,
    sin,
    cos,
    tan,
    sinh,
    cosh,
    tanh,
    asin,
    acos,
    atan,
    log,
    log2,
    log10,
    exp,
    conj

immutable type ADForward{T<:Real,n} <: Number
    x::T
    d::Vector{T}
end

constant_ad{T<:Real}(x::T) = ADForward{T,0}(x, zeros(T, 1))
constant_ad{T<:Real}(x::T, n::Int) = ADForward{T,n}(x, zeros(T, n))

ADForward{T<:Real}        (x::T, d::T)         = ADForward{T,1}(x, [ d ])
ADForward{T<:Real}        (x::T, d::Vector{T}) = ADForward{T,length(d)}(x, d)
ADForward{T<:Real,S<:Real}(x::T, d::S)         = ADForward(x,zero(x))+ADForward(zero(d),  d )
ADForward{T<:Real,S<:Real}(x::T, d::Vector{S}) = ADForward{length(d)}(x,zeros(T, length(d)))+ADForward{length(d)}(zero(S), d)
ADForward{T<:Real}        (x::T)               = constant_ad(x)

function unitvec{T}(x::Vector{T},i)
    x = zeros(T,n)
    x[i] = one(T)
    return x
end

function ADForward{T<:Real}(x::Vector{T}) 
    n = length(x)
    X = Array(ADForward{T,n},n)
    for i=1:n
        X[i] = ADForward(x[i], unitvec(x,i))
    end
    return X
end

const ad = ADForward
zero{T,n}(::Type{ADForward{T,n}}) = ADForward{T,n}(zero(T), zeros(T, n))
one{T,n}(::Type{ADForward{T,n}}) = ADForward{T,n}(one(T), zeros(T, n))

value(x::ADForward) = x.x
value{T<:Real,n,m}(X::Array{ADForward{T,n},m}) = convert(Array{T,m}, X)

gradient{T}(x::ADForward{T}) = x.d
function gradient{T<:Real,n,m}(X::Array{ADForward{T,n},m})
    reshape([ x.d[i] for x=X,i=1:n ], tuple(arraysize(X)...,n))
end
gradient(x::Real) = zero(x)
const deriv = gradient

show(io::IO, z::ADForward) = print(io, "ADForward(",value(z),",",deriv(z),")")

function float{T<:Real,n}(x::ADForward{T,n}) 
    S = typeof(float(x.x))
    ADForward{S, n}(float(x.x), float(x.d))
end

convert{T<:Real,n}        (::Type{ADForward{T,n}}, x::ADForward{T,n}) = x
convert{T<:Real,S<:Real,n}(::Type{ADForward{T,n}}, x::ADForward{S,n}) = ADForward{T,n}(convert(T, x.x), convert(Vector{T}, x.d))
convert{T<:Real}          (::Type{T},              x::ADForward)      = convert(T, x.x)
convert{T<:Real,n}        (::Type{ADForward{T,n}}, x::T)              = ADForward{T,n}(x, zeros(T,n))
convert{T<:Real,S<:Real,n}(::Type{ADForward{T,n}}, x::S)              = ADForward{T,n}(convert(T, x), zeros(T,n))

promote_rule{T<:Real,n}(::Type{ADForward{T,n}}, ::Type{T}) = ADForward{T,n}
promote_rule{T<:Real,S<:Real,n}(::Type{ADForward{T,n}}, ::Type{ADForward{S,n}}) = ADForward{promote_type(T,S),n}
promote_rule{T<:Real,S<:Real,n}(::Type{ADForward{T,n}}, ::Type{S}) = ADForward{promote_type(T,S),n}

/{T<:Real,n}(x::ADForward{T,n}, y::ADForward{T,n}) = ADForward{T,n}(x.x/y.x, (x.d*y.x-x.x*y.d)/(y.x*y.x))
+{T<:Real,n}(x::ADForward{T,n}, y::ADForward{T,n}) = ADForward{T,n}(x.x+y.x, x.d+y.d)
-{T<:Real,n}(x::ADForward{T,n}) = ADForward{T,n}(-x.x, -x.d)
-{T<:Real,n}(x::ADForward{T,n}, y::ADForward{T,n}) = ADForward{T,n}(x.x-y.x, x.d-y.d)
*{T<:Real,n}(x::ADForward{T,n}, y::ADForward{T,n}) = ADForward{T,n}(x.x*y.x, x.d*y.x+x.x*y.d)
^{T<:Real,n}(x::ADForward{T,n}, y::ADForward{T,n}) = x.x == 0 ? zero(x) : ADForward{T,n}(x.x^y.x, x.x^y.x*(x.d/x.x*y.x + y.d*log(x.x)))

<{T<:Real,n}(x::ADForward{T,n}, y::ADForward{T,n}) = x.x < y.x
=={T<:Real,n}(x::ADForward{T,n}, y::ADForward{T,n}) = x.x == y.x
<={T<:Real,n}(x::ADForward{T,n}, y::ADForward{T,n}) = !(y.x < x.x)

isless{T<:Real,n}(x::ADForward{T,n}, y::ADForward{T,n}) = isless(x.x, y.x)
isless{T<:Real,n}(x::ADForward{T,n}, y::Real)           = isless(x.x, y)
isless{T<:Real,n}(y::Real, x::ADForward{T,n})           = isless(y, x.x)

abs(x::ADForward) = x.x>=zero(x.x) ? x : -x

max{T<:Real,n}        (x::ADForward{T,n}, y::ADForward{T,n}) = x.x > y.x ? x : y
max{S<:Real,T<:Real,n}(x::ADForward{S,n}, y::ADForward{T,n}) = max(promote(x,y)...)
max{S<:Real,T<:Real,n}(x::ADForward{T,n}, y::S)              = max(promote(x,y)...)
max{S<:Real,T<:Real,n}(x::S, y::ADForward{T,n})              = max(promote(x,y)...)

min{T<:Real,n}(x::ADForward{T,n}, y::ADForward{T,n}) = x.x < y.x ? x : y
min{S<:Real,T<:Real,n}(x::ADForward{S,n}, y::ADForward{T,n}) = min(promote(x,y)...)
min{S<:Real,T<:Real,n}(x::ADForward{T,n}, y::S)              = min(promote(x,y)...)
min{S<:Real,T<:Real,n}(x::S, y::ADForward{T,n})              = min(promote(x,y)...)

sqrt{T<:Real,n}(x::ADForward{T,n}) = ADForward{T,n}(sqrt(x.x), 1/(2*sqrt(x.x))*x.d)
cbrt{T<:Real,n}(x::ADForward{T,n}) = ADForward{T,n}(cbrt(x.x), 1/(3*square(cbrt(x.x)))*x.d)

sin{T<:Real,n}(x::ADForward{T,n}) = ADForward{T,n}(sin(x.x), cos(x.x)*x.d)
cos{T<:Real,n}(x::ADForward{T,n}) = ADForward{T,n}(cos(x.x), -sin(x.x)*x.d)
tan{T<:Real,n}(x::ADForward{T,n}) = ADForward{T,n}(tan(x.x), square(sec(x.x))*x.d)

sinh{T<:Real,n}(x::ADForward{T,n}) = ADForward{T,n}(sinh(x.x), cosh(x.x)*x.d)
cosh{T<:Real,n}(x::ADForward{T,n}) = ADForward{T,n}(cosh(x.x), sinh(x.x)*x.d)
tanh{T<:Real,n}(x::ADForward{T,n}) = ADForward{T,n}(tanh(x.x), 1-square(tanh(x.x))*x.d)

asin{T<:Real,n}(x::ADForward{T,n}) = ADForward{T,n}(asin(x.x), 1/sqrt(1-square(x.x))*x.d)
acos{T<:Real,n}(x::ADForward{T,n}) = ADForward{T,n}(acos(x.x),  -1/sqrt(1-square(x.x))*x.d)
atan{T<:Real,n}(x::ADForward{T,n}) = ADForward{T,n}(atan(x.x), 1/(1+square(x.x))*x.d)

log{T<:Real,n}(x::ADForward{T,n}) = ADForward{T,n}(log(x.x), x.d/x.x)
log2{T<:Real,n}(x::ADForward{T,n}) = log(x)/log(2.0)
log10{T<:Real,n}(x::ADForward{T,n}) = log(x)/log(10.0)
exp{T<:Real,n}(x::ADForward{T,n}) = ADForward{T,n}(exp(x.x), exp(x.x)*x.d)

#conj{n}(x::ADForward{Complex,n}) = error("Derivative of complex conjugate is undefined")
conj{T<:Real,n}(x::ADForward{T,n}) = x


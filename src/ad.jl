typealias ADForwardBase Real

type ADForward{T<:ADForwardBase,n}<:ADForwardBase
    x::T
    d::Vector{T}
end

Value(x::ADForward) = x.x
Value{T<:ADForwardBase,n,m}(X::Array{ADForward{T,n},m}) = convert(Array{T,m}, X)
Gradient(x::ADForward) = x.d
function Gradient{T<:ADForwardBase,n,m}(X::Array{ADForward{T,n},m})
    D = reshape([ x.d[i] | x=X,i=1:n ], tuple(arraysize(X)...,n))
end

function float{T<:ADForwardBase,n}(x::ADForward{T,n}) 
    S=typeof(float(x.x))
    ADForward{S, n}(float(x.x), float(x.d))
end

constant_ad{T<:ADForwardBase}(x::T)=ADForward{T,0}(x, zeros(T, 0))
constant_ad{T<:ADForwardBase}(x::T, n::Int)=ADForward{T,n}(x, zeros(T, n))

ad{T<:ADForwardBase}(x::T, d::T)=ADForward{T,1}(x, [ d ])
ad{T<:ADForwardBase}(x::T, d::Vector{T})=ADForward{T,length(d)}(x, d)
ad{T<:ADForwardBase,S<:ADForwardBase}(x::T, d::S)=ad(x,zero(x))+ad(zero(d),  d )
ad{T<:ADForwardBase,S<:ADForwardBase}(x::T, d::Vector{S})=ADForward{length(d)}(x,zeros(T, length(d)))+ADForward{length(d)}(zero(S), d)
ad{T<:ADForwardBase}(x::T) = constant_ad(x)
function ad{T<:ADForwardBase}(x::Vector{T}) 
    n=length(x)
    function unitvec(i)
        x=zeros(T,n)
        x[i]=one(T)
        return x
    end
    X=Array(ADForward{T,n},n)
    for i=1:n
        X[i]=ad(x[i], unitvec(i))
    end
    return X
end

convert{T<:ADForwardBase,n}(::Type{ADForward{T,n}}, x::ADForward{T,n}) = x
convert{T<:ADForwardBase,S<:ADForwardBase,n}(::Type{ADForward{T,n}}, x::ADForward{S,n}) = ADForward{T,n}(convert(T, x.x), convert(Vector{T}, x.d))
convert{T<:ADForwardBase}(::Type{T}, x::ADForward) = convert(T, x.x)
convert{T<:ADForwardBase,n}(::Type{ADForward{T,n}}, x::T) = ADForward{T,n}(x, zeros(T,n))
convert{T<:ADForwardBase,S<:ADForwardBase,n}(::Type{ADForward{T,n}}, x) = ADForward{T,n}(convert(T, x), zeros(T,n))

promote_rule{T<:ADForwardBase,n}(::Type{ADForward{T,n}}, ::Type{T}) = ADForward{T,n}
promote_rule{T<:ADForwardBase,S<:ADForwardBase,n}(::Type{ADForward{T,n}}, ::Type{ADForward{S,n}}) = ADForward{promote_type(T,S),n}
promote_rule{T<:ADForwardBase,S<:ADForwardBase,n}(::Type{ADForward{T,n}}, ::Type{S}) = ADForward{promote_type(T,S),n}

/{T<:ADForwardBase,n}(x::ADForward{T,n}, y::ADForward{T,n}) = ADForward{T,n}(x.x/y.x, (x.d*y.x-x.x*y.d)/(y.x*y.x))
+{T<:ADForwardBase,n}(x::ADForward{T,n}, y::ADForward{T,n}) = ADForward{T,n}(x.x+y.x, x.d+y.d)
-{T<:ADForwardBase,n}(x::ADForward{T,n}) = ADForward{T,n}(-x.x, -x.d)
-{T<:ADForwardBase,n}(x::ADForward{T,n}, y::ADForward{T,n}) = ADForward{T,n}(x.x-y.x, x.d-y.d)
*{T<:ADForwardBase,n}(x::ADForward{T,n}, y::ADForward{T,n}) = ADForward{T,n}(x.x*y.x, x.d*y.x+x.x*y.d)
^{T<:ADForwardBase,n}(x::ADForward{T,n}, y::ADForward{T,n}) = x.x == 0 ? zero(x) : ADForward{T,n}(x.x^y.x, x.x^y.x*(x.d/x.x*y.x + y.d*log(x.x)))

<{T<:ADForwardBase,n}(x::ADForward{T,n}, y::ADForward{T,n}) = x.x < y.x
=={T<:ADForwardBase,n}(x::ADForward{T,n}, y::ADForward{T,n}) = x.x == y.x
<={T<:ADForwardBase,n}(x::ADForward{T,n}, y::ADForward{T,n}) = !(y.x < x.x)

abs(x::ADForward) = x.x>=zero(x.x) ? x : -x

max{T<:ADForwardBase,n}(x::ADForward{T,n}, y::ADForward{T,n}) = x.x > y.x ? x : y
max{T<:ADForwardBase,n}(x::ADForward{T,n}, y) = max(promote(x,y)...)
max{T<:ADForwardBase,n}(x, y::ADForward{T,n}) = max(promote(x,y)...)

min{T<:ADForwardBase,n}(x::ADForward{T,n}, y::ADForward{T,n}) = x.x < y.x ? x : y
min{T<:ADForwardBase,n}(x::ADForward{T,n}, y) = min(promote(x,y)...)
min{T<:ADForwardBase,n}(x, y::ADForward{T,n}) = min(promote(x,y)...)

sqrt{T<:ADForwardBase,n}(x::ADForward{T,n}) = ADForward{T,n}(sqrt(x.x), 1/(2*sqrt(x.x))*x.d)
cbrt{T<:ADForwardBase,n}(x::ADForward{T,n}) = ADForward{T,n}(cbrt(x.x), 1/(3*square(cbrt(x.x)))*x.d)

sin{T<:ADForwardBase,n}(x::ADForward{T,n}) = ADForward{T,n}(sin(x.x), cos(x.x)*x.d)
cos{T<:ADForwardBase,n}(x::ADForward{T,n}) = ADForward{T,n}(cos(x.x), -sin(x.x)*x.d)
tan{T<:ADForwardBase,n}(x::ADForward{T,n}) = ADForward{T,n}(tan(x.x), square(sec(x.x))*x.d)

sinh{T<:ADForwardBase,n}(x::ADForward{T,n}) = ADForward{T,n}(sinh(x.x), cosh(x.x)*x.d)
cosh{T<:ADForwardBase,n}(x::ADForward{T,n}) = ADForward{T,n}(cosh(x.x), sinh(x.x)*x.d)
tanh{T<:ADForwardBase,n}(x::ADForward{T,n}) = ADForward{T,n}(tanh(x.x), 1-square(tanh(x.x))*x.d)

asin{T<:ADForwardBase,n}(x::ADForward{T,n}) = ADForward{T,n}(asin(x.x), 1/sqrt(1-square(x.x))*x.d)
acos{T<:ADForwardBase,n}(x::ADForward{T,n}) = ADForward{T,n}(acos(x.x),  -1/sqrt(1-square(x.x))*x.d)
atan{T<:ADForwardBase,n}(x::ADForward{T,n}) = ADForward{T,n}(atan(x.x), 1/(1+square(x.x))*x.d)

log{T<:ADForwardBase,n}(x::ADForward{T,n}) = ADForward{T,n}(log(x.x), x.d/x.x)
log2{T<:ADForwardBase,n}(x::ADForward{T,n}) = log(x)/log(2.0)
log10{T<:ADForwardBase,n}(x::ADForward{T,n}) = log(x)/log(10.0)
exp{T<:ADForwardBase,n}(x::ADForward{T,n}) = ADForward{T,n}(exp(x.x), exp(x.x)*x.d)

#conj{n}(x::ADForward{Complex,n}) = error("Derivative of complex conjugate is undefined")
conj{T<:ADForwardBase,n}(x::ADForward{T,n}) = x


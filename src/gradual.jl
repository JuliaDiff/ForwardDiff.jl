immutable GraDual{T<:Real, n} <: Number
    v::T
    g::Vector{T}
end

GraDual{T<:Real} (v::T, g::Vector{T}) = GraDual{T, length(g)}(v, g)

function GraDual{T<:Real}(v::Vector{T})
  n = length(v)
  G = Array(GraDual{T, n}, n)
  for i=1:n
    g = zeros(T, n)
    g[i] = one(T)
    G[i] = GraDual{T, n}(v[i], g)
  end
  return G
end

GraDual{T<:Real}(v::T, g::T) = GraDual{T, 1}(v, [g])
GraDual{T<:Real}(v::T) = GraDual{T, 1}(v, [one(T)])

const gradual = GraDual

zero{T, n}(::Type{GraDual{T, n}}) = GraDual{T, n}(zero(T), zeros(T, n))
one{T, n}(::Type{GraDual{T, n}}) = GraDual{T, n}(one(T), zeros(T, n))

value(x::GraDual) = x.v
value{T<:Real, n}(X::Vector{GraDual{T, n}}) = [x.v for x in X]

grad(x::GraDual) = x.g
function grad{T<:Real, n}(X::Vector{GraDual{T, n}})
  m = length(X)
  reshape([x.g[i] for x in X, i in 1:n], m, n)
end

const jacobian = grad

convert{T<:Real, n}(::Type{GraDual{T, n}}, x::GraDual{T, n}) = x
convert{T<:Real, n}(::Type{GraDual{T, n}}, x::T) = GraDual{T, n}(x, zeros(T,n))
convert{T<:Real, S<:Real, n}(::Type{GraDual{T, n}}, x::S) = 
  GraDual{T, n}(convert(T, x), zeros(T, n))
convert{T<:Real, S<:Real, n}(::Type{GraDual{T, n}}, x::GraDual{S, n}) = 
  GraDual{T, n}(convert(T, x.v), convert(Vector{T}, x.g))
convert{T<:Real, S<:Real, n}(::Type{T}, x::GraDual{S, n}) =
  (grad(x) == zeros(S, n) ? convert(T, x.v) : throw(InexactError()))
  
promote_rule{T<:Real, n}(::Type{GraDual{T, n}}, ::Type{T}) = GraDual{T, n}
promote_rule{T<:Real, S<:Real, n}(::Type{GraDual{T, n}}, ::Type{S}) = 
  GraDual{promote_type(T, S), n}
promote_rule{T<:Real, S<:Real, n}(::Type{GraDual{T, n}}, ::Type{GraDual{S,n}}) =
  GraDual{promote_type(T, S), n}

isgradual(x::GraDual) = true
isgradual(x::Number) = false

isconstant{T<:Real, n}(x::GraDual{T, n}) = (grad(x) == zeros(T, n))
iszero{T<:Real, n}(x::GraDual{T, n}) = isconstant(x) && (x.v == zero(T))
isfinite{T<:Real, n}(x::GraDual{T, n}) =
  isfinite(x.v) && (isfinite(x.g) == ones(n))

=={T<:Real, n}(x1::GraDual{T, n}, x2::GraDual{T, n}) = 
  (x1.v == x2.v) && (x1.g == x2.g)
  
show(io::IO, x::GraDual) = print(io, "GraDual(", value(x), ",", grad(x), ")")

function conj{T<:Real, n}(x::GraDual{T, n})
  g = Array(T, n)
  for i in 1:n
    g[i] = -x.g[i]
  end
  GraDual{T, n}(x.v, g)
end

function abs{T<:Real, n}(x::GraDual{T, n})
  y = Array(T, n)
  vsq = x.v*x.v
  for i in 1:n
    y[i] = sqrt(vsq+x.g[i]*x.g[i])
  end
  y
end

function abs2{T<:Real, n}(x::GraDual{T, n})
  y = Array(T, n)
  vsq = x.v*x.v
  for i in 1:n
    y[i] = vsq+x.g[i]*x.g[i]
  end
  y
end

inv{T<:Real, n}(x::GraDual{T, n})  = conj(x)/(x.v*x.v)

+{T<:Real,n}(x1::GraDual{T,n}, x2::GraDual{T,n}) =
  GraDual{T,n}(x1.v+x2.v, x1.g+x2.g)

-{T<:Real, n}(x::GraDual{T, n}) = GraDual{T,n}(-x.v, -x.g)
-{T<:Real, n}(x::GraDual{T, n}, y::GraDual{T,n}) =
  GraDual{T,n}(x.v-y.v, x.g-y.g)

function *{T<:Real, n}(x1::GraDual{T, n}, x2::GraDual{T, n})
  g = Array(T, n)
  for i in 1:n
    g[i] = x1.g[i]*x2.v+x1.v*x2.g[i]
  end
  GraDual{T, n}(x1.v*x2.v, g)
end

*{T<:Real, n}(x1::T, x2::GraDual{T, n}) = GraDual{T, n}(x1*x2.v, x1*x2.g)
*{T<:Real, n}(x1::GraDual{T, n}, x2::T) = GraDual{T, n}(x2*x1.v, x2*x1.g)

function /{T<:Real,n}(x1::GraDual{T,n}, x2::GraDual{T,n})
  g = Array(T, n)
  vsq = x2.v*x2.v
  for i in 1:n
    g[i] = (x1.g[i]*x2.v-x1.v*x2.g[i])/vsq
  end
  GraDual{T, n}(x1.v/x2.v, g)
end

/{T<:Real, n}(x1::T, x2::GraDual{T, n}) = x1*inv(x2)
/{T<:Real, n}(x1::GraDual{T, n}, x2::T) = GraDual{T, n}(x1.v/x2, x1.g/x2)

sqrt{T<:Real, n}(x::GraDual{T, n}) = GraDual{T, n}(sqrt(x.v), x.g/(2*sqrt(x.v)))
cbrt{T<:Real, n}(x::GraDual{T, n}) =
  GraDual{T, n}(cbrt(x.v), x.g/(3*square(cbrt(x.v))))

function ^{T<:Real, n}(x1::GraDual{T, n}, x2::GraDual{T, n})
  g = Array(T, n)
  v = x1^x2
  for i in 1:n
    g[i] = x1.g[i]*x2.v*(x1.v^(x2.v-1))+x2.g[i]*(x1.v^x2.v)*log(x1.v)
  end
  GraDual{T, n}(v, g)
end

exp{T<:Real, n}(x::GraDual{T, n}) = GraDual{T, n}(exp(x.v), exp(x.v)*x.g)
log{T<:Real, n}(x::GraDual{T, n}) = GraDual{T, n}(log(x.v), x.g/x.v)
log2{T<:Real, n}(x::GraDual{T, n}) = log(x)/oftype(T, 0.6931471805599453)
log10{T<:Real, n}(x::GraDual{T, n}) = log(x)/oftype(T, 2.302585092994046)

sin{T<:Real, n}(x::GraDual{T, n}) = GraDual{T, n}(sin(x.v), cos(x.v)*x.g)
cos{T<:Real, n}(x::GraDual{T, n}) = GraDual{T, n}(cos(x.v), -sin(x.v)*x.g)
tan{T<:Real, n}(x::GraDual{T, n}) =
  GraDual{T, n}(tan(x.v), square(sec(x.v))*x.g)

asin{T<:Real, n}(x::GraDual{T, n}) =
  GraDual{T, n}(asin(x.v), x.g/sqrt(1-square(x.v)))
acos{T<:Real, n}(x::GraDual{T, n}) =
  GraDual{T, n}(acos(x.v), -x.g/sqrt(1-square(x.v)))
atan{T<:Real, n}(x::GraDual{T, n}) =
  GraDual{T, n}(atan(x.v), x.g/(1+square(x.v)))

sinh{T<:Real, n}(x::GraDual{T, n}) = GraDual{T, n}(sinh(x.v), cosh(x.v)*x.g)
cosh{T<:Real, n}(x::GraDual{T, n}) = GraDual{T, n}(cosh(x.v), sinh(x.v)*x.g)
tanh{T<:Real, n}(x::GraDual{T, n}) =
  GraDual{T, n}(tanh(x.v), 1-square(tanh(x.v))*x.g)

asinh{T<:Real, n}(x::GraDual{T, n}) =
  GraDual{T, n}(asinh(x.v), x.g/sqrt(x.v*x.v+1))
acosh{T<:Real, n}(x::GraDual{T, n}) =
  GraDual{T, n}(acosh(x.v), x.g/sqrt(x.v*x.v-1))
atanh{T<:Real, n}(x::GraDual{T, n}) = GraDual{T, n}(atanh(x.v), x.g/(1-x.v*x.v))

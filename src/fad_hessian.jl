immutable FADHessian{T<:Real, n} <: Number
    d::GraDual{T, n} 
    h::Vector{T}
end

FADHessian{T<:Real, n} (d::GraDual{T, n}, h::Vector{T}) =
  FADHessian{T, length(d.g)}(d, h)

FADHessian{T<:Real, n} (d::GraDual{T, n}) =
  FADHessian{T, length(d.g)}(d, zeros(T, convert(Int, n*(n+1)/2)))

function FADHessian{T<:Real}(v::Vector{T})
  n = length(v)
  H = Array(FADHessian{T, n}, n)
  for i=1:n
    g = zeros(T, n)
    g[i] = one(T)
    H[i] =
      FADHessian(GraDual{T, n}(v[i], g), zeros(T, convert(Int, n*(n+1)/2)))
  end
  return H
end

zero{T, n}(::Type{FADHessian{T, n}}) =
  FADHessian(GraDual{T, n}(zero(T), zeros(T, n)),
  zeros(T, convert(Int, n*(n+1)/2)))
one{T, n}(::Type{FADHessian{T, n}}) =
  FADHessian(GraDual{T, n}(one(T), zeros(T, n)),
  zeros(T, convert(Int, n*(n+1)/2)))

value(x::FADHessian) = x.d.v
value{T<:Real, n}(X::Vector{FADHessian{T, n}}) = [x.d.v for x in X]

grad(x::FADHessian) = x.d.g
function grad{T<:Real, n}(X::Vector{FADHessian{T, n}})
  m = length(X)
  reshape([x.d.g[i] for x in X, i in 1:n], m, n)
end

function hessian{T<:Real, n}(x::FADHessian{T, n})
  y = Array(T, n, n)
  k = 1
  
  for i in 1:n
    for j in 1:i
      y[i, j] = x.h[k]
     k += 1
    end
  end

  for i in 1:n
    for j in (i+1):n
      y[i, j] = y[j, i]
    end
  end
  
  y
end

convert{T<:Real, n}(::Type{FADHessian{T, n}}, x::FADHessian{T, n}) = x
convert{T<:Real, n}(::Type{FADHessian{T, n}}, x::T) =
  FADHessian(GraDual{T, n}(x, zeros(T, n)), zeros(T, convert(Int, n*(n+1)/2)))
convert{T<:Real, S<:Real, n}(::Type{FADHessian{T, n}}, x::S) = 
  FADHessian(GraDual{T, n}(convert(T, x), zeros(T, n)),
  zeros(T, convert(Int, n*(n+1)/2)))
convert{T<:Real, S<:Real, n}(::Type{FADHessian{T, n}}, x::FADHessian{S, n}) =
  FADHessian(GraDual{T, n}(convert(T, x.d.v), convert(Vector{T}, x.d.g)),
  convert(Vector{T}, x.h))
convert{T<:Real, S<:Real, n}(::Type{T}, x::FADHessian{S, n}) =
  ((x.d.g == zeros(S, n) && x.h == zeros(S, convert(Int, n*(n+1)/2))) ? 
  convert(T, x.v) : throw(InexactError()))

promote_rule{T<:Real, n}(::Type{FADHessian{T, n}}, ::Type{T}) = FADHessian{T, n}
promote_rule{T<:Real, S<:Real, n}(::Type{FADHessian{T, n}}, ::Type{S}) = 
  FADHessian{promote_type(T, S), n}
promote_rule{T<:Real, S<:Real, n}(::Type{FADHessian{T, n}}, 
  ::Type{FADHessian{S,n}}) = FADHessian{promote_type(T, S), n}

isfadhessian(x::FADHessian) = true
isfadhessian(x::Number) = false

isconstant{T<:Real, n}(x::FADHessian{T, n}) =
  (x.d.g == zeros(T, n) && x.h == zeros(T, convert(Int, n*(n+1)/2)))
iszero{T<:Real, n}(x::FADHessian{T, n}) = isconstant(x) && (x.d.v == zero(T))
isfinite{T<:Real, n}(x::FADHessian{T, n}) =
  (isfinite(x.d.v) && isfinite(x.d.g) == ones(n) &&
  x.h == ones(convert(Int, n*(n+1)/2)))

=={T<:Real, n}(x1::FADHessian{T, n}, x2::FADHessian{T, n}) = 
  (x1.d.v == x2.d.v) && (x1.d.g == x2.d.g) && (x1.h == x2.h)
  
show(io::IO, x::FADHessian) =
  print(io, "FADHessian(\nvalue:\n", value(x),
  "\n\ngrad:\n", grad(x),
  "\n\nHessian:\n", hessian(x),"\n)")

function +{T<:Real,n}(x1::FADHessian{T, n}, x2::FADHessian{T, n})
  FADHessian{T, n}(x1.d+x2.d, x1.h+x2.h)
end

-{T<:Real, n}(x::FADHessian{T, n}) = FADHessian{T,n}(-x.d, -x.h)
-{T<:Real, n}(x::FADHessian{T, n}, y::FADHessian{T, n}) =
  FADHessian{T,n}(x.d-y.d, x.h-y.h)

function *{T<:Real, n}(x1::FADHessian{T, n}, x2::FADHessian{T, n})
  h = Array(T, convert(Int, n*(n+1)/2))
  k = 1
  for i in 1:n
    for j in 1:i
      h[k] = 
        x1.h[k]*x2.d.v+x1.d.g[i]*x2.d.g[j]+x1.d.g[j]*x2.d.g[i]+x1.d.v*x2.h[k]
      k += 1
    end
  end
  FADHessian{T, n}(x1.d*x2.d, h)
end

*{T<:Real, n}(x1::T, x2::FADHessian{T, n}) = FADHessian{T, n}(x1*x2.d, x1*x2.h)
*{T<:Real, n}(x1::FADHessian{T, n}, x2::T) = FADHessian{T, n}(x2*x1.d, x2*x1.h)

function /{T<:Real, n}(x1::FADHessian{T, n}, x2::FADHessian{T, n})
  h = Array(T, convert(Int, n*(n+1)/2))
  k = 1
  for i in 1:n
    for j in 1:i
      h[k] = ((2*x1.d.v*x2.d.g[j]*x2.d.g[i]+x2.d.v*x2.d.v*x1.h[k]
        -x2.d.v*(x1.d.g[i]*x2.d.g[j]+x1.d.g[j]*x2.d.g[i]+x1.d.v*x2.h[k]))
        /(x2.d.v*x2.d.v*x2.d.v))
      k += 1
    end
  end
  FADHessian{T, n}(x1.d/x2.d, h)
end

function sqrt{T<:Real, n}(x::FADHessian{T, n})
  h = Array(T, convert(Int, n*(n+1)/2))
  k = 1
  for i in 1:n
    for j in 1:i
      h[k] = (-x.d.g[i]*x.d.g[j]+2*x.d.v*x.h[k])/(4*(x.d.v^(1.5)))
      k += 1
    end
  end
  FADHessian{T, n}(sqrt(x.d), h)
end

function cbrt{T<:Real, n}(x::FADHessian{T, n})
  h = Array(T, convert(Int, n*(n+1)/2))
  k = 1
  for i in 1:n
    for j in 1:i
      h[k] = ((-2*x.d.g[i]*x.d.g[j]+3*x.d.v*x.h[k])
        /(9*cbrt(x.d.v*x.d.v*x.d.v*x.d.v*x.d.v)))
      k += 1
    end
  end
  FADHessian{T, n}(cbrt(x.d), h)
end

function ^{T<:Real, n}(x1::FADHessian{T, n}, x2::FADHessian{T, n})
  h = Array(T, convert(Int, n*(n+1)/2))
  k = 1
  for i in 1:n
    for j in 1:i
      h[k] = ((x1.d.v^(x2.d.v-2))*(x2.d.v*x2.d.v*x1.d.g[i]*x1.d.g[j]
      +x2.d.v*(x1.d.g[j]*(-x1.d.g[i]+x1.d.v*log(x1.d.v)*x2.d.g[i])
      +x1.d.v*(log(x1.d.v)*x1.d.g[i]*x2.d.g[j]+x1.h[k]))
      +x1.d.v*(x1.d.g[j]*x2.d.g[i]+x2.d.g[j]*(x1.d.g[i]
      +x1.d.v*log(x1.d.v)*log(x1.d.v)*x2.d.g[i])
      +x1.d.v*log(x1.d.v)*x2.h[k])))
      k += 1
    end
  end
  FADHessian{T, n}(x1.d^x2.d, h)
end

function exp{T<:Real, n}(x::FADHessian{T, n})
  h = Array(T, convert(Int, n*(n+1)/2))
  k = 1
  for i in 1:n
    for j in 1:i
      h[k] = exp(x.d.v)*(x.d.g[i]*x.d.g[j]+x.h[k])
      k += 1
    end
  end
  FADHessian{T, n}(exp(x.d), h)
end

function log{T<:Real, n}(x::FADHessian{T, n})
  h = Array(T, convert(Int, n*(n+1)/2))
  k = 1
  for i in 1:n
    for j in 1:i
      h[k] = (x.d.v*x.h[k]-x.d.g[i]*x.d.g[j])/(x.d.v*x.d.v)
      k += 1
    end
  end
  FADHessian{T, n}(log(x.d), h)
end

function log2{T<:Real, n}(x::FADHessian{T, n})
  h = Array(T, convert(Int, n*(n+1)/2))
  k = 1
  for i in 1:n
    for j in 1:i
      h[k] = ((x.d.v*x.h[k]-x.d.g[i]*x.d.g[j])
      /(x.d.v*x.d.v*oftype(T, 0.6931471805599453)))
      k += 1
    end
  end
  FADHessian{T, n}(log2(x.d), h)
end

function log10{T<:Real, n}(x::FADHessian{T, n})
  h = Array(T, convert(Int, n*(n+1)/2))
  k = 1
  for i in 1:n
    for j in 1:i
      h[k] = ((x.d.v*x.h[k]-x.d.g[i]*x.d.g[j])
      /(x.d.v*x.d.v*oftype(T, 2.302585092994046)))
      k += 1
    end
  end
  FADHessian{T, n}(log10(x.d), h)
end

function sin{T<:Real, n}(x::FADHessian{T, n})
  h = Array(T, convert(Int, n*(n+1)/2))
  k = 1
  for i in 1:n
    for j in 1:i
      h[k] = (-sin(x.d.v)*x.d.g[i]*x.d.g[j]+cos(x.d.v)*x.h[k])
      k += 1
    end
  end
  FADHessian{T, n}(sin(x.d), h)
end

function cos{T<:Real, n}(x::FADHessian{T, n})
  h = Array(T, convert(Int, n*(n+1)/2))
  k = 1
  for i in 1:n
    for j in 1:i
      h[k] = (-cos(x.d.v)*x.d.g[i]*x.d.g[j]-sin(x.d.v)*x.h[k])
      k += 1
    end
  end
  FADHessian{T, n}(cos(x.d), h)
end

function tan{T<:Real, n}(x::FADHessian{T, n})
  h = Array(T, convert(Int, n*(n+1)/2))
  k = 1
  for i in 1:n
    for j in 1:i
      h[k] = sec(x.d.v)*sec(x.d.v)*(2*tan(x.d.v)*x.d.g[i]*x.d.g[j]+x.h[k])
      k += 1
    end
  end
  FADHessian{T, n}(tan(x.d), h)
end

function asin{T<:Real, n}(x::FADHessian{T, n})
  h = Array(T, convert(Int, n*(n+1)/2))
  k = 1
  for i in 1:n
    for j in 1:i
      h[k] = ((x.d.v*x.d.g[i]*x.d.g[j]-(x.d.v*x.d.v-1)*h[k])
        /((1-x.d.v*x.d.v)^1.5))
      k += 1
    end
  end
  FADHessian{T, n}(asin(x.d), h)
end

function acos{T<:Real, n}(x::FADHessian{T, n})
  h = Array(T, convert(Int, n*(n+1)/2))
  k = 1
  for i in 1:n
    for j in 1:i
      h[k] = ((-x.d.v*x.d.g[i]*x.d.g[j]+(x.d.v*x.d.v-1)*h[k])
        /((1-x.d.v*x.d.v)^1.5))
      k += 1
    end
  end
  FADHessian{T, n}(acos(x.d), h)
end

function atan{T<:Real, n}(x::FADHessian{T, n})
  h = Array(T, convert(Int, n*(n+1)/2))
  k = 1
  for i in 1:n
    for j in 1:i
      h[k] = ((-2*x.d.v*x.d.g[i]*x.d.g[j]+(x.d.v*x.d.v+1)*h[k])
        /((1+x.d.v*x.d.v)^2))
      k += 1
    end
  end
  FADHessian{T, n}(atan(x.d), h)
end

function sinh{T<:Real, n}(x::FADHessian{T, n})
  h = Array(T, convert(Int, n*(n+1)/2))
  k = 1
  for i in 1:n
    for j in 1:i
      h[k] = (sinh(x.d.v)*x.d.g[i]*x.d.g[j]+cosh(x.d.v)*x.h[k])
      k += 1
    end
  end
  FADHessian{T, n}(sinh(x.d), h)
end

function cosh{T<:Real, n}(x::FADHessian{T, n})
  h = Array(T, convert(Int, n*(n+1)/2))
  k = 1
  for i in 1:n
    for j in 1:i
      h[k] = (cosh(x.d.v)*x.d.g[i]*x.d.g[j]+sinh(x.d.v)*x.h[k])
      k += 1
    end
  end
  FADHessian{T, n}(cosh(x.d), h)
end

function tanh{T<:Real, n}(x::FADHessian{T, n})
  h = Array(T, convert(Int, n*(n+1)/2))
  k = 1
  for i in 1:n
    for j in 1:i
      h[k] = sech(x.d.v)*sech(x.d.v)*(-2*tanh(x.d.v)*x.d.g[i]*x.d.g[j]+x.h[k])
      k += 1
    end
  end
  FADHessian{T, n}(tanh(x.d), h)
end

function asinh{T<:Real, n}(x::FADHessian{T, n})
  h = Array(T, convert(Int, n*(n+1)/2))
  k = 1
  for i in 1:n
    for j in 1:i
      h[k] = ((-x.d.v*x.d.g[i]*x.d.g[j]+(1+x.d.v*x.d.v)*x.h[k])
        /((1+x.d.v*x.d.v)^1.5))
      k += 1
    end
  end
  FADHessian{T, n}(asinh(x.d), h)
end

function acosh{T<:Real, n}(x::FADHessian{T, n})
  h = Array(T, convert(Int, n*(n+1)/2))
  k = 1
  for i in 1:n
    for j in 1:i
      h[k] = ((-x.d.v*x.d.g[i]*x.d.g[j]+(-1+x.d.v*x.d.v)*x.h[k])
        /(((1+x.d.v)^1.5)*((-1+x.d.v)^1.5 )))
      k += 1
    end
  end
  FADHessian{T, n}(acosh(x.d), h)
end

function atanh{T<:Real, n}(x::FADHessian{T, n})
  h = Array(T, convert(Int, n*(n+1)/2))
  k = 1
  for i in 1:n
    for j in 1:i
      h[k] = ((2*x.d.v*x.d.g[i]*x.d.g[j]-(-1+x.d.v*x.d.v)*x.h[k])
        /((-1+x.d.v*x.d.v)^2))
      k += 1
    end
  end
  FADHessian{T, n}(atanh(x.d), h)
end

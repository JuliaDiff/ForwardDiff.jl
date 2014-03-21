immutable FADTensor{T<:Real, n} <: Number
  h::FADHessian{T, n} 
  t::Vector{T}
end

FADTensor{T<:Real, n}(h::FADHessian{T, n}, t::Vector{T}) = FADTensor{T, length(h.d.g)}(h, t)

FADTensor{T<:Real, n}(h::FADHessian{T, n}) = FADTensor{T, length(h.d.g)}(h, convert(Int, n*n*(n+1)/2))

function FADTensor{T<:Real}(v::Vector{T})
  n = length(v)
  Tensor = Array(FADTensor{T, n}, n)
  for i=1:n
    g = zeros(T, n)
    g[i] = one(T)
    Tensor[i] = FADTensor(FADHessian(GraDual{T, n}(v[i], g), zeros(T, convert(Int, n*(n+1)/2))), 
      zeros(T, convert(Int, n*n*(n+1)/2)))
  end
  return Tensor
end

zero{T, n}(::Type{FADTensor{T, n}}) = FADTensor(zero(FADHessian{T, n}), zeros(T, convert(Int, n*n*(n+1)/2)))
one{T, n}(::Type{FADTensor{T, n}}) = FADTensor(one(FADHessian{T, n}), zeros(T, convert(Int, n*n*(n+1)/2)))

value(x::FADTensor) = value(x.h)
value{T<:Real, n}(X::Vector{FADTensor{T, n}}) = [value(x) for x in X]

grad(x::FADTensor) = grad(x.h)
function grad{T<:Real, n}(X::Vector{FADTensor{T, n}})
  m = length(X)
  reshape([x.h.d.g[i] for x in X, i in 1:n], m, n)
end

hessian{T<:Real, n}(x::FADTensor{T, n}) = hessian(x.h)

function tensor{T<:Real, n}(x::FADTensor{T, n})
  y = Array(T, n, n, n)
  k = 1
  
  for a in 1:n
    for i in 1:n
      for j in 1:i
        y[i, j, a] = x.t[k]
        k += 1
      end
    end

    for i in 1:n
      for j in (i+1):n
        y[i, j, a] = y[j, i, a]
      end
    end
  end

  y
end

convert{T<:Real, n}(::Type{FADTensor{T, n}}, x::FADTensor{T, n}) = x
convert{T<:Real, n}(::Type{FADTensor{T, n}}, x::T) =
  FADTensor(FADHessian{T, n}(x, zeros(T, convert(Int, n*(n+1)/2))), zeros(T, convert(Int, n*n*(n+1)/2)))
convert{T<:Real, S<:Real, n}(::Type{FADTensor{T, n}}, x::S) = 
  FADTensor(FADHessian{T, n}(convert(T, x), zeros(T, convert(Int, n*(n+1)/2))), zeros(T, convert(Int, n*n*(n+1)/2)))
convert{T<:Real, S<:Real, n}(::Type{FADTensor{T, n}}, x::FADTensor{S, n}) =
  FADTensor(FADHessian{T, n}(GraDual{T, n}(convert(T, x.h.d.v), convert(Vector{T}, x.h.d.g)),
  convert(Vector{T}, x.h.h)), convert(Vector{T}, x.t))
convert{T<:Real, S<:Real, n}(::Type{T}, x::FADTensor{S, n}) =
  ((x.h.d.g == zeros(S, n) && x.h.h == zeros(S, convert(Int, n*(n+1)/2)) && 
    x.h.t == zeros(S, convert(Int, n*n*(n+1)/2))) ? convert(T, x.h.d.v) : throw(InexactError()))

promote_rule{T<:Real, n}(::Type{FADTensor{T, n}}, ::Type{T}) = FADTensor{T, n}
promote_rule{T<:Real, S<:Real, n}(::Type{FADTensor{T, n}}, ::Type{S}) = FADTensor{promote_type(T, S), n}
promote_rule{T<:Real, S<:Real, n}(::Type{FADTensor{T, n}}, ::Type{FADTensor{S, n}}) = FADTensor{promote_type(T, S), n}

isfadtensor(x::FADTensor) = true
isfadtensor(x::Number) = false

isconstant{T<:Real, n}(x::FADTensor{T, n}) = (isconstant(x.h) && x.t == zeros(T, convert(Int, n*n*(n+1)/2)))
iszero{T<:Real, n}(x::FADTensor{T, n}) = isconstant(x) && (x.h.d.v == zero(T))
isfinite{T<:Real, n}(x::FADTensor{T, n}) = (isfinite(x.h) && x.t == ones(T, convert(Int, n*n*(n+1)/2)))

=={T<:Real, n}(x1::FADTensor{T, n}, x2::FADTensor{T, n}) = ((x1.h == x2.h) && (x1.t == x2.t))
  
show(io::IO, x::FADTensor) =
  print(io, "FADTensor(\nvalue:\n", value(x),
  "\n\ngrad:\n", grad(x),
  "\n\nHessian:\n", hessian(x),
  "\n\nTensor:\n", tensor(x),
  "\n)")

function t2h(i::Int64, j::Int64)
  m, n = i >= j ? (i, j) : (j, i)
  int64(m*(m-1)/2+n)
end

+{T<:Real, n}(x1::FADTensor{T, n}, x2::FADTensor{T, n}) = FADTensor{T, n}(x1.h+x2.h, x1.t+x2.t)

-{T<:Real, n}(x::FADTensor{T, n}) = FADTensor{T,n}(-x.h, -x.t)
-{T<:Real, n}(x1::FADTensor{T, n}, x2::FADTensor{T, n}) = FADTensor{T, n}(x1.h-x2.h, x1.t-x2.t)

function *{T<:Real, n}(x1::FADTensor{T, n}, x2::FADTensor{T, n})
  t = Array(T, convert(Int, n*n*(n+1)/2))
  local l, m, r
  q = 1
  for a in 1:n
    for i in 1:n
      for j in 1:i
        l, m, r = t2h(a, i), t2h(a, j), t2h(i, j)
        t[q] =
         (x2.h.d.g[a]*x1.h.h[r]+x1.h.d.g[a]*x2.h.h[r]
         +x2.h.d.g[i]*x1.h.h[m]+x1.h.d.g[i]*x2.h.h[m]
         +x2.h.d.g[j]*x1.h.h[l]+x1.h.d.g[j]*x2.h.h[l]
         +x2.h.d.v*x1.t[q]+x1.h.d.v*x2.t[q])
        q += 1
      end
    end
  end
  FADTensor{T, n}(x1.h*x2.h, t)
end

*{T<:Real, n}(x1::Bool, x2::FADTensor{T, n}) = convert(T, x1)*x2
*{T<:Real, n}(x1::T, x2::FADTensor{T, n}) = FADTensor{T, n}(x1*x2.d, x1*x2.h)
*{n}(x1::FADTensor{Bool, n}, x2::Bool) = x1*x2
*{T<:Real, n}(x1::FADTensor{T, n}, x2::T) = FADTensor{T, n}(x2*x1.d, x2*x1.h)

function /{T<:Real, n}(x1::FADTensor{T, n}, x2::FADTensor{T, n})
  t = Array(T, convert(Int, n*n*(n+1)/2))
  local l, m, r
  q = 1
  for a in 1:n
    for i in 1:n
      for j in 1:i
        l, m, r = t2h(a, i), t2h(a, j), t2h(i, j)
        t[q] = ((x1.t[q]
          +(-(x1.h.d.g[a]*x2.h.h[r]
            +x1.h.d.g[i]*x2.h.h[m]
            +x1.h.d.g[j]*x2.h.h[l]
            +x2.h.d.g[a]*x1.h.h[r]
            +x2.h.d.g[i]*x1.h.h[m]
            +x2.h.d.g[j]*x1.h.h[l]
            +x1.h.d.v*x2.t[q]
          )+
          2*(x1.h.d.g[a]*x2.h.d.g[i]*x2.h.d.g[j]
            +x2.h.d.g[a]*x1.h.d.g[i]*x2.h.d.g[j]
            +x2.h.d.g[a]*x2.h.d.g[i]*x1.h.d.g[j]
            +x1.h.d.v*(x2.h.d.g[a]*x2.h.h[r]+x2.h.d.g[i]*x2.h.h[m]+x2.h.d.g[j]*x2.h.h[l])
          -3*x1.h.d.v*x2.h.d.g[a]*x2.h.d.g[i]*x2.h.d.g[j]/x2.h.d.v)/x2.h.d.v)/x2.h.d.v
          )/x2.h.d.v)
        q += 1
      end
    end
  end
  FADTensor{T, n}(x1.h/x2.h, t)
end

function sqrt{T<:Real, n}(x::FADTensor{T, n})
  t = Array(T, convert(Int, n*n*(n+1)/2))
  local l, m, r
  q = 1
  for a in 1:n
    for i in 1:n
      for j in 1:i
        l, m, r = t2h(a, i), t2h(a, j), t2h(i, j)
        t[q] = (((0.375*x.h.d.g[a]*x.h.d.g[i]*x.h.d.g[j]/x.h.d.v
          -0.25*(x.h.d.g[a]*x.h.h[r]
           +x.h.d.g[i]*x.h.h[m]
           +x.h.d.g[j]*x.h.h[l]
          ))/x.h.d.v
          +0.5*x.t[q])/sqrt(x.h.d.v)
        )
        q += 1
      end
    end
  end
  FADTensor{T, n}(sqrt(x.h), t)
end

function cbrt{T<:Real, n}(x::FADTensor{T, n})
  t = Array(T, convert(Int, n*n*(n+1)/2))
  local l, m, r
  q = 1
  for a in 1:n
    for i in 1:n
      for j in 1:i
        l, m, r = t2h(a, i), t2h(a, j), t2h(i, j)
        t[q] = (((10*x.h.d.g[a]*x.h.d.g[i]*x.h.d.g[j]/(3*x.h.d.v)
          -2*(x.h.d.g[a]*x.h.h[r]
           +x.h.d.g[i]*x.h.h[m]
           +x.h.d.g[j]*x.h.h[l]
          ))/(3*x.h.d.v)
          +x.t[q])/(3*x.h.d.v^(2/3))
        )
        q += 1
      end
    end
  end
  FADTensor{T, n}(cbrt(x.h), t)
end

^{T1<:Real, T2<:Integer, n}(x::FADTensor{T1, n}, p::T2) = x^convert(Rational{T2}, p)
^{T1<:Real, T2<:Rational, n}(x::FADTensor{T1, n}, p::T2) = x^convert(FloatingPoint, p)

function ^{T1<:Real, T2<:Real, n}(x::FADTensor{T1, n}, p::T2)
  t = Array(T1, convert(Int, n*n*(n+1)/2))
  local l, m, r
  q = 1
  for a in 1:n
    for i in 1:n
      for j in 1:i
        l, m, r = t2h(a, i), t2h(a, j), t2h(i, j)
        t[q] = (p*(
          (p-1)*x.h.d.v^(p-3)*(
          (p-2)*x.h.d.g[a]*x.h.d.g[i]*x.h.d.g[j]
          +x.h.d.v*(x.h.d.g[a]*x.h.h[r]+x.h.d.g[i]*x.h.h[m]+x.h.d.g[j]*x.h.h[l])
          )+
          x.h.d.v^2*x.t[q]))
        q += 1
      end
    end
  end
  FADTensor{T1, n}(x.h^p, t)
end

function exp{T<:Real, n}(x::FADTensor{T, n})
  t = Array(T, convert(Int, n*n*(n+1)/2))
  local l, m, r
  q = 1
  for a in 1:n
    for i in 1:n
      for j in 1:i
        l, m, r = t2h(a, i), t2h(a, j), t2h(i, j)
        t[q] = (exp(x.h.d.v)*(
          x.h.d.g[a]*x.h.d.g[i]*x.h.d.g[j]
         +x.h.d.g[a]*x.h.h[r]
         +x.h.d.g[i]*x.h.h[m]
         +x.h.d.g[j]*x.h.h[l]
         +x.t[q])
        )
        q += 1
      end
    end
  end
  FADTensor{T, n}(exp(x.h), t)
end

function log{T<:Real, n}(x::FADTensor{T, n})
  t = Array(T, convert(Int, n*n*(n+1)/2))
  local l, m, r
  q = 1
  for a in 1:n
    for i in 1:n
      for j in 1:i
        l, m, r = t2h(a, i), t2h(a, j), t2h(i, j)
        t[q] = (((2*x.h.d.g[a]*x.h.d.g[i]*x.h.d.g[j]/x.h.d.v
          -(x.h.d.g[a]*x.h.h[r]
           +x.h.d.g[i]*x.h.h[m]
           +x.h.d.g[j]*x.h.h[l]
          ))/x.h.d.v
          +x.t[q])/x.h.d.v
        )
        q += 1
      end
    end
  end
  FADTensor{T, n}(log(x.h), t)
end

function log2{T<:Real, n}(x::FADTensor{T, n})
  t = Array(T, convert(Int, n*n*(n+1)/2))
  local l, m, r
  q = 1
  for a in 1:n
    for i in 1:n
      for j in 1:i
        l, m, r = t2h(a, i), t2h(a, j), t2h(i, j)
        t[q] = (((2*x.h.d.g[a]*x.h.d.g[i]*x.h.d.g[j]/x.h.d.v
          -(x.h.d.g[a]*x.h.h[r]
           +x.h.d.g[i]*x.h.h[m]
           +x.h.d.g[j]*x.h.h[l]
          ))/x.h.d.v
          +x.t[q])/(x.h.d.v*oftype(T, 0.6931471805599453))
        )
        q += 1
      end
    end
  end
  FADTensor{T, n}(log2(x.h), t)
end

function log10{T<:Real, n}(x::FADTensor{T, n})
  t = Array(T, convert(Int, n*n*(n+1)/2))
  local l, m, r
  q = 1
  for a in 1:n
    for i in 1:n
      for j in 1:i
        l, m, r = t2h(a, i), t2h(a, j), t2h(i, j)
        t[q] = (((2*x.h.d.g[a]*x.h.d.g[i]*x.h.d.g[j]/x.h.d.v
          -(x.h.d.g[a]*x.h.h[r]
           +x.h.d.g[i]*x.h.h[m]
           +x.h.d.g[j]*x.h.h[l]
          ))/x.h.d.v
          +x.t[q])/(x.h.d.v*oftype(T, 2.302585092994046))
        )
        q += 1
      end
    end
  end
  FADTensor{T, n}(log10(x.h), t)
end

function typed_fad_tensor{T<:Real}(f::Function, ::Type{T})
  g(x::Vector{T}) = tensor(f(FADTensor(x)...))
  return g
end

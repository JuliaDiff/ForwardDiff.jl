immutable FADTensor{T<:Real, n} <: Number
  h::FADHessian{T, n} 
  t::Vector{T}
end

FADTensor{T<:Real, n} (h::FADHessian{T, n}, t::Vector{T}) = FADTensor{T, length(h.d.g)}(h, t)

FADTensor{T<:Real, n} (h::FADHessian{T, n}) = FADTensor{T, length(h.d.g)}(h, zeros(T, n^3))

function FADTensor{T<:Real}(v::Vector{T})
  n = length(v)
  Tensor = Array(FADTensor{T, n}, n)
  for i=1:n
    g = zeros(T, n)
    g[i] = one(T)
    Tensor[i] = FADTensor(FADHessian(GraDual{T, n}(v[i], g), zeros(T, convert(Int, n*(n+1)/2))), zeros(T, n^3))
  end
  return Tensor
end

zero{T, n}(::Type{FADTensor{T, n}}) = FADTensor(zero(FADHessian{T, n}), zeros(T, convert(Int, n^3)))
one{T, n}(::Type{FADTensor{T, n}}) = FADTensor(one(FADHessian{T, n}), zeros(T, convert(Int, n^3)))

value(x::FADTensor) = value(x.h)
value{T<:Real, n}(X::Vector{FADTensor{T, n}}) = [x.h.d.v for x in X]

grad(x::FADTensor) = grad(x.h)
function grad{T<:Real, n}(X::Vector{FADTensor{T, n}})
  m = length(X)
  reshape([x.h.d.g[i] for x in X, i in 1:n], m, n)
end

hessian{T<:Real, n}(x::FADTensor{T, n}) = hessian(x.h)

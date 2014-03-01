immutable FADTensor{T<:Real, n} <: Number
    h::FADHessian{T, n} 
    t::Vector{T}
end

FADTensor{T<:Real, n} (h::FADHessian{T, n}, t::Vector{T}) =
  FADTensor{T, length(h.d.g)}(h, t)

FADTensor{T<:Real, n} (h::FADHessian{T, n}) =
  FADTensor{T, length(h.d.g)}(h, zeros(T, n^3))

function FADTensor{T<:Real}(v::Vector{T})
  n = length(v)
  Tensor = Array(FADTensor{T, n}, n)
  for i=1:n
    g = zeros(T, n)
    g[i] = one(T)
    Tensor[i] = FADTensor(
      FADHessian(GraDual{T, n}(v[i], g), zeros(T, convert(Int, n*(n+1)/2))),
      zeros(T, n^3))
  end
  return Tensor
end

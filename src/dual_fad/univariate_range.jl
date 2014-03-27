function dual_fad{T <: Real}(f::Function, x::Vector{T}, gradient_output, dualvec)
  for i in 1:length(x)
    dualvec[i] = Dual(x[i], zero(T))
  end
  for i in 1:length(x)
    dualvec[i] = Dual(real(dualvec[i]), one(T))
    gradient_output[i] = epsilon(f(dualvec))
    dualvec[i] = Dual(real(dualvec[i]), zero(T))
  end
end

function dual_fad_gradient!{T <: Real}(f::Function, ::Type{T}; n::Int=1)
  dualvec = Array(Dual{T}, n)
  g!(x::Vector{T}, gradient_output::Vector{T}) = dual_fad(f, x, gradient_output, dualvec)
  return g!
end

function dual_fad_gradient{T <: Real}(f::Function, ::Type{T}; n::Int=1)
  dualvec = Array(Dual{T}, n)
  gradient_output = Array(T, n)
  function g(x::Vector{T})
    dual_fad(f, x, gradient_output, dualvec)
    gradient_output
  end
  return g
end

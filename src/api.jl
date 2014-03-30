function forwarddiff_gradient!{T<:Real}(f::Function, ::Type{T}; fadtype::Symbol=:dual, args...)
  if fadtype == :dual
    dual_fad_gradient!(f, T; args...)
  elseif fadtype == :typed
    typed_fad_gradient!(f, T)
  else
    error("forwarddiff_gradient not supported for $fadtype FAD")
  end
end

function forwarddiff_gradient{T<:Real}(f::Function, ::Type{T}; fadtype::Symbol=:dual, args...)
  if fadtype == :dual
    dual_fad_gradient(f, T; args...)
  elseif fadtype == :typed
    typed_fad_gradient(f, T)
  else
    error("forwarddiff_gradient not supported for $fadtype FAD")
  end
end

function forwarddiff_jacobian!{T<:Real}(f::Function, ::Type{T}; fadtype::Symbol=:dual, args...)
  if fadtype == :dual
    dual_fad_jacobian!(f, T; args...)
  else
    error("forwarddiff_jacobian not supported for $fadtype FAD")
  end
end

function forwarddiff_jacobian{T<:Real}(f::Function, ::Type{T}; fadtype::Symbol=:dual, args...)
  if fadtype == :dual
    dual_fad_jacobian(f, T; args...)
  elseif fadtype == :typed
    typed_fad_jacobian(f, T)
  else
    error("forwarddiff_jacobian not supported for $fadtype FAD")
  end
end

function forwarddiff_hessian{T<:Real}(f::Function, ::Type{T}; fadtype::Symbol=:typed, args...)
  if fadtype == :typed
    typed_fad_hessian(f, T)
  else
    error("forwarddiff_hessian not supported for $fadtype FAD")
  end
end

function forwarddiff_tensor{T<:Real}(f::Function, ::Type{T}; fadtype::Symbol=:typed, args...)
  if fadtype == :typed
    typed_fad_tensor(f, T)
  else
    error("forwarddiff_tensor not supported for $fadtype FAD")
  end
end

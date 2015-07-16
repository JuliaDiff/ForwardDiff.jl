module ForwardDiff
  using DualNumbers
  
  importall Base

  include("GraDual.jl")
  include("FADHessian.jl")
  include("FADTensor.jl")
  include("api.jl")

  export
    # API
    forwarddiff_gradient!,
    forwarddiff_gradient,
    forwarddiff_jacobian!,
    forwarddiff_jacobian,
    forwarddiff_hessian!,
    forwarddiff_hessian,
    forwarddiff_tensor!,
    forwarddiff_tensor,    
    # exports for typespecific_fad    
    GraDual,
    value,
    grad,
    jacobian,
    isgradual,
    isconstant,
    iszero,
    FADHessian,
    hessian,
    isfadhessian,
    FADTensor,
    tensor,
    isfadtensor

end # module ForwardDiff

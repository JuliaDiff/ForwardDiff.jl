module ForwardDiff
  using DualNumbers
  
  importall Base
  
  export
    # exports for dual_fad
    forwarddiff_gradient,
    forwarddiff_jacobian,
    # exports for typespecific_fad    
    GraDual,
    gradual,
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

  include(joinpath("dual_fad", "univariate_range.jl"))
  include(joinpath("dual_fad", "multivariate_range.jl"))
  include(joinpath("typed_fad", "GraDual.jl"))
  include(joinpath("typed_fad", "FADHessian.jl"))
  include(joinpath("typed_fad", "FADTensor.jl"))
end # module ForwardDiff

module ForwardDiff
  using DualNumbers
  
  # importall Base
  
  export
    # exports for dual_fad
    autodiff,
    # exports for typespecific_fad
    Dual,
    Dual128,
    Dual64,
    DualPair,
    dual,
    dual128,
    dual64,
    isdual,
    dual_show,
    
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
    isfadhessian

  include(joinpath("dual_fad", "univariate_range.jl"))
  include(joinpath("dual_fad", "multivariate_range.jl"))
  include(joinpath("typespecific_fad", "GraDual.jl"))
  include(joinpath("typespecific_fad", "FADHessian.jl"))
end # module ForwardDiff

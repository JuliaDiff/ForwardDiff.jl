module AutoDiff
  using DualNumbers
  
  importall Base
  
  include("gradual.jl")
  include("source_transformation.jl")
  
  export
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
    iszero
end # module Autodiff

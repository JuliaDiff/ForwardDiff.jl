module AutoDiff
  using DualNumbers
  
  importall Base
  
  include("ad_jonas_rauch.jl")
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
    dual_show
end # module Autodiff

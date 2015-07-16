module ForwardDiff
  using NDuals
  
  importall Base

  include("FADNumber.jl")
  include("FADHessian.jl")
  include("FADTensor.jl")

  export NDualTup,
         NDualVec,
         gradient!,
         gradient,
         jacobian!,
         jacobian,
         hessian!,
         hessian,
         tensor!,
         tensor

end # module ForwardDiff

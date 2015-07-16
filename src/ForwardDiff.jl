module ForwardDiff
    using NDuals

    importall Base

    include("FADNumber.jl")
    include("FADHessian.jl")
    include("FADTensor.jl")
    include("autodiff_funcs.jl")

    export NDualTup,
           NDualVec,
           FADNumber,
           FADHessian,
           FADTensor,
           isconstant,
           neps,
           value,
           gradient!,
           gradient,
           gradient_func,
           jacobian!,
           jacobian,
           hessian!,
           hessian,
           hessian_func,
           tensor!,
           tensor,
           tensor_func

end # module ForwardDiff

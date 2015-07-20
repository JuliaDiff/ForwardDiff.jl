module ForwardDiff

    importall Base

    import NaNMath
    import Calculus

    if VERSION < v"0.4-"
        warn("ForwardDiff.jl is only officially compatible with Julia v0.4-. You're currently running Julia $VERSION.")
    end

    include("FADNumber.jl")
    include("grad/FADGradient.jl")
    include("FADHessian.jl")
    include("FADTensor.jl")
    include("autodiff_funcs.jl")

    export GradVec,
           GradTup,
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

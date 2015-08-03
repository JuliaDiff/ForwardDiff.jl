module ForwardDiff

    importall Base
    using Calculus
    using NaNMath

    const fad_supported_univar_funcs = symbolic_derivatives_1arg()

    if VERSION < v"0.4-"
        warn("ForwardDiff.jl is only officially compatible with Julia v0.4-. You're currently running Julia $VERSION.")
    end

    abstract Dim{N}

    include("ForwardDiffNum.jl")
    include("GradientNum.jl")
    include("HessianNum.jl")
    include("TensorNum.jl")
    include("fad_api.jl")
    include("deprecated.jl")

    export fad_derivative!,
           fad_derivative,
           fad_gradient!,
           fad_gradient,
           fad_jacobian!,
           fad_jacobian,
           fad_hessian!,
           fad_hessian,
           fad_tensor!,
           fad_tensor

end # module ForwardDiff

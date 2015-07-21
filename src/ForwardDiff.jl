module ForwardDiff

    importall Base

    import NaNMath
    import Calculus

    if VERSION < v"0.4-"
        warn("ForwardDiff.jl is only officially compatible with Julia v0.4-. You're currently running Julia $VERSION.")
    end

    include("ForwardDiffNum.jl")
    include("GradientNum.jl")
    include("HessianNum.jl")
    include("TensorNum.jl")
    include("fad_api.jl")

    typealias PartialsTuple{N,T} GradNumTup{N,T}
    typealias PartialsVector{N,T} GradNumVec{N,T}

    export PartialsTuple,
           PartialsVector,
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

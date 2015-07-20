module ForwardDiff

    importall Base

    import NaNMath
    import Calculus

    if VERSION < v"0.4-"
        warn("ForwardDiff.jl is only officially compatible with Julia v0.4-. You're currently running Julia $VERSION.")
    end

    switch_eltype{T,S}(::Type{Vector{T}}, ::Type{S}) = Vector{S}
    switch_eltype{N,T,S}(::Type{NTuple{N,T}}, ::Type{S}) = NTuple{N,S}

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

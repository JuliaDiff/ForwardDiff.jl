module ForwardDiff

    importall Base
    importall Calculus

    import NaNMath

    if VERSION < v"0.4-"
        warn("ForwardDiff.jl is only officially compatible with Julia v0.4-. You're currently running Julia $VERSION.")
    end

    abstract Partials{N}
    
    npartials{N}(::Type{Partials{N}}) = N

    include("ForwardDiffNum.jl")
    include("GradientNum.jl")
    include("HessianNum.jl")
    include("TensorNum.jl")
    include("fad_api.jl")

    @generated function pick_implementation{N,T}(::Type{Partials{N}}, ::Type{T})
        if N > 10
            return :(Vector{$T})
        else
            return :(NTuple{$N,$T})
        end
    end

    export Partials,
           derivative!,
           derivative,
           derivative_func,
           gradient!,
           gradient,
           gradient_func,
           jacobian!,
           jacobian,
           jacobian_func,
           hessian!,
           hessian,
           hessian_func,
           tensor!,
           tensor,
           tensor_func

end # module ForwardDiff

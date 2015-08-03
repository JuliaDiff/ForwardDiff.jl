module ForwardDiff

    import Calculus
    import NaNMath
    import Base: *, /, +, -, ^,
                 hash, ==, isequal, copy,
                 zero, one, convert, promote_rule,
                 read, write, isless, isreal,
                 isnan, isfinite, eps, conj,
                 transpose, ctranspose, eltype,
                 abs, abs2

    const fad_supported_univar_funcs = Calculus.symbolic_derivatives_1arg()

    for (fsym,expr) in fad_supported_univar_funcs
        @eval import Base.$(fsym)
    end

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

    export derivative!,
           derivative,
           gradient!,
           gradient,
           jacobian!,
           jacobian,
           hessian!,
           hessian,
           tensor!,
           tensor

end # module ForwardDiff

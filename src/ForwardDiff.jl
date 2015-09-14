module ForwardDiff
    
    if VERSION < v"0.4-"
        warn("ForwardDiff.jl is only officially compatible with Julia v0.4-. You're currently running Julia $VERSION.")
    end

    import Calculus
    import NaNMath
    import Base: *, /, +, -, ^, getindex, length,
                 hash, ==, isequal, copy, zero, 
                 one, rand, convert, promote_rule, 
                 read, write, isless, isreal, isnan, 
                 isfinite, eps, conj, transpose, 
                 ctranspose, eltype, abs, abs2, start, 
                 next, done, atan2

    const supported_unary_funcs = map(first, Calculus.symbolic_derivatives_1arg())

    for fsym in supported_unary_funcs
        @eval import Base.$(fsym);
    end

    include("Partials.jl")
    include("ForwardDiffNumber.jl")
    include("GradientNumber.jl")
    include("HessianNumber.jl")
    include("TensorNumber.jl")
    include("api/api.jl")

    export AllResults,
           ForwardDiffCache,
           value,
           value!,
           derivative!,
           derivative,
           gradient!,
           jacobian!,
           jacobian,
           hessian!,
           hessian,
           tensor!,
           tensor

end # module ForwardDiff

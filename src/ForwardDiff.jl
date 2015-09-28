module ForwardDiff

    if VERSION < v"0.4-"
        warn("ForwardDiff.jl is only officially compatible with Julia v0.4-. You're currently running Julia $VERSION.")
    end

    ###########
    # imports #
    ###########
    import Calculus
    import NaNMath
    import Base: *, /, +, -, ^, getindex, length,
                 hash, ==, isequal, copy, zero,
                 one, float, rand, convert, promote_rule,
                 read, write, isless, isreal, isnan,
                 isfinite, isinf, eps, conj, transpose,
                 ctranspose, eltype, abs, abs2, start,
                 next, done, atan2

    const auto_defined_unary_funcs = map(first, Calculus.symbolic_derivatives_1arg())

    for fsym in auto_defined_unary_funcs
        @eval import Base.$(fsym);
    end

    ############
    # includes #
    ############
    include("Partials.jl")
    include("ForwardDiffNumber.jl")
    include("GradientNumber.jl")
    include("HessianNumber.jl")
    include("TensorNumber.jl")
    include("api/api.jl")

    ###########################
    # Misc. Utility Functions #
    ###########################
    @inline switch_eltype{T,S}(::Type{Vector{T}}, ::Type{S}) = Vector{S}
    @inline switch_eltype{N,T,S}(::Type{NTuple{N,T}}, ::Type{S}) = NTuple{N,S}

    for F in (:GradientNumber, :HessianNumber, :TensorNumber)
        @eval begin
            @inline switch_eltype{N,T,S}(::Type{$F{N,T,NTuple{N,T}}}, ::Type{S}) = $F{N,S,NTuple{N,S}}
            @inline switch_eltype{N,T,S}(::Type{$F{N,T,Vector{T}}}, ::Type{S}) = $F{N,S,Vector{S}}
        end
    end

    @inline function promote_eltype{F<:ForwardDiffNumber}(::Type{F}, types::DataType...)
        return switch_eltype(F, promote_type(eltype(F), types...))
    end

    @inline promote_typeof(n1::ForwardDiffNumber, n2::ForwardDiffNumber) = promote_type(typeof(n1), typeof(n2))

    @inline promote_eltypesof(n1::ForwardDiffNumber, n2::ForwardDiffNumber, a) = promote_eltype(promote_typeof(n1, n2), typeof(a))

    @inline promote_eltypeof(n::ForwardDiffNumber, a) = promote_eltype(typeof(n), typeof(a))
    @inline promote_eltypeof(n::ForwardDiffNumber, a, b) = promote_eltype(typeof(n), typeof(a), typeof(b))

    ###########
    # exports #
    ###########
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

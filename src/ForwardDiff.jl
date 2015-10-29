isdefined(Base, :__precompile__) && __precompile__()

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

    macro defambiguous(ex)
        message = """Sorry! This method should never have been called: $ex
                     It was defined to resolve ambiguity, and should always
                     fallback to a more specific method defined elsewhere.
                     Please report this bug to ForwardDiff.jl's issue tracker.
                  """
        return esc(quote
            $ex = error($message)
        end)
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

    #######################
    # Promotion Utilities #
    #######################
    @inline switch_eltype{T,S}(::Type{Vector{T}}, ::Type{S}) = Vector{S}
    @inline switch_eltype{N,T,S}(::Type{NTuple{N,T}}, ::Type{S}) = NTuple{N,S}

    for F in (:TensorNumber, :GradientNumber, :HessianNumber)
        @eval begin
            @inline switch_eltype{N,T,S}(::Type{$F{N,T,NTuple{N,T}}}, ::Type{S}) = $F{N,S,NTuple{N,S}}
            @inline switch_eltype{N,T,S}(::Type{$F{N,T,Vector{T}}}, ::Type{S}) = $F{N,S,Vector{S}}

            function promote_rule{N,T1,C1,T2,C2}(::Type{($F){N,T1,C1}},
                                                 ::Type{($F){N,T2,C2}})
                P = promote_type(Partials{T1,C1}, Partials{T2,C2})
                T, C = eltype(P), containtype(P)
                return ($F){N,T,C}
            end

            function promote_rule{N,T,C,S<:Real}(::Type{($F){N,T,C}}, ::Type{S})
                R = promote_type(T, S)
                return ($F){N,R,switch_eltype(C, R)}
            end
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

module ForwardDiffStaticArraysExt

using ForwardDiff, StaticArrays
using ForwardDiff.LinearAlgebra
using ForwardDiff.DiffResults
using ForwardDiff: Dual, partials, GradientConfig, JacobianConfig, HessianConfig, Tag, Chunk,
                   gradient, hessian, jacobian, gradient!, hessian!, jacobian!,
                   extract_gradient!, extract_jacobian!, extract_value!,
                   vector_mode_gradient, vector_mode_gradient!,
                   vector_mode_jacobian, vector_mode_jacobian!, valtype, value, _lyap_div!
using DiffResults: DiffResult, ImmutableDiffResult, MutableDiffResult

@generated function dualize(::Type{T}, x::StaticArray) where T
    N = length(x)
    dx = Expr(:tuple, [:(Dual{T}(x[$i], chunk, Val{$i}())) for i in 1:N]...)
    V = StaticArrays.similar_type(x, Dual{T,eltype(x),N})
    return quote
        chunk = Chunk{$N}()
        $(Expr(:meta, :inline))
        return $V($(dx))
    end
end

@inline static_dual_eval(::Type{T}, f, x::StaticArray) where T = f(dualize(T, x))

function LinearAlgebra.eigvals(A::Symmetric{<:Dual{Tg,T,N}, <:StaticArrays.StaticMatrix}) where {Tg,T<:Real,N}
    λ,Q = eigen(Symmetric(value.(parent(A))))
    parts = ntuple(j -> diag(Q' * getindex.(partials.(A), j) * Q), N)
    Dual{Tg}.(λ, tuple.(parts...))
end

function LinearAlgebra.eigen(A::Symmetric{<:Dual{Tg,T,N}, <:StaticArrays.StaticMatrix}) where {Tg,T<:Real,N}
    λ = eigvals(A)
    _,Q = eigen(Symmetric(value.(parent(A))))
    parts = ntuple(j -> Q*ForwardDiff._lyap_div!(Q' * getindex.(partials.(A), j) * Q - Diagonal(getindex.(partials.(λ), j)), value.(λ)), N)
    Eigen(λ,Dual{Tg}.(Q, tuple.(parts...)))
end

# Gradient
@inline ForwardDiff.gradient(f, x::StaticArray)                      = vector_mode_gradient(f, x)
@inline ForwardDiff.gradient(f, x::StaticArray, cfg::GradientConfig) = gradient(f, x)
@inline ForwardDiff.gradient(f, x::StaticArray, cfg::GradientConfig, ::Val) = gradient(f, x)

@inline ForwardDiff.gradient!(result::Union{AbstractArray,DiffResult}, f, x::StaticArray) = vector_mode_gradient!(result, f, x)
@inline ForwardDiff.gradient!(result::Union{AbstractArray,DiffResult}, f, x::StaticArray, cfg::GradientConfig) = gradient!(result, f, x)
@inline ForwardDiff.gradient!(result::Union{AbstractArray,DiffResult}, f, x::StaticArray, cfg::GradientConfig, ::Val) = gradient!(result, f, x)

@generated function extract_gradient(::Type{T}, y::Real, x::S) where {T,S<:StaticArray}
    result = Expr(:tuple, [:(partials(T, y, $i)) for i in 1:length(x)]...)
    return quote
        $(Expr(:meta, :inline))
        V = StaticArrays.similar_type(S, valtype($y))
        return V($result)
    end
end

@inline function ForwardDiff.vector_mode_gradient(f, x::StaticArray)
    T = typeof(Tag(f, eltype(x)))
    return extract_gradient(T, static_dual_eval(T, f, x), x)
end

@inline function ForwardDiff.vector_mode_gradient!(result, f, x::StaticArray)
    T = typeof(Tag(f, eltype(x)))
    return extract_gradient!(T, result, static_dual_eval(T, f, x))
end

# Jacobian
@inline ForwardDiff.jacobian(f, x::StaticArray) = vector_mode_jacobian(f, x)
@inline ForwardDiff.jacobian(f, x::StaticArray, cfg::JacobianConfig) = jacobian(f, x)
@inline ForwardDiff.jacobian(f, x::StaticArray, cfg::JacobianConfig, ::Val) = jacobian(f, x)

@inline ForwardDiff.jacobian!(result::Union{AbstractArray,DiffResult}, f, x::StaticArray) = vector_mode_jacobian!(result, f, x)
@inline ForwardDiff.jacobian!(result::Union{AbstractArray,DiffResult}, f, x::StaticArray, cfg::JacobianConfig) = jacobian!(result, f, x)
@inline ForwardDiff.jacobian!(result::Union{AbstractArray,DiffResult}, f, x::StaticArray, cfg::JacobianConfig, ::Val) = jacobian!(result, f, x)

@generated function extract_jacobian(::Type{T}, ydual::StaticArray, x::S) where {T,S<:StaticArray}
    M, N = length(ydual), length(x)
    result = Expr(:tuple, [:(partials(T, ydual[$i], $j)) for i in 1:M, j in 1:N]...)
    return quote
        $(Expr(:meta, :inline))
        V = StaticArrays.similar_type(S, valtype(eltype($ydual)), Size($M, $N))
        return V($result)
    end
end

function extract_jacobian(::Type{T}, ydual::AbstractArray, x::StaticArray) where T
    result = similar(ydual, valtype(eltype(ydual)), length(ydual), length(x))
    return extract_jacobian!(T, result, ydual, length(x))
end

@inline function ForwardDiff.vector_mode_jacobian(f, x::StaticArray)
    T = typeof(Tag(f, eltype(x)))
    return extract_jacobian(T, static_dual_eval(T, f, x), x)
end

@inline function ForwardDiff.vector_mode_jacobian!(result, f, x::StaticArray)
    T = typeof(Tag(f, eltype(x)))
    ydual = static_dual_eval(T, f, x)
    result = extract_jacobian!(T, result, ydual, length(x))
    result = extract_value!(T, result, ydual)
    return result
end

@inline function ForwardDiff.vector_mode_jacobian!(result::ImmutableDiffResult, f, x::StaticArray)
    T = typeof(Tag(f, eltype(x)))
    ydual = static_dual_eval(T, f, x)
    result = DiffResults.jacobian!(result, extract_jacobian(T, ydual, x))
    result = DiffResults.value!(d -> value(T,d), result, ydual)
    return result
end

# Hessian
ForwardDiff.hessian(f, x::StaticArray) = jacobian(y -> gradient(f, y), x)
ForwardDiff.hessian(f, x::StaticArray, cfg::HessianConfig) = hessian(f, x)
ForwardDiff.hessian(f, x::StaticArray, cfg::HessianConfig, ::Val) = hessian(f, x)

ForwardDiff.hessian!(result::AbstractArray, f, x::StaticArray) = jacobian!(result, y -> gradient(f, y), x)

ForwardDiff.hessian!(result::MutableDiffResult, f, x::StaticArray) = hessian!(result, f, x, HessianConfig(f, result, x))

ForwardDiff.hessian!(result::ImmutableDiffResult, f, x::StaticArray, cfg::HessianConfig) = hessian!(result, f, x)
ForwardDiff.hessian!(result::ImmutableDiffResult, f, x::StaticArray, cfg::HessianConfig, ::Val) = hessian!(result, f, x)

function ForwardDiff.hessian!(result::ImmutableDiffResult, f, x::StaticArray)
    T = typeof(Tag(f, eltype(x)))
    d1 = dualize(T, x)
    d2 = dualize(T, d1)
    fd2 = f(d2)
    val = value(T,value(T,fd2))
    grad = extract_gradient(T,value(T,fd2), x)
    hess = extract_jacobian(T,partials(T,fd2), x)
    result = DiffResults.hessian!(result, hess)
    result = DiffResults.gradient!(result, grad)
    result = DiffResults.value!(result, val)
    return result
end

end
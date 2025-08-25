module ForwardDiffStaticArraysExt

using ForwardDiff, StaticArrays
using ForwardDiff.LinearAlgebra
using ForwardDiff.DiffResults
using ForwardDiff: Dual, partials, GradientConfig, JacobianConfig, HessianConfig, Tag, Chunk,
                   gradient, hessian, jacobian, gradient!, hessian!, jacobian!,
                   extract_gradient!, extract_jacobian!, extract_value!,
                   vector_mode_gradient, vector_mode_gradient!,
                   vector_mode_jacobian, vector_mode_jacobian!, valtype, value
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

@inline static_dual_eval(::Type{T}, f::F, x::StaticArray) where {T,F} = f(dualize(T, x))

# To fix method ambiguity issues:
function LinearAlgebra.eigvals(A::Symmetric{<:Dual{Tg,T,N}, <:StaticArrays.StaticMatrix}) where {Tg,T<:Real,N}
    return ForwardDiff._eigvals(A)
end
function LinearAlgebra.eigen(A::Symmetric{<:Dual{Tg,T,N}, <:StaticArrays.StaticMatrix}) where {Tg,T<:Real,N}
    return ForwardDiff._eigen(A)
end

# For `MMatrix` we can use the in-place method
ForwardDiff._lyap_div!!(A::StaticArrays.MMatrix, λ::AbstractVector) = ForwardDiff._lyap_div!(A, λ)

# Gradient
@inline ForwardDiff.gradient(f::F, x::StaticArray) where {F} = vector_mode_gradient(f, x)
@inline ForwardDiff.gradient(f::F, x::StaticArray, cfg::GradientConfig) where {F} = gradient(f, x)
@inline ForwardDiff.gradient(f::F, x::StaticArray, cfg::GradientConfig, ::Val) where {F} = gradient(f, x)

@inline ForwardDiff.gradient!(result::Union{AbstractArray,DiffResult}, f::F, x::StaticArray) where {F} = vector_mode_gradient!(result, f, x)
@inline ForwardDiff.gradient!(result::Union{AbstractArray,DiffResult}, f::F, x::StaticArray, cfg::GradientConfig) where {F} = gradient!(result, f, x)
@inline ForwardDiff.gradient!(result::Union{AbstractArray,DiffResult}, f::F, x::StaticArray, cfg::GradientConfig, ::Val) where {F} = gradient!(result, f, x)

@generated function extract_gradient(::Type{T}, y::Real, x::S) where {T,S<:StaticArray}
    result = Expr(:tuple, [:(partials(T, y, $i)) for i in 1:length(x)]...)
    return quote
        $(Expr(:meta, :inline))
        V = StaticArrays.similar_type(S, valtype($y))
        return V($result)
    end
end

@inline function ForwardDiff.vector_mode_gradient(f::F, x::StaticArray) where {F}
    T = typeof(Tag(f, eltype(x)))
    return extract_gradient(T, static_dual_eval(T, f, x), x)
end

@inline function ForwardDiff.vector_mode_gradient!(result, f::F, x::StaticArray) where {F}
    T = typeof(Tag(f, eltype(x)))
    return extract_gradient!(T, result, static_dual_eval(T, f, x))
end

# Jacobian
@inline ForwardDiff.jacobian(f::F, x::StaticArray) where {F} = vector_mode_jacobian(f, x)
@inline ForwardDiff.jacobian(f::F, x::StaticArray, cfg::JacobianConfig) where {F} = jacobian(f, x)
@inline ForwardDiff.jacobian(f::F, x::StaticArray, cfg::JacobianConfig, ::Val) where {F} = jacobian(f, x)

@inline ForwardDiff.jacobian!(result::Union{AbstractArray,DiffResult}, f::F, x::StaticArray) where {F} = vector_mode_jacobian!(result, f, x)
@inline ForwardDiff.jacobian!(result::Union{AbstractArray,DiffResult}, f::F, x::StaticArray, cfg::JacobianConfig) where {F} = jacobian!(result, f, x)
@inline ForwardDiff.jacobian!(result::Union{AbstractArray,DiffResult}, f::F, x::StaticArray, cfg::JacobianConfig, ::Val) where {F} = jacobian!(result, f, x)

@generated function extract_jacobian(::Type{T}, ydual::StaticArray, x::S) where {T,S<:StaticArray}
    M, N = length(ydual), length(x)
    result = Expr(:tuple, [:(partials(T, ydual[$i], $j)) for i in 1:M, j in 1:N]...)
    return quote
        $(Expr(:meta, :inline))
        V = StaticArrays.similar_type(S, valtype(eltype($ydual)), Size($M, $N))
        return V($result)
    end
end

@inline function ForwardDiff.vector_mode_jacobian(f::F, x::StaticArray) where {F}
    T = typeof(Tag(f, eltype(x)))
    return extract_jacobian(T, static_dual_eval(T, f, x), x)
end

function extract_jacobian(::Type{T}, ydual::AbstractArray, x::StaticArray) where T
    result = similar(ydual, valtype(eltype(ydual)), length(ydual), length(x))
    return extract_jacobian!(T, result, ydual, length(x))
end

@inline function ForwardDiff.vector_mode_jacobian!(result, f::F, x::StaticArray) where {F}
    T = typeof(Tag(f, eltype(x)))
    ydual = static_dual_eval(T, f, x)
    result = extract_jacobian!(T, result, ydual, length(x))
    result = extract_value!(T, result, ydual)
    return result
end

@inline function ForwardDiff.vector_mode_jacobian!(result::ImmutableDiffResult, f::F, x::StaticArray) where {F}
    T = typeof(Tag(f, eltype(x)))
    ydual = static_dual_eval(T, f, x)
    result = DiffResults.jacobian!(result, extract_jacobian(T, ydual, x))
    result = DiffResults.value!(Base.Fix1(value, T), result, ydual)
    return result
end

# Hessian
ForwardDiff.hessian(f::F, x::StaticArray) where {F} = jacobian(Base.Fix1(gradient, f), x)
ForwardDiff.hessian(f::F, x::StaticArray, cfg::HessianConfig) where {F} = hessian(f, x)
ForwardDiff.hessian(f::F, x::StaticArray, cfg::HessianConfig, ::Val) where {F} = hessian(f, x)

ForwardDiff.hessian!(result::AbstractArray, f::F, x::StaticArray) where {F} = jacobian!(result, Base.Fix1(gradient, f), x)

ForwardDiff.hessian!(result::AbstractArray, f::F, grad::AbstractArray, x::StaticArray) where {F} = hessian!(result, grad, f, x, HessianConfig(f, grad, x))

ForwardDiff.hessian!(result::MutableDiffResult, f::F, x::StaticArray) where {F} = hessian!(result, f, x, HessianConfig(f, result, x))

ForwardDiff.hessian!(result::ImmutableDiffResult, f::F, x::StaticArray, cfg::HessianConfig) where {F} = hessian!(result, f, x)
ForwardDiff.hessian!(result::ImmutableDiffResult, f::F, x::StaticArray, cfg::HessianConfig, ::Val) where {F} = hessian!(result, f, x)

function ForwardDiff.hessian!(result::ImmutableDiffResult, f::F, x::StaticArray) where {F}
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

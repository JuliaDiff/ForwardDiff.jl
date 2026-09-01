module ForwardDiffStaticArraysExt

using ForwardDiff, StaticArrays
using ForwardDiff.LinearAlgebra
using ForwardDiff.DiffResults
using ForwardDiff: Dual, partials, npartials, Partials, GradientConfig, JacobianConfig, HessianConfig, Tag, Chunk,
                   gradient, hessian, jacobian, gradient!, hessian!, jacobian!,
                   extract_gradient!, extract_jacobian!, extract_value!, structural_linearindices,
                   vector_mode_gradient, vector_mode_gradient!, outer_tag,
                   vector_mode_jacobian, vector_mode_jacobian!, HESSIAN_ERROR, valtype, value
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
        V = StaticArrays.similar_type(S, valtype(T, $y))
        return V($result)
    end
end

@inline function ForwardDiff.vector_mode_gradient(f::F, x::StaticArray) where {F}
    T = typeof(Tag(f, eltype(x)))
    return extract_gradient(T, f(dualize(T, x)), x)
end

@inline function ForwardDiff.vector_mode_gradient!(result, f::F, x::StaticArray) where {F}
    T = typeof(Tag(f, eltype(x)))
    return extract_gradient!(T, result, f(dualize(T, x)))
end

# Jacobian
@inline ForwardDiff.jacobian(f::F, x::StaticArray) where {F} = vector_mode_jacobian(f, x)
@inline ForwardDiff.jacobian(f::F, x::StaticArray, cfg::JacobianConfig) where {F} = jacobian(f, x)
@inline ForwardDiff.jacobian(f::F, x::StaticArray, cfg::JacobianConfig, ::Val) where {F} = jacobian(f, x)

@inline ForwardDiff.jacobian!(result::Union{AbstractArray,DiffResult}, f::F, x::StaticArray) where {F} = vector_mode_jacobian!(result, f, x)
@inline ForwardDiff.jacobian!(result::Union{AbstractArray,DiffResult}, f::F, x::StaticArray, cfg::JacobianConfig) where {F} = jacobian!(result, f, x)
@inline ForwardDiff.jacobian!(result::Union{AbstractArray,DiffResult}, f::F, x::StaticArray, cfg::JacobianConfig, ::Val) where {F} = jacobian!(result, f, x)

@generated function extract_jacobian(::Type{T}, ydual::Union{StaticArray,Partials}, x::S) where {T,S<:StaticArray}
    M = ydual <: Partials ? npartials(ydual) : length(ydual)
    N = length(x)
    result = Expr(:tuple, [:(partials(T, ydual[$i], $j)) for i in 1:M, j in 1:N]...)
    return quote
        $(Expr(:meta, :inline))
        V = StaticArrays.similar_type(S, valtype(T, eltype($ydual)), Size($M, $N))
        return V($result)
    end
end

@inline function ForwardDiff.vector_mode_jacobian(f::F, x::StaticArray) where {F}
    T = typeof(Tag(f, eltype(x)))
    return extract_jacobian(T, f(dualize(T, x)), x)
end

function extract_jacobian(::Type{T}, ydual::AbstractArray, x::StaticArray) where T
    result = similar(ydual, valtype(T, eltype(ydual)), length(ydual), length(x))
    return extract_jacobian!(T, result, ydual, length(x))
end

@inline function ForwardDiff.vector_mode_jacobian!(result, f::F, x::StaticArray) where {F}
    T = typeof(Tag(f, eltype(x)))
    ydual = f(dualize(T, x))
    result = extract_jacobian!(T, result, ydual, length(x))
    result = extract_value!(T, result, ydual)
    return result
end

@inline function ForwardDiff.vector_mode_jacobian!(result::ImmutableDiffResult, f::F, x::StaticArray) where {F}
    T = typeof(Tag(f, eltype(x)))
    ydual = f(dualize(T, x))
    result = DiffResults.jacobian!(result, extract_jacobian(T, ydual, x))
    result = DiffResults.value!(Base.Fix1(value, T), result, ydual)
    return result
end

# Hessian
@inline function extract_hessian(::Type{T}, ::Type{TO}, ydual::Dual{TO,<:Dual{T}}, x::StaticArray) where {T,TO}
    H = extract_jacobian(T, partials(TO, ydual), x)
    return typeof(H)(Symmetric(H, :U))
end

# A result that never picked up both perturbations has no second derivatives, and offers neither
# the `length(x)` rows the method above reads nor, for an `f` ignoring its argument, any at all.
@inline function extract_hessian(::Type{T}, ::Type{TO}, ydual, x::S) where {T,TO,S<:StaticArray}
    R = StaticArrays.similar_type(S, valtype(T, valtype(TO, typeof(ydual))),
                                  Size(length(x), length(x)))
    return zero(R)
end

# The layers need distinct tags; see `ForwardDiff.outer_tag`.
@inline function hessian_tags(f::F, x::StaticArray) where {F}
    T = typeof(Tag(f, eltype(x)))
    return T, outer_tag(T, Dual{T,eltype(x),length(x)})
end

@inline function ForwardDiff.hessian(f::F, x::StaticArray) where {F}
    T, TO = hessian_tags(f, x)
    ydual = f(dualize(TO, dualize(T, x)))
    ydual isa Real || throw(HESSIAN_ERROR)
    return extract_hessian(T, TO, ydual, x)
end

ForwardDiff.hessian(f::F, x::StaticArray, cfg::HessianConfig) where {F} = hessian(f, x)
ForwardDiff.hessian(f::F, x::StaticArray, cfg::HessianConfig, ::Val) where {F} = hessian(f, x)

@inline function ForwardDiff.hessian!(result::AbstractArray, f::F, x::StaticArray) where {F}
    T, TO = hessian_tags(f, x)
    ydual = f(dualize(TO, dualize(T, x)))
    ydual isa Real || throw(HESSIAN_ERROR)
    H = ForwardDiff.reshape_hessian(result, x)
    ForwardDiff.extract_hessian_chunk!(T, TO, H, ydual, structural_linearindices(x), 0, 0, length(x), length(x))
    return result
end

ForwardDiff.hessian!(result::MutableDiffResult, f::F, x::StaticArray) where {F} = hessian!(result, f, x, HessianConfig(f, result, x))

ForwardDiff.hessian!(result::ImmutableDiffResult, f::F, x::StaticArray, cfg::HessianConfig) where {F} = hessian!(result, f, x)
ForwardDiff.hessian!(result::ImmutableDiffResult, f::F, x::StaticArray, cfg::HessianConfig, ::Val) where {F} = hessian!(result, f, x)

function ForwardDiff.hessian!(result::ImmutableDiffResult, f::F, x::StaticArray) where {F}
    T, TO = hessian_tags(f, x)
    d1 = dualize(T, x)
    d2 = dualize(TO, d1)
    fd2 = f(d2)
    fd2 isa Real || throw(HESSIAN_ERROR)
    val = value(T,value(TO,fd2))
    grad = extract_gradient(T,value(TO,fd2), x)
    hess = extract_hessian(T,TO,fd2, x)
    result = DiffResults.hessian!(result, hess)
    result = DiffResults.gradient!(result, grad)
    result = DiffResults.value!(result, val)
    return result
end

end

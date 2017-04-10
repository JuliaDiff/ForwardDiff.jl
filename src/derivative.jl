###############
# API methods #
###############

@inline function derivative(f::F, x::R) where {F,R<:Real}
    T = Tag(F, R)
    return extract_derivative(f(Dual{T}(x, one(x))))
end

@generated function derivative(f::F, x::NTuple{N,Real}) where {F,N}
    args = [:(Dual{T}(x[$i], Val{N}, Val{$i})) for i in 1:N]
    return quote
        $(Expr(:meta, :inline))
        T = Tag(F, typeof(x))
        extract_derivative(f($(args...)))
    end
end

@inline function derivative!(out, f::F, x::R) where {F,R<:Real}
    T = Tag(F, typeof(x))
    extract_derivative!(out, f(Dual{T}(x, one(x))))
    return out
end

@generated function derivative!(out::NTuple{N,Any}, f::F, x::NTuple{N,Real}) where {F,N}
    args = [:(Dual{T}(x[$i], Val{N}, Val{$i})) for i in 1:N]
    return quote
        $(Expr(:meta, :inline))
        T = Tag(F, typeof(x))
        extract_derivative!(out, f($(args...)))
    end
end

#####################
# result extraction #
#####################

# non-mutating #
#--------------#

@generated function extract_derivative(y::Dual{T,V,N}) where {T,V,N}
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:tuple, [:(partials(y, $i)) for i in 1:N]...))
    end
end

@inline extract_derivative{T,V}(y::Dual{T,V,1}) = partials(y, 1)
@inline extract_derivative(y::Real) = zero(y)
@inline extract_derivative(y::AbstractArray) = extract_derivative!(similar(y, valtype(eltype(y))), y)

# mutating #
#----------#

@generated function extract_derivative!(out::NTuple{N,Any}, y::Dual{T,V,N}) where {T,V,N}
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:block, [:(out[$i][] = partials(y, $i)) for i in 1:N]...))
        return out
    end
end

@generated function extract_derivative!(out::NTuple{N,Any}, y::AbstractArray) where {N}
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:block, [:(extract_derivative!(out[$i], y, $i)) for i in 1:N]...))
        return out
    end
end

extract_derivative!(out::AbstractArray, y::AbstractArray) = map!(extract_derivative, out, y)
extract_derivative!(out::AbstractArray, y::AbstractArray, p) = map!(x -> partials(x, p), out, y)

function extract_derivative!(out::DiffResult, y)
    DiffBase.value!(value, out, y)
    DiffBase.derivative!(extract_derivative, out, y)
    return out
end

function extract_derivative!(out::DiffResult, y::AbstractArray, p)
    DiffBase.value!(value, out, y)
    DiffBase.derivative!(x -> partials(x, p), out, y)
    return out
end

###############
# API methods #
###############

@inline function derivative(f::F, x::R) where {F,R<:Real}
    T = typeof(Tag(F, R))
    return extract_derivative(f(Dual{T}(x, one(x))))
end

@generated function derivative(f::F, x::NTuple{N,Real}) where {F,N}
    args = [:(Dual{T}(x[$i], Val{N}, Val{$i})) for i in 1:N]
    return quote
        $(Expr(:meta, :inline))
        T = typeof(Tag(F, typeof(x)))
        extract_derivative(f($(args...)), Chunk{N}())
    end
end

@inline function derivative!(out, f::F, x::R) where {F,R<:Real}
    T = typeof(Tag(F, typeof(x)))
    extract_derivative!(out, f(Dual{T}(x, one(x))))
    return out
end

@generated function derivative!(out::NTuple{N,Any}, f::F, x::NTuple{N,Real}) where {F,N}
    args = [:(Dual{T}(x[$i], Val{N}, Val{$i})) for i in 1:N]
    return quote
        $(Expr(:meta, :inline))
        T = typeof(Tag(F, typeof(x)))
        extract_derivative!(out, f($(args...)))
    end
end

#####################
# result extraction #
#####################

# non-mutating #
#--------------#

@inline extract_derivative(y::Dual{T,V,1}) where {T,V} = partials(y, 1)
@inline extract_derivative(y::Real) = zero(y)
@inline extract_derivative(y::AbstractArray) = extract_derivative!(similar(y, valtype(eltype(y))), y)

@generated function extract_derivative(y::Dual{T,V,N}, ::Chunk{N}) where {T,V,N}
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:tuple, [:(partials(y, $i)) for i in 1:N]...))
    end
end

@generated function extract_derivative(y::AbstractArray, ::Chunk{N}) where {N}
    return quote
        $(Expr(:meta, :inline))
        V = valtype(eltype(y))
        out = $(Expr(:tuple, [:(similar(y, V)) for i in 1:N]...))
        return extract_derivative!(out, y)
    end
end
# mutating #
#----------#

@generated function extract_derivative!(out::NTuple{N,Any}, y) where {N}
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:block, [:(extract_derivative!(out[$i], y, $i)) for i in 1:N]...))
        return out
    end
end

extract_derivative!(out::AbstractArray, y::AbstractArray) = map!(extract_derivative, out, y)
extract_derivative!(out::AbstractArray, y::AbstractArray, p) = map!(x -> partials(x, p), out, y)
extract_derivative!(out::Union{AbstractArray,Base.Ref}, y::Dual, p) = (out[] = partials(y, p); out)

function extract_derivative!(out::DiffResult, y)
    DiffBase.value!(value, out, y)
    DiffBase.derivative!(extract_derivative, out, y)
    return out
end

function extract_derivative!(out::DiffResult, y, p)
    DiffBase.value!(value, out, y)
    DiffBase.derivative!(x -> partials(x, p), out, y)
    return out
end

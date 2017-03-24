###############
# API methods #
###############

@generated function derivative{F,R<:Real}(f::F, x::R)
    T = Tag{F,order(R)}
    return quote
        $(Expr(:meta, :inline))
        return extract_derivative(f(Dual{$T}(x, one(x))))
    end
end

@generated function derivative{F,N}(f::F, x::NTuple{N,Real})
    T = Tag{F,maximum(order(R) for R in x.parameters)}
    args = [:(Dual{$T}(x[$i], Val{N}, Val{$i})) for i in 1:N]
    return quote
        $(Expr(:meta, :inline))
        extract_derivative(f($(args...)))
    end
end

@generated function derivative!{F,R<:Real}(out, f::F, x::R)
    T = Tag{F,order(R)}
    return quote
        $(Expr(:meta, :inline))
        extract_derivative!(out, f(Dual{$T}(x, one(x))))
        return out
    end
end

#####################
# result extraction #
#####################

@generated function extract_derivative{T,V,N}(y::Dual{T,V,N})
    return quote
        $(Expr(:meta, :inline))
        $(Expr(:tuple, [:(partials(y, $i)) for i in 1:N]...))
    end
end

@inline extract_derivative{T,V}(y::Dual{T,V,1}) = partials(y, 1)
@inline extract_derivative(y::Real) = zero(y)
@inline extract_derivative(y::AbstractArray) = extract_derivative!(similar(y, valtype(eltype(y))), y)

extract_derivative!(out::AbstractArray, y::AbstractArray) = map!(extract_derivative, out, y)

function extract_derivative!(out::DiffResult, y)
    DiffBase.value!(value, out, y)
    DiffBase.derivative!(extract_derivative, out, y)
    return out
end

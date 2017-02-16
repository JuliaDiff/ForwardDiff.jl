###############
# API methods #
###############

derivative{F}(f::F, x::Real) = extract_derivative(f(Dual(x, one(x))))

@generated function derivative{F,N}(f::F, x::NTuple{N,Real})
    args = [:(Dual(x[$i], Val{N}, Val{$i})) for i in 1:N]
    return :(extract_derivative(f($(args...))))
end

function derivative!{F}(out, f::F, x::Real)
    y = f(Dual(x, one(x)))
    extract_derivative!(out, y)
    return out
end

#####################
# result extraction #
#####################

@generated extract_derivative{N}(y::Dual{N}) = Expr(:tuple, [:(partials(y, $i)) for i in 1:N]...)

@inline extract_derivative(y::Dual{1}) = partials(y, 1)
@inline extract_derivative(y::Real) = zero(y)
@inline extract_derivative(y::AbstractArray) = extract_derivative!(similar(y, valtype(eltype(y))), y)

extract_derivative!(out::AbstractArray, y::AbstractArray) = map!(extract_derivative, out, y)

function extract_derivative!(out::DiffResult, y)
    DiffBase.value!(value, out, y)
    DiffBase.derivative!(extract_derivative, out, y)
    return out
end

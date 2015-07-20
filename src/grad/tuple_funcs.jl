##################################
# Generated Functions on NTuples #
##################################
# The below functions are generally
# equivalent to directly mapping over
# tuples using `map`, but run a bit
# faster since they generate inline code
# that doesn't rely on closures.

function tupexpr(f,N)
    ex = Expr(:tuple, [f(i) for i=1:N]...)
    return quote
        @inbounds return $ex
    end
end

@generated function zero_tuple{N,T}(::Type{NTuple{N,T}})
    result = tupexpr(i -> :z, N)
    return quote
        z = zero($T)
        return $result
    end
end

zero_tuple(::Type{Tuple{}}) = tuple()

@generated scale_tuple{N}(x, tup::NTuple{N}) = tupexpr(i -> :(x * tup[$i]), N)

@generated div_tuple_by_scalar{N}(tup::NTuple{N}, x) = tupexpr(i -> :(tup[$i]/x), N)

@generated minus_tuple{N}(tup::NTuple{N}) = tupexpr(i -> :(-tup[$i]), N)

@generated subtract_tuples{N}(a::NTuple{N}, b::NTuple{N}) = tupexpr(i -> :(a[$i]-b[$i]), N)

@generated add_tuples{N}(a::NTuple{N}, b::NTuple{N}) = tupexpr(i -> :(a[$i]+b[$i]), N)

@generated function mul_dus{N}(zdus::NTuple{N}, wdus::NTuple{N}, z_r, w_r)
    return tupexpr(i -> :((zdus[$i] * w_r) + (z_r * wdus[$i])), N)
end

@generated function div_dus{N}(zdus::NTuple{N}, wdus::NTuple{N}, z_r, w_r, denom)
    return tupexpr(i -> :(((zdus[$i] * w_r) - (z_r * wdus[$i]))/denom), N)
end

@generated function div_real_by_dus{N}(neg_x, dus::NTuple{N}, z_r_sq)
    return tupexpr(i -> :((neg_x * dus[$i])/z_r_sq), N)
end
# TODO: make ths work with OffsetArrays
# TODO: optimize performance
# TODO: add documentation
# TODO: use a reasonable API

"""
    pack(x::AbstractArray{T}...) where T -> (format, AbstractArray{T})

Pack a tuple of arrays into a format and single array for use in multi_gradient.

This can be reversed with `unpack(format, array)`, so that `unpack(pack(x...)...) == x`.

This is not part of the public API
"""
pack(x::AbstractArray{T}...) where T = two(Pack{T, typeof(x)}(x))
two(x) = (x,x)

struct Pack{T,U} <: AbstractVector{T}
    arrays::U
end

unpack(format::Pack, array::Pack) = format === array ? array.arrays : _unpack(format, array)
unpack(format::Pack, array::AbstractArray) = _unpack(format, array)

function _unpack(format::Pack, array::AbstractArray)
    i, t = foldl(format.arrays, init=(firstindex(array), ())) do (i, t), a
        j = i + length(a)
        (j, (t..., reshape(view(array, i:j-1), size(a))))
    end
    i == lastindex(array) + 1 || throw(DimensionMismatch("array is too long"))
    t
end


"""
    multi_gradient(f, xs...)

Like `gradient`, but can accept multiple arrays and returns a tuple of gradients.

This is part of the public API
"""
mulit_gradient(f, x) = (gradient(f, x),)
function multi_gradient(f, xs...)
    fr, a = pack(xs...)
    unpack(fr, gradient(x -> f(unpack(fr, x)...), a))
end

# These functions should rarely be called but are needed to satisfy the AbstractArray
# interface. They have poor performance.
Base.size(p::Pack) = (sum(length, p.arrays),)
function Base.getindex(p::Pack, i::Int)
    i < 1 && throw(BoundsError(p, i))
    i0 = i
    for a in p.arrays
        if i <= length(a)
            return getindex(a, firstindex(a) + i - 1)
        end
        i -= length(a)
    end
    throw(BoundsError(p, i0))
end
function Base.setindex!(p::Pack, v, i::Int)
    i < 1 && throw(BoundsError(p, i))
    i0 = i
    for a in p.arrays
        if i <= length(a)
            setindex!(a, v, firstindex(a) + i - 1)
            return p
        end
        i -= length(a)
    end
    throw(BoundsError(p, i0))
end

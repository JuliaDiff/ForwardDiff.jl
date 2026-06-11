const NANSAFE_MODE_ENABLED = @load_preference("nansafe_mode", false)
const DEFAULT_CHUNK_THRESHOLD = @load_preference("default_chunk_threshold", 12)

const AMBIGUOUS_TYPES = (AbstractFloat, Irrational, Integer, Rational, Real, RoundingMode)

const UNARY_PREDICATES = Symbol[:isinf, :isnan, :isfinite, :iseven, :isodd, :isreal, :isinteger]

struct Chunk{N} end

function Chunk(input_length::Integer, threshold::Integer = DEFAULT_CHUNK_THRESHOLD)
    N = pickchunksize(input_length, threshold)
    Base.@nif 12 d->(N == d) d->(Chunk{d}()) d->(Chunk{N}())
end

structural_length(x::AbstractArray) = length(x)
function structural_length(x::Union{LowerTriangular,UpperTriangular})
    n = size(x, 1)
    return (n * (n + 1)) >> 1
end
structural_length(x::Diagonal) = size(x, 1)

function Chunk(x::AbstractArray, threshold::Integer = DEFAULT_CHUNK_THRESHOLD)
    return Chunk(structural_length(x), threshold)
end

# Constrained to `N <= threshold`, minimize (in order of priority):
#   1. the number of chunks that need to be computed
#   2. the number of "left over" perturbations in the final chunk
function pickchunksize(input_length, threshold = DEFAULT_CHUNK_THRESHOLD)
    if input_length <= threshold
        return input_length
    else
        nchunks = round(Int, input_length / threshold, RoundUp)
        return round(Int, input_length / nchunks, RoundUp)
    end
end

chunksize(::Chunk{N}) where {N} = N

replace_match!(f, ismatch, x) = x

function replace_match!(f, ismatch, lines::AbstractArray)
    for i in eachindex(lines)
        line = lines[i]
        if ismatch(line)
            lines[i] = f(line)
        elseif isa(line, Expr)
            replace_match!(f, ismatch, line.args)
        end
    end
    return lines
end

# This is basically a workaround that allows one to use CommonSubexpressions.cse, but with
# qualified bindings. This is not guaranteed to be correct if the input expression contains
# field accesses.
function qualified_cse!(expr)
    placeholders = Dict{Symbol,Expr}()
    replace_match!(x -> isa(x, Expr) && x.head == :(.), expr.args) do x
        placeholder = Symbol("#$(hash(x))")
        placeholders[placeholder] = x
        placeholder
    end
    cse_expr = CommonSubexpressions.cse(expr, warn=false)
    replace_match!(x -> haskey(placeholders, x), cse_expr.args) do x
        placeholders[x]
    end
    return cse_expr
end

#= 
######
# AbstractTag interface

Required definitions: 
- ≺ (between two AbstractTags of the same type)
- maketagtype(f,::Type{V}) where {V <: Real}

Optional definitions:
- ≺ (between two AbstractTags of the of different type)
- maketag(f,::Type{V}) where {V <: Real} (default: defined in terms of maketagtype)
- checktag(tag::MyTagType,f,x)
###### 
=#
abstract type AbstractTag{F,V} end

@inline maketagtype(f::Nothing,::Type{V}) where {V} = Nothing
@inline maketag(f::F,::Type{V}) where {F,V} = maketagtype(f,V)()

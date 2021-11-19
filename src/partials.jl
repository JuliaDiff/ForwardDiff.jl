struct Partials{N,V} <: AbstractVector{V}
    values::NTuple{N,V}
end

##############################
# Utility/Accessor Functions #
##############################

@generated function single_seed(::Type{Partials{N,V}}, ::Val{i}) where {N,V,i}
    ex = Expr(:tuple, [ifelse(i === j, :(one(V)), :(zero(V))) for j in 1:N]...)
    return :(Partials($(ex)))
end

@inline valtype(::Partials{N,V}) where {N,V} = V
@inline valtype(::Type{Partials{N,V}}) where {N,V} = V

@inline npartials(::Partials{N}) where {N} = N
@inline npartials(::Type{Partials{N,V}}) where {N,V} = N

@inline Base.length(::Partials{N}) where {N} = N
@inline Base.size(::Partials{N}) where {N} = (N,)

@inline Base.@propagate_inbounds Base.getindex(partials::Partials, i::Int) = partials.values[i]

Base.iterate(partials::Partials) = iterate(partials.values)
Base.iterate(partials::Partials, i) = iterate(partials.values, i)

Base.IndexStyle(::Type{<:Partials}) = IndexLinear()

# Can be deleted after https://github.com/JuliaLang/julia/pull/29854 is on a release
Base.mightalias(x::AbstractArray, y::Partials) = false

#####################
# Generic Functions #
#####################

@inline Base.iszero(partials::Partials) = iszero_tuple(partials.values)

@inline Base.zero(partials::Partials) = zero(typeof(partials))
@inline Base.zero(::Type{Partials{N,V}}) where {N,V} = Partials{N,V}(zero_tuple(NTuple{N,V}))

@inline Base.one(partials::Partials) = one(typeof(partials))
@inline Base.one(::Type{Partials{N,V}}) where {N,V} = Partials{N,V}(one_tuple(NTuple{N,V}))

@inline Random.rand(partials::Partials) = rand(typeof(partials))
@inline Random.rand(::Type{Partials{N,V}}) where {N,V} = Partials{N,V}(rand_tuple(NTuple{N,V}))
@inline Random.rand(rng::AbstractRNG, partials::Partials) = rand(rng, typeof(partials))
@inline Random.rand(rng::AbstractRNG, ::Type{Partials{N,V}}) where {N,V} = Partials{N,V}(rand_tuple(rng, NTuple{N,V}))

Base.isequal(a::Partials{N}, b::Partials{N}) where {N} = isequal(a.values, b.values)
Base.:(==)(a::Partials{N}, b::Partials{N}) where {N} = a.values == b.values

const PARTIALS_HASH = hash(Partials)

Base.hash(partials::Partials) = hash(partials.values, PARTIALS_HASH)
Base.hash(partials::Partials, hsh::UInt64) = hash(hash(partials), hsh)

@inline Base.copy(partials::Partials) = partials

Base.read(io::IO, ::Type{Partials{N,V}}) where {N,V} = Partials{N,V}(ntuple(i->read(io, V), N))

function Base.write(io::IO, partials::Partials)
    for p in partials
        write(io, p)
    end
end

########################
# Conversion/Promotion #
########################

Base.promote_rule(::Type{Partials{N,A}}, ::Type{Partials{N,B}}) where {N,A,B} = Partials{N,promote_type(A, B)}

Base.convert(::Type{Partials{N,V}}, partials::Partials) where {N,V} = Partials{N,V}(partials.values)
Base.convert(::Type{Partials{N,V}}, partials::Partials{N,V}) where {N,V} = partials

########################
# Arithmetic Functions #
########################

@inline Base.:+(a::Partials{N}, b::Partials{N}) where {N} = Partials(add_tuples(a.values, b.values))
@inline Base.:-(a::Partials{N}, b::Partials{N}) where {N} = Partials(sub_tuples(a.values, b.values))
@inline Base.:-(partials::Partials) = Partials(minus_tuple(partials.values))
@inline Base.:*(x::Real, partials::Partials) = partials*x

@inline function _div_partials(a::Partials, b::Partials, aval, bval)
    return _mul_partials(a, b, inv(bval), -(aval / (bval*bval)))
end

# NaN/Inf-safe methods #
#----------------------#

if NANSAFE_MODE_ENABLED
    @inline function Base.:*(partials::Partials, x::Real)
        x = ifelse(!isfinite(x) && iszero(partials), one(x), x)
        return Partials(scale_tuple(partials.values, x))
    end

    @inline function Base.:/(partials::Partials, x::Real)
        x = ifelse(x == zero(x) && iszero(partials), one(x), x)
        return Partials(div_tuple_by_scalar(partials.values, x))
    end

    @inline function _mul_partials(a::Partials{N}, b::Partials{N}, x_a, x_b) where N
        x_a = ifelse(!isfinite(x_a) && iszero(a), one(x_a), x_a)
        x_b = ifelse(!isfinite(x_b) && iszero(b), one(x_b), x_b)
        return Partials(mul_tuples(a.values, b.values, x_a, x_b))
    end
else
    @inline function Base.:*(partials::Partials, x::Real)
        return Partials(scale_tuple(partials.values, x))
    end

    @inline function Base.:/(partials::Partials, x::Real)
        return Partials(div_tuple_by_scalar(partials.values, x))
    end

    @inline function _mul_partials(a::Partials{N}, b::Partials{N}, x_a, x_b) where N
        return Partials(mul_tuples(a.values, b.values, x_a, x_b))
    end
end

# edge cases where N == 0 #
#-------------------------#

@inline Base.:+(a::Partials{0,A}, b::Partials{0,B}) where {A,B} = Partials{0,promote_type(A,B)}(tuple())
@inline Base.:+(a::Partials{0,A}, b::Partials{N,B}) where {N,A,B} = convert(Partials{N,promote_type(A,B)}, b)
@inline Base.:+(a::Partials{N,A}, b::Partials{0,B}) where {N,A,B} = convert(Partials{N,promote_type(A,B)}, a)

@inline Base.:-(a::Partials{0,A}, b::Partials{0,B}) where {A,B} = Partials{0,promote_type(A,B)}(tuple())
@inline Base.:-(a::Partials{0,A}, b::Partials{N,B}) where {N,A,B} = -(convert(Partials{N,promote_type(A,B)}, b))
@inline Base.:-(a::Partials{N,A}, b::Partials{0,B}) where {N,A,B} = convert(Partials{N,promote_type(A,B)}, a)
@inline Base.:-(partials::Partials{0,V}) where {V} = partials

@inline Base.:*(partials::Partials{0,V}, x::Real) where {V} = Partials{0,promote_type(V,typeof(x))}(tuple())
@inline Base.:*(x::Real, partials::Partials{0,V}) where {V} = Partials{0,promote_type(V,typeof(x))}(tuple())

@inline Base.:/(partials::Partials{0,V}, x::Real) where {V} = Partials{0,promote_type(V,typeof(x))}(tuple())

@inline _mul_partials(a::Partials{0,A}, b::Partials{0,B}, afactor, bfactor) where {A,B} = Partials{0,promote_type(A,B)}(tuple())
@inline _mul_partials(a::Partials{0,A}, b::Partials{N,B}, afactor, bfactor) where {N,A,B} = bfactor * b
@inline _mul_partials(a::Partials{N,A}, b::Partials{0,B}, afactor, bfactor) where {N,A,B} = afactor * a

##################################
# Generated Functions on NTuples #
##################################
# The below functions are generally
# equivalent to directly mapping over
# tuples using `map`, but run a bit
# faster since they generate inline code
# that doesn't rely on closures.

function tupexpr(f, N)
    ex = Expr(:tuple, [f(i) for i=1:N]...)
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $ex
    end
end

@inline iszero_tuple(::Tuple{}) = true
@inline zero_tuple(::Type{Tuple{}}) = tuple()
@inline one_tuple(::Type{Tuple{}}) = tuple()
@inline rand_tuple(::AbstractRNG, ::Type{Tuple{}}) = tuple()
@inline rand_tuple(::Type{Tuple{}}) = tuple()

@generated function iszero_tuple(tup::NTuple{N,V}) where {N,V}
    ex = Expr(:&&, [:(z == tup[$i]) for i=1:N]...)
    return quote
        z = zero(V)
        $(Expr(:meta, :inline))
        @inbounds return $ex
    end
end

@generated function zero_tuple(::Type{NTuple{N,V}}) where {N,V}
    ex = tupexpr(i -> :(z), N)
    return quote
        z = zero(V)
        return $ex
    end
end

@generated function one_tuple(::Type{NTuple{N,V}}) where {N,V}
    ex = tupexpr(i -> :(z), N)
    return quote
        z = one(V)
        return $ex
    end
end

@generated function rand_tuple(rng::AbstractRNG, ::Type{NTuple{N,V}}) where {N,V}
    return tupexpr(i -> :(rand(rng, V)), N)
end

@generated function rand_tuple(::Type{NTuple{N,V}}) where {N,V}
    return tupexpr(i -> :(rand(V)), N)
end

const SIMDFloat = Union{Float64, Float32}
const SIMDInt = Union{
                       Int128, Int64, Int32, Int16, Int8,
                       UInt128, UInt64, UInt32, UInt16, UInt8,
                       Bool
                     }
const SIMDType = Union{SIMDFloat, SIMDInt}

function julia_type_to_llvm_type(@nospecialize(T::DataType))
    T === Float64 ? "double" :
    T === Float32 ? "float"  :
    T <: Union{Int128,UInt128} ? "i128" :
    T <: Union{Int64,UInt64} ? "i64" :
    T <: Union{Int32,UInt32} ? "i32" :
    T <: Union{Int16,UInt16} ? "i16" :
    T <: Union{Bool,Int8,UInt8} ? "i8" :
    error("$T cannot be mapped to a LLVM type")
end

@generated function scale_tuple(tup::NTuple{N,T}, x::S) where {N,T,S}
    if !(T === S && S <: SIMDType)
        return tupexpr(i -> :(tup[$i] * x), N)
    end

    S = julia_type_to_llvm_type(T)
    VT = NTuple{N, VecElement{T}}
    op = T <: SIMDFloat ? "fmul nsz contract" : "mul"
    llvmir = """
    %el = insertelement <$N x $S> undef, $S %1, i32 0
    %vx = shufflevector <$N x $S> %el, <$N x $S> undef, <$N x i32> zeroinitializer
    %res = $op <$N x $S> %0, %vx
    ret <$N x $S> %res
    """

    quote
        $(Expr(:meta, :inline))
        ret = Base.llvmcall($llvmir, $VT, Tuple{$VT, $T}, $VT(tup), x)
        Base.@ntuple $N i->ret[i].value
    end
end

@generated function div_tuple_by_scalar(tup::NTuple{N,T}, x::S) where {N,T,S}
    if !(T === S === typeof(one(T) / one(S)) && S <: SIMDType)
        return tupexpr(i -> :(tup[$i] / x), N)
    end

    S = julia_type_to_llvm_type(T)
    VT = NTuple{N, VecElement{T}}
    op = T <: SIMDFloat ? "fdiv nsz contract" : "div"
    llvmir = """
    %el = insertelement <$N x $S> undef, $S %1, i32 0
    %vx = shufflevector <$N x $S> %el, <$N x $S> undef, <$N x i32> zeroinitializer
    %res = $op <$N x $S> %0, %vx
    ret <$N x $S> %res
    """

    quote
        $(Expr(:meta, :inline))
        ret = Base.llvmcall($llvmir, $VT, Tuple{$VT, $T}, $VT(tup), x)
        Base.@ntuple $N i->ret[i].value
    end
end

@generated function add_tuples(a::NTuple{N,T}, b::NTuple{N,S}) where {N,T,S}
    if !(T === S && S <: SIMDType)
        return tupexpr(i -> :(a[$i] + b[$i]), N)
    end

    S = julia_type_to_llvm_type(T)
    VT = NTuple{N, VecElement{T}}
    op = T <: SIMDFloat ? "fadd nsz contract" : "add"
    llvmir = """
    %res = $op <$N x $S> %0, %1
    ret <$N x $S> %res
    """

    quote
        $(Expr(:meta, :inline))
        ret = Base.llvmcall($llvmir, $VT, Tuple{$VT, $VT}, $VT(a), $VT(b))
        Base.@ntuple $N i->ret[i].value
    end
end

@generated function sub_tuples(a::NTuple{N,T}, b::NTuple{N,S}) where {N,T,S}
    if !(T === S && S <: SIMDType)
        return tupexpr(i -> :(a[$i] - b[$i]), N)
    end

    S = julia_type_to_llvm_type(T)
    VT = NTuple{N, VecElement{T}}
    op = T <: SIMDFloat ? "fsub nsz contract" : "sub"
    llvmir = """
    %res = $op <$N x $S> %0, %1
    ret <$N x $S> %res
    """

    quote
        $(Expr(:meta, :inline))
        ret = Base.llvmcall($llvmir, $VT, Tuple{$VT, $VT}, $VT(a), $VT(b))
        Base.@ntuple $N i->ret[i].value
    end
end

if VERSION >= v"1.4" # fsub requires LLVM 8 (Julia 1.4)
    @generated function minus_tuple(tup::NTuple{N,T}) where {N,T}
        T <: SIMDType || return tupexpr(i -> :(-tup[$i]), N)

        S = julia_type_to_llvm_type(T)
        VT = NTuple{N, VecElement{T}}
        if T <: SIMDFloat
            llvmir = """
            %res = fneg nsz contract <$N x $S> %0
            ret <$N x $S> %res
            """
        else
            llvmir = """
            %res = sub <$N x $S> zeroinitializer, %0
            ret <$N x $S> %res
            """
        end

        quote
            $(Expr(:meta, :inline))
            ret = Base.llvmcall($llvmir, $VT, Tuple{$VT}, $VT(tup))
            Base.@ntuple $N i->ret[i].value
        end
    end
else
    @generated function minus_tuple(tup::NTuple{N,T}) where {N,T}
        return tupexpr(i -> :(-tup[$i]), N)
    end
end

@generated function mul_tuples(a::NTuple{N,V1}, b::NTuple{N,V2}, afactor::S1, bfactor::S2) where {N,V1,V2,S1,S2}
    if !(V1 === V2 === S1 === S2 && S2 <: SIMDFloat)
        return tupexpr(i -> :((afactor * a[$i]) + (bfactor * b[$i])), N)
    end

    T = V1
    S = julia_type_to_llvm_type(T)
    fmuladd = "@llvm.fmuladd.v$(N)f$(sizeof(T)*8)"

    VT = NTuple{N, VecElement{T}}
    llvmir = """
    declare <$N x $S> $fmuladd(<$N x $S>, <$N x $S>, <$N x $S>)

    define <$N x $S> @entry(<$N x $S>, <$N x $S>, $S, $S) alwaysinline {
    top:
        %el1 = insertelement <$N x $S> undef, $S %2, i32 0
        %afactor = shufflevector <$N x $S> %el1, <$N x $S> undef, <$N x i32> zeroinitializer
        %el2 = insertelement <$N x $S> undef, $S %3, i32 0
        %bfactor = shufflevector <$N x $S> %el2, <$N x $S> undef, <$N x i32> zeroinitializer
        %tmp = fmul nsz contract <$N x $S> %1, %bfactor
        %res = call nsz contract <$N x $S> $fmuladd(<$N x $S> %0, <$N x $S> %afactor, <$N x $S> %tmp)
        ret <$N x $S> %res
    }
    """
    quote
        $(Expr(:meta, :inline))
        ret = Base.llvmcall(($llvmir, "entry"), $VT, Tuple{$VT, $VT, $T, $T}, $VT(a), $VT(b), afactor, bfactor)
        Base.@ntuple $N i->ret[i].value
    end
end

###################
# Pretty Printing #
###################

Base.show(io::IO, p::Partials{N}) where {N} = print(io, "Partials", p.values)

########################
# Math with GradNumVec #
########################

# Addition/Subtraction #
#----------------------#
+{N,A,B}(z::GradNumVec{N,A}, w::GradNumVec{N,B}) = GradNumVec{N,promote_type(A,B)}(value(z)+value(w), grad(z)+grad(w))
+{N,T}(x::Real, z::GradNumVec{N,T}) = GradNumVec{N,promote_type(typeof(x),T)}(x+value(z), grad(z))
+{N,T}(z::GradNumVec{N,T}, x::Real) = x+z

-{N,T}(z::GradNumVec{N,T}) = GradNumVec{N,T}(-value(z), -grad(z))
-{N,A,B}(z::GradNumVec{N,A}, w::GradNumVec{N,B}) = GradNumVec{N,promote_type(A,B)}(value(z)-value(w), grad(z)-grad(w))
-{N,T}(x::Real, z::GradNumVec{N,T}) = GradNumVec{N,promote_type(typeof(x),T)}(x-value(z), -grad(z))
-{N,T}(z::GradNumVec{N,T}, x::Real) = GradNumVec{N,promote_type(T,typeof(x))}(value(z)-x, grad(z))

# Multiplication #
#----------------#
# avoid ambiguous definition with Bool*Number
*{N,T}(x::Bool, z::GradNumVec{N,T}) = ifelse(x, z, ifelse(signbit(value(z))==0, zero(z), -zero(z)))
*{N,T}(z::GradNumVec{N,T}, x::Bool) = x*z

function *{N,A,B}(z::GradNumVec{N,A}, w::GradNumVec{N,B})
    z_r, w_r = value(z), value(w)
    T = promote_type(A,B)
    dus = mul_dus!(Array(T, N), grad(z), grad(w), z_r, w_r)
    return GradNumVec{N,T}(z_r*w_r, dus)
end

function mul_dus!(result, zdus, wdus, z_r, w_r)
    @simd for i=1:length(result)
        @inbounds result[i] = (zdus[i] * w_r) + (z_r * wdus[i])
    end
    return result
end

function *{N,A}(x::Real, z::GradNumVec{N,A})
    T = promote_type(typeof(x),A)
    return GradNumVec{N,T}(x*value(z), x*grad(z))
end

*{N,T}(z::GradNumVec{N,T}, x::Real) = x*z

# Division #
#----------#
/{N,A,B<:Real}(z::GradNumVec{N,A}, x::B) = GradNumVec{N,promote_type(A,B,Float64)}(value(z)/x, grad(z)/x)

function /{N,A<:Real,B}(x::A, z::GradNumVec{N,B})
    z_r = value(z)
    T = promote_type(A, B, Float64)
    dus = div_real_by_dus!(Array(T, N), -x, grad(z), z_r^2)
    return GradNumVec{N,T}(x/z_r, dus)
end

function div_real_by_dus!(result, neg_x, dus, z_r_sq)
    @simd for i=1:length(result)
        @inbounds result[i] = (neg_x * dus[i]) / z_r_sq
    end
    return result
end

function /{N,A,B}(z::GradNumVec{N,A}, w::GradNumVec{N,B})
    z_r, w_r = value(z), value(w)
    T = promote_type(A, B, Float64)
    dus = div_dus!(Array(T,N), grad(z), grad(w), z_r, w_r, w_r^2)
    return GradNumVec{N,T}(z_r/w_r, dus)
end

function div_dus!(result, zdus, wdus, z_r, w_r, denom)
    @simd for i=1:length(result)
        @inbounds result[i] = ((zdus[i] * w_r) - (z_r * wdus[i]))/denom
    end
    return result
end

# Exponentiation #
#----------------#
for f in (:^, :(NaNMath.pow))

    @eval function ($f){N,A,B}(z::GradNumVec{N,A}, w::GradNumVec{N,B})
        z_r, w_r = value(z), value(w)    
        re = $f(z_r, w_r)
        powval = w_r * (($f)(z_r, w_r-1))
        logval = ($f)(z_r, w_r) * log(z_r)
        T = promote_type(A,B)
        dus = mul_dus!(Array(T, N), grad(z), grad(w), logval, powval)
        return GradNumVec{N,T}(re, dus)
    end

    @eval ($f)(::Base.MathConst{:e}, z::GradNumVec) = exp(z)

    # generate redundant definitions to resolve ambiguity warnings
    for R in (:Integer, :Rational, :Real)
        @eval function ($f){N,A}(z::GradNumVec{N,A}, x::$R)
            z_r = value(z)
            powval = x*($f)(z_r, x-1)
            T = promote_type(A,typeof(x))
            return GradNumVec{N,T}(($f)(z_r, x), powval*grad(z))
        end

        @eval function ($f){N,A}(x::$R, z::GradNumVec{N,A})
            z_r = value(z)
            logval = ($f)(x, z_r)*log(x)
            T = promote_type(typeof(x), A)
            return GradNumVec{N,T}(($f)(x, z_r), logval*grad(z))
        end
    end

end

# from Calculus.jl #
#------------------#
for fsym in fad_supported_univar_funcs
    fsym == :exp && continue
    
    valexpr = :($(fsym)(x))
    dfexpr = Calculus.differentiate(valexpr)

    expr = parse("""
        @generated function $(fsym){N,A}(z::GradNumVec{N,A})
            T = typeof($(fsym)(one(A)))
            return quote
                x = value(z)
                df = $dfexpr
                return GradNumVec{N,\$T}($valexpr, df*grad(z))
            end
        end
    """)

    @eval $expr

    # extend corresponding NaNMath methods
    if fsym in (:sin, :cos, :tan,
                :asin, :acos, :acosh,
                :atanh, :log, :log2,
                :log10, :lgamma, :log1p)

        nan_fsym = Expr(:.,:NaNMath,Base.Meta.quot(fsym))
        nan_valexpr = :($(nan_fsym)(x))
        nan_dfexpr = to_nanmath(dfexpr)

        expr = parse("""
            @generated function $(nan_fsym){N,A}(z::GradNumVec{N,A})
                T = typeof($(nan_fsym)(one(A)))
                return quote
                    x = value(z)
                    df = $nan_dfexpr
                    return GradNumVec{N,\$T}($nan_valexpr, df*grad(z))
                end
            end
        """)

        @eval $expr
    end
end

@generated function exp{N,A}(z::GradNumVec{N,A})
    T = typeof(exp(one(A)))
    return quote 
        df = exp(value(z))
        return GradNumVec{N,$T}(df, df*grad(z))
    end
end

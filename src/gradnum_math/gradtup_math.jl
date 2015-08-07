########################
# Math with GradNumTup #
########################

# Addition/Subtraction #
#----------------------#
+{N,A,B}(z::GradNumTup{N,A}, w::GradNumTup{N,B}) = GradientNum(value(z)+value(w), add_tuples(grad(z), grad(w)))
+{N,T}(x::Real, z::GradNumTup{N,T}) = GradientNum(x+value(z), grad(z))
+{N,T}(z::GradNumTup{N,T}, x::Real) = x+z

-{N,T}(z::GradNumTup{N,T}) = GradientNum(-value(z), minus_tuple(grad(z)))
-{N,A,B}(z::GradNumTup{N,A}, w::GradNumTup{N,B}) = GradientNum(value(z)-value(w), subtract_tuples(grad(z), grad(w)))
-{N,T}(x::Real, z::GradNumTup{N,T}) = GradientNum(x-value(z), minus_tuple(grad(z)))
-{N,T}(z::GradNumTup{N,T}, x::Real) = GradientNum(value(z)-x, grad(z))

# Multiplication #
#----------------#
# avoid ambiguous definition with Bool*Number
*{N,T}(x::Bool, z::GradNumTup{N,T}) = ifelse(x, z, ifelse(signbit(value(z))==0, zero(z), -zero(z)))
*{N,T}(z::GradNumTup{N,T}, x::Bool) = x*z

function *{N,A,B}(z::GradNumTup{N,A}, w::GradNumTup{N,B})
    z_r, w_r = value(z), value(w)
    dus =  mul_dus(grad(z), grad(w), z_r, w_r)
    return GradientNum(z_r*w_r, dus)
end

*{N,T}(x::Real, z::GradNumTup{N,T}) = GradientNum(x*value(z), scale_tuple(x, grad(z)))
*{N,T}(z::GradNumTup{N,T}, x::Real) = x*z

# Division #
#----------#
/{N,T}(z::GradNumTup{N,T}, x::Real) = GradientNum(value(z)/x, div_tuple_by_scalar(grad(z), x))

function /{N,T}(x::Real, z::GradNumTup{N,T})
    z_r = value(z)
    return GradientNum(x/z_r, div_real_by_dus(-x, grad(z), z_r^2))
end

function /{N,A,B}(z::GradNumTup{N,A}, w::GradNumTup{N,B})
    z_r, w_r = value(z), value(w)    
    dus = div_dus(grad(z), grad(w), z_r, w_r, w_r^2)
    return GradientNum(z_r/w_r, dus)
end

# Exponentiation #
#----------------#
for f in (:^, :(NaNMath.pow))

    @eval function ($f){N,A,B}(z::GradNumTup{N,A}, w::GradNumTup{N,B})
        z_r, w_r = value(z), value(w)    
        re = $f(z_r, w_r)
        powval = w_r * (($f)(z_r, w_r-1))
        logval = ($f)(z_r, w_r) * log(z_r)
        dus = mul_dus(grad(z), grad(w), logval, powval)
        return GradientNum(re, dus)
    end

    @eval ($f){N,T}(::Base.MathConst{:e}, z::GradNumTup{N,T}) = exp(z)

    # generate redundant definitions to resolve ambiguity warnings
    for R in (:Integer, :Rational, :Real)
        @eval function ($f){N,T}(z::GradNumTup{N,T}, x::($R))
            z_r = value(z)
            powval = x*($f)(z_r, x-1)
            return GradientNum(($f)(z_r, x), scale_tuple(powval, grad(z)))
        end

        @eval function ($f){N,T}(x::($R), z::GradNumTup{N,T})
            z_r = value(z)
            logval = ($f)(x, z_r)*log(x)
            return GradientNum(($f)(x, z_r), scale_tuple(logval, grad(z)))
        end
    end

end

# from Calculus.jl #
#------------------#
for fsym in fad_supported_univar_funcs
    fsym == :exp && continue
    
    valexpr = :($(fsym)(x))
    dfexpr = Calculus.differentiate(valexpr)

    @eval function $(fsym){N,T}(z::GradNumTup{N,T})
        x = value(z)
        df = $dfexpr
        return GradientNum($valexpr, scale_tuple(df, grad(z)))
    end

    # extend corresponding NaNMath methods
    if fsym in (:sin, :cos, :tan, 
                :asin, :acos, :acosh, 
                :atanh, :log, :log2, 
                :log10, :lgamma, :log1p)

        nan_fsym = Expr(:.,:NaNMath,Base.Meta.quot(fsym))

        @eval function $(nan_fsym){N,T}(z::GradNumTup{N,T})
            x = value(z)
            df = $(to_nanmath(dfexpr))
            return GradientNum($(nan_fsym)(x), scale_tuple(df, grad(z)))
        end

    end
end

function exp{N,T}(z::GradNumTup{N,T})
    df = exp(value(z))
    return GradientNum(df, scale_tuple(df, grad(z)))
end
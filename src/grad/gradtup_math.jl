########################
# Math with GradTup #
########################

## Addition/Subtraction ##
##----------------------##

+{N,A,B}(z::GradTup{N,A}, w::GradTup{N,B}) = GradTup(value(z)+value(w), add_tuples(partials(z), partials(w)))
+{N,T}(x::Real, z::GradTup{N,T}) = GradTup(x+value(z), partials(z))
+{N,T}(z::GradTup{N,T}, x::Real) = x+z

-{N,T}(z::GradTup{N,T}) = GradTup(-value(z), minus_tuple(partials(z)))
-{N,A,B}(z::GradTup{N,A}, w::GradTup{N,B}) = GradTup(value(z)-value(w), subtract_tuples(partials(z), partials(w)))
-{N,T}(x::Real, z::GradTup{N,T}) = GradTup(x-value(z), minus_tuple(partials(z)))
-{N,T}(z::GradTup{N,T}, x::Real) = GradTup(value(z)-x, partials(z))

## Multiplication ##
##----------------##

# avoid ambiguous definition with Bool*Number
*{N,T}(x::Bool, z::GradTup{N,T}) = ifelse(x, z, ifelse(signbit(value(z))==0, zero(z), -zero(z)))
*{N,T}(z::GradTup{N,T}, x::Bool) = x*z

function *{N,A,B}(z::GradTup{N,A}, w::GradTup{N,B})
    z_r, w_r = value(z), value(w)
    dus =  mul_dus(partials(z), partials(w), z_r, w_r)
    return GradTup(z_r*w_r, dus)
end

*{N,T}(x::Real, z::GradTup{N,T}) = GradTup(x*value(z), scale_tuple(x, partials(z)))
*{N,T}(z::GradTup{N,T}, x::Real) = x*z

## Division ##
##----------##

/{N,T}(z::GradTup{N,T}, x::Real) = GradTup(value(z)/x, div_tuple_by_scalar(partials(z), x))

function /{N,T}(x::Real, z::GradTup{N,T})
    z_r = value(z)
    return GradTup(x/z_r, div_real_by_dus(-x, partials(z), z_r^2))
end

function /{N,A,B}(z::GradTup{N,A}, w::GradTup{N,B})
    z_r, w_r = value(z), value(w)    
    dus = div_dus(partials(z), partials(w), z_r, w_r, w_r^2)
    return GradTup(z_r/w_r, dus)
end

## Exponentiation ##
##----------------##

for f in (:^, :(NaNMath.pow))

    @eval function ($f){N,A,B}(z::GradTup{N,A}, w::GradTup{N,B})
        z_r, w_r = value(z), value(w)    
        re = $f(z_r, w_r)
        powval = w_r * (($f)(z_r, w_r-1))
        logval = ($f)(z_r, w_r) * log(z_r)
        dus = mul_dus(partials(z), partials(w), logval, powval)
        return GradTup(re, dus)
    end

    @eval ($f){N,T}(::Base.MathConst{:e}, z::GradTup{N,T}) = exp(z)

    # generate redundant definitions to resolve ambiguity warnings
    for R in (:Integer, :Rational, :Real)
        @eval function ($f){N,T}(z::GradTup{N,T}, x::($R))
            z_r = value(z)
            powval = x*($f)(z_r, x-1)
            return GradTup(($f)(z_r, x), scale_tuple(powval, partials(z)))
        end

        @eval function ($f){N,T}(x::($R), z::GradTup{N,T})
            z_r = value(z)
            logval = ($f)(x, z_r)*log(x)
            return GradTup(($f)(x, z_r), scale_tuple(logval, partials(z)))
        end
    end

end

## from Calculus.jl ##
##------------------##

for (funsym, exp) in Calculus.symbolic_derivatives_1arg()
    funsym == :exp && continue
    
    @eval function $(funsym){N,T}(z::GradTup{N,T})
        x = value(z) # `x` is the variable name for $exp
        df = $exp
        return GradTup($(funsym)(x), scale_tuple(df, partials(z)))
    end

    # extend corresponding NaNMath methods
    if funsym in (:sin, :cos, :tan, 
                  :asin, :acos, :acosh, 
                  :atanh, :log, :log2, 
                  :log10, :lgamma, :log1p)

        funsym = Expr(:.,:NaNMath,Base.Meta.quot(funsym))

        @eval function $(funsym){N,T}(z::GradTup{N,T})
            x = value(z) # `x` is the variable name for $exp
            df = $(to_nanmath(exp))
            return GradTup($(funsym)(x), scale_tuple(df, partials(z)))
        end

    end
end

function exp{N,T}(z::GradTup{N,T})
    df = exp(value(z))
    return GradTup(df, scale_tuple(df, partials(z)))
end
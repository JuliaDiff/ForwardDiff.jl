immutable HessianNumber{N,T,C} <: ForwardDiffNumber{N,T,C}
    gradnum::GradientNumber{N,T,C} 
    hess::Vector{T}
    function HessianNumber(gradnum, hess)
        @assert length(hess) == halfhesslen(N)
        return new(gradnum, hess)
    end
end

function HessianNumber{N,T,C}(gradnum::GradientNumber{N,T,C},
                              hess::Vector=zeros(T, halfhesslen(N)))
    return HessianNumber{N,T,C}(gradnum, hess)
end

HessianNumber(value::Real) = HessianNumber(GradientNumber(value))

##############################
# Utility/Accessor Functions #
##############################
zero{N,T,C}(::Type{HessianNumber{N,T,C}}) = HessianNumber(zero(GradientNumber{N,T,C}))
one{N,T,C}(::Type{HessianNumber{N,T,C}}) = HessianNumber(one(GradientNumber{N,T,C}))
rand{N,T,C}(::Type{HessianNumber{N,T,C}}) = HessianNumber(rand(GradientNumber{N,T,C}), rand(T, halfhesslen(N)))

@inline gradnum(h::HessianNumber) = h.gradnum

@inline value(h::HessianNumber) = value(gradnum(h))
@inline grad(h::HessianNumber) = grad(gradnum(h))
@inline hess(h::HessianNumber) = h.hess
@inline tens(h::HessianNumber) = error("HessianNumbers do not store tensor values")

@inline npartials{N,T,C}(::Type{HessianNumber{N,T,C}}) = N
@inline eltype{N,T,C}(::Type{HessianNumber{N,T,C}}) = T

#####################
# Generic Functions #
#####################
function isconstant(h::HessianNumber)
    zeroT = zero(eltype(h))
    return isconstant(gradnum(h)) && all(x -> x == zeroT, hess(h))
end

=={N}(a::HessianNumber{N}, b::HessianNumber{N}) = (gradnum(a) == gradnum(b)) && (hess(a) == hess(b))

isequal{N}(a::HessianNumber{N}, b::HessianNumber{N}) = isequal(gradnum(a), gradnum(b)) && isequal(hess(a),hess(b))

hash(h::HessianNumber) = isconstant(h) ? hash(value(h)) : hash(gradnum(h), hash(hess(h)))
hash(h::HessianNumber, hsh::Uint64) = hash(hash(h), hsh)

function read{N,T,C}(io::IO, ::Type{HessianNumber{N,T,C}})
    gradnum = read(io, GradientNumber{N,T,C})
    hess = [read(io, T) for i in 1:halfhesslen(N)]
    return HessianNumber{N,T,C}(gradnum, hess)
end

function write(io::IO, h::HessianNumber)
    write(io, gradnum(h))
    for du in hess(h)
        write(io, du)
    end
end

########################
# Conversion/Promotion #
########################
convert{N,T,C}(::Type{HessianNumber{N,T,C}}, h::HessianNumber{N,T,C}) = h
convert{N,T,C}(::Type{HessianNumber{N,T,C}}, x::Real) = HessianNumber(GradientNumber{N,T,C}(x))

function convert{N,T,C}(::Type{HessianNumber{N,T,C}}, h::HessianNumber{N})
    return HessianNumber(convert(GradientNumber{N,T,C}, gradnum(h)), hess(h))
end

function convert{T<:Real}(::Type{T}, h::HessianNumber)
    if isconstant(h)
        return convert(T, value(h))
    else
        throw(InexactError)
    end
end

promote_rule{N,T<:Number,C}(::Type{HessianNumber{N,T,C}}, ::Type{T}) = HessianNumber{N,T,C}

function promote_rule{N,T,C,S<:Number}(::Type{HessianNumber{N,T,C}}, ::Type{S})
    R = promote_type(T, S)
    return HessianNumber{N,R,switch_eltype(C, R)}
end

function promote_rule{N,T1,C1,T2,C2}(::Type{HessianNumber{N,T1,C1}}, ::Type{HessianNumber{N,T2,C2}})
    R = promote_type(T1, T2)
    return HessianNumber{N,R,switch_eltype(C1, R)}
end

##########################
# Math on HessianNumbers #
##########################
# Math on HessianNumbers is developed by examining hyperdual numbers
# with 2 different infinitesmal parts (ϵ₁, ϵ₂). These numbers
# can be formulated like the following:
#
#   h = h₀ + h₁ϵ₁ + h₂ϵ₂ + h₃ϵ₁ϵ₂
# 
# where the h-components are real numbers, and the infinitesmal 
# ϵ-components are defined as:
#
#   ϵ₁ != ϵ₂ != 0
#   ϵ₁² = ϵ₂² = (ϵ₁ϵ₂)² = 0
#
# Taylor series expansion of a unary function `f` on a 
# HessianNumber `h`:
#
#   f(h) = f(h₀ + h₁ϵ₁ + h₂ϵ₂ + h₃ϵ₁ϵ₂) 
#        = f(h₀) + f'(h₀)*(h₁ϵ₁ + h₂ϵ₂ + h₃ϵ₁ϵ₂) + f''(h₀)*h₁h₂ϵ₂ϵ₁
#
# The coefficients of ϵ₁ϵ₂ are what's stored by HessianNumber's `hess` field:
#
#   f(h)_ϵ₁ϵ₂ = (f'(h₀)*h₃ + f''(h₀)*h₁h₂)
#
# where, in loop code:
#
#   h₀ = value(h)
#   h₁ = grad(h, i)
#   h₂ = grad(h, j)
#   h₃ = hess(h, q)
#
# see http://adl.stanford.edu/hyperdual/Fike_AIAA-2011-886.pdf for details.

function loadhess_deriv!{N}(h::HessianNumber{N}, deriv1, deriv2, output)
    q = 1
    for i in 1:N
        for j in 1:i
            output[q] = deriv1*hess(h, q) + deriv2*grad(h, i)*grad(h, j)
            q += 1
        end
    end
    return output
end

# Binary functions on HessianNumbers #
#------------------------------------#
function loadhess_mul!{N}(a::HessianNumber{N}, b::HessianNumber{N}, output)
    aval = value(a)
    bval = value(b)
    q = 1
    for i in 1:N
        for j in 1:i
            output[q] = (hess(a,q)*bval
                         + grad(a,i)*grad(b,j)
                         + grad(a,j)*grad(b,i)
                         + aval*hess(b,q))
            q += 1
        end
    end
    return output
end

function loadhess_div!{N}(a::HessianNumber{N}, b::HessianNumber{N}, output)
    aval = value(a)
    two_aval = aval + aval
    bval = value(b)
    bval_sq = bval * bval
    inv_bval_cb = inv(bval_sq * bval)
    q = 1
    for i in 1:N
        for j in 1:i
            g_bi, g_bj = grad(b, i), grad(b, j)
            term1 = two_aval*g_bj*g_bi + bval_sq*hess(a,q)
            term2 = grad(a,i)*g_bj + grad(a,j)*g_bi + aval*hess(b,q)
            output[q] = (term1 - bval*term2) * inv_bval_cb
            q += 1
        end
    end
    return output
end

function loadhess_div!{N}(x::Real, h::HessianNumber{N}, output)
    hval = value(h)
    hval_sq = hval * hval
    inv_hval_sq = inv(hval_sq) * x
    inv_hval_cb = inv(hval_sq * hval)
    two_inv_hval_cb = (inv_hval_cb + inv_hval_cb) * x
    return loadhess_deriv!(h, -inv_hval_sq, two_inv_hval_cb, output)
end

function loadhess_exp!{N}(a::HessianNumber{N}, b::HessianNumber{N}, output)
    aval = value(a)
    bval = value(b)
    aval_exp_bval = aval^(bval-2)
    bval_sq = bval * bval
    log_aval = log(aval)
    log_bval = log(bval)
    aval_x_logaval = aval * log_aval
    aval_x_logaval_x_logaval = aval_x_logaval * log_aval
    q = 1
    for i in 1:N
        for j in 1:i
            g_ai, g_aj = grad(a, i), grad(a, j)
            g_bi, g_bj = grad(b, i), grad(b, j)
            output[q] = (aval_exp_bval*(
                              bval_sq*g_ai*g_aj
                            + bval*(
                                  g_aj*(
                                      aval_x_logaval*g_bi
                                    - g_ai)
                                + aval*(
                                      log_aval*g_ai*g_bj
                                    + hess(a,q)))
                            + aval*(
                                  g_aj*g_bi
                                + aval_x_logaval*hess(b,q)
                                + g_bj*(
                                      g_ai
                                    + aval_x_logaval_x_logaval*g_bi))))
            q += 1
        end
    end
    return output
end

function loadhess_exp!{N}(h::HessianNumber{N}, x::Real, output)
    hval = value(h)
    x_min_one = x - 1
    deriv1 = x * hval^x_min_one
    deriv2 = x * x_min_one * hval^(x - 2)
    return loadhess_deriv!(h, deriv1, deriv2, output)
end

function loadhess_exp!{N}(x::Real, h::HessianNumber{N}, output)
    log_x = log(x)
    deriv1 = x^value(h) * log_x
    deriv2 = deriv1 * log_x
    return loadhess_deriv!(h, deriv1, deriv2, output)
end

for (fsym, loadfsym) in [(:*, symbol("loadhess_mul!")),
                         (:/, symbol("loadhess_div!")), 
                         (:^, symbol("loadhess_exp!"))]
    @eval function $(fsym){N,A,B}(a::HessianNumber{N,A}, b::HessianNumber{N,B})
        new_hess = Array(promote_type(A, B), halfhesslen(N))
        return HessianNumber($(fsym)(gradnum(a), gradnum(b)), $(loadfsym)(a, b, new_hess))
    end
end

function /{N,T}(x::Real, h::HessianNumber{N,T})
    new_hess = Array(promote_type(T, typeof(x)), halfhesslen(N))
    return HessianNumber(x / gradnum(h), loadhess_div!(x, h, new_hess))
end

^{N}(::Base.Irrational{:e}, h::HessianNumber{N}) = exp(h)

for T in (:Rational, :Integer, :Real)
    @eval begin
        function ^{N}(h::HessianNumber{N}, x::$(T))
            new_hess = Array(promote_type(eltype(h), typeof(x)), halfhesslen(N))
            return HessianNumber(gradnum(h)^x, loadhess_exp!(h, x, new_hess))
        end

        function ^{N}(x::$(T), h::HessianNumber{N})
            new_hess = Array(promote_type(eltype(h), typeof(x)), halfhesslen(N))
            return HessianNumber(x^gradnum(h), loadhess_exp!(x, h, new_hess))
        end
    end
end

+{N}(a::HessianNumber{N}, b::HessianNumber{N}) = HessianNumber(gradnum(a) + gradnum(b), hess(a) + hess(b))
-{N}(a::HessianNumber{N}, b::HessianNumber{N}) = HessianNumber(gradnum(a) - gradnum(b), hess(a) - hess(b))

for T in (:Bool, :Real)
    @eval begin
        *(h::HessianNumber, x::$(T)) = HessianNumber(gradnum(h) * x, hess(h) * x)
        *(x::$(T), h::HessianNumber) = HessianNumber(x * gradnum(h), x * hess(h))
    end
end

/(h::HessianNumber, x::Real) = HessianNumber(gradnum(h) / x, hess(h) / x)

# Unary functions on HessianNumbers #
#-----------------------------------#
-(h::HessianNumber) = HessianNumber(-gradnum(h), -hess(h))

# the second derivatives of functions in unsupported_univar_hess_funcs involves differentiating 
# elementary functions that are unsupported by Calculus.jl
const unsupported_univar_hess_funcs = [:asec, :acsc, :asecd, :acscd, :acsch, :trigamma]
const univar_hess_funcs = filter!(sym -> !in(sym, unsupported_univar_hess_funcs), fad_supported_univar_funcs)

for fsym in univar_hess_funcs
    hval = :hval
    new_val = :($(fsym)($hval))
    deriv1 = Calculus.differentiate(new_val, hval)
    deriv2 = Calculus.differentiate(deriv1, hval)

    @eval function $(fsym){N}(h::HessianNumber{N})
        hval, hg = value(h), gradnum(h)

        new_val = $new_val
        deriv1 = $deriv1
        deriv2 = $deriv2

        G = promote_typeof(hg, deriv1, deriv2)
        new_g = G(new_val, deriv1*partials(hg))

        new_hessvec = Array(eltype(new_g), halfhesslen(N))
        loadhess_deriv!(h, deriv1, deriv2, new_hessvec)
        return HessianNumber(new_g, new_hessvec)
    end
end

# Special Cases #
#---------------#
@inline calc_atan2(y::HessianNumber, x::HessianNumber) = calc_atan2(gradnum(y), gradnum(x))
@inline calc_atan2(y::Real, x::HessianNumber) = calc_atan2(y, gradnum(x))
@inline calc_atan2(y::HessianNumber, x::Real) = calc_atan2(gradnum(y), x)

for Y in (:Real, :HessianNumber), X in (:Real, :HessianNumber)
    if !(Y == :Real && X == :Real)
        @eval begin
            @inline function atan2(y::$Y, x::$X)
                z = y/x
                zval, zg = value(z), gradnum(z)
                N, T = npartials(z), eltype(z)

                deriv1 = inv(one(zval) + zval^2)
                deriv2 = 2 * zval * -abs2(deriv1)

                new_g = typeof(zg)(calc_atan2(y, x), deriv1*partials(zg))

                new_hessvec = Array(T, halfhesslen(N))
                loadhess_deriv!(z, deriv1, deriv2, new_hessvec)

                return HessianNumber(new_g, new_hessvec)
            end
        end
    end
end

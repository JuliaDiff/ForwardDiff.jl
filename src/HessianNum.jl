immutable HessianNum{N,T,C} <: ForwardDiffNum{N,T,C}
    gradnum::GradientNum{N,T,C} 
    hess::Vector{T}
    function HessianNum(gradnum, hess)
        @assert length(hess) == halfhesslen(N)
        return new(gradnum, hess)
    end
end

function HessianNum{N,T,C}(gradnum::GradientNum{N,T,C},
                           hess::Vector=zeros(T, halfhesslen(N)))
    return HessianNum{N,T,C}(gradnum, hess)
end

HessianNum(value::Real) = HessianNum(GradientNum(value))

##############################
# Utility/Accessor Functions #
##############################
zero{N,T,C}(::Type{HessianNum{N,T,C}}) = HessianNum(zero(GradientNum{N,T,C}))
one{N,T,C}(::Type{HessianNum{N,T,C}}) = HessianNum(one(GradientNum{N,T,C}))
rand{N,T,C}(::Type{HessianNum{N,T,C}}) = HessianNum(rand(GradientNum{N,T,C}), rand(T, halfhesslen(N)))

gradnum(h::HessianNum) = h.gradnum

value(h::HessianNum) = value(gradnum(h))
grad(h::HessianNum) = grad(gradnum(h))
hess(h::HessianNum) = h.hess
tens(h::HessianNum) = error("HessianNums do not store tensor values")

npartials{N,T,C}(::Type{HessianNum{N,T,C}}) = N
eltype{N,T,C}(::Type{HessianNum{N,T,C}}) = T

#####################
# Generic Functions #
#####################
function isconstant(h::HessianNum)
    zeroT = zero(eltype(h))
    return isconstant(gradnum(h)) && all(x -> x == zeroT, hess(h))
end

isconstant(h::HessianNum{0}) = true

=={N}(a::HessianNum{N}, b::HessianNum{N}) = (gradnum(a) == gradnum(b)) && (hess(a) == hess(b))

isequal{N}(a::HessianNum{N}, b::HessianNum{N}) = isequal(gradnum(a), gradnum(b)) && isequal(hess(a),hess(b))

hash(h::HessianNum) = isconstant(h) ? hash(value(h)) : hash(gradnum(h), hash(hess(h)))
hash(h::HessianNum, hsh::Uint64) = hash(hash(h), hsh)

function read{N,T,C}(io::IO, ::Type{HessianNum{N,T,C}})
    gradnum = read(io, GradientNum{N,T,C})
    hess = [read(io, T) for i in 1:halfhesslen(N)]
    return HessianNum{N,T,C}(gradnum, hess)
end

function write(io::IO, h::HessianNum)
    write(io, gradnum(h))
    for du in hess(h)
        write(io, du)
    end
end

########################
# Conversion/Promotion #
########################
convert{N,T,C}(::Type{HessianNum{N,T,C}}, h::HessianNum{N,T,C}) = h
convert{N,T,C}(::Type{HessianNum{N,T,C}}, x::Real) = HessianNum(GradientNum{N,T,C}(x))

function convert{N,T,C}(::Type{HessianNum{N,T,C}}, h::HessianNum{N})
    return HessianNum(convert(GradientNum{N,T,C}, gradnum(h)), hess(h))
end

function convert{T<:Real}(::Type{T}, h::HessianNum)
    if isconstant(h)
        return convert(T, value(h))
    else
        throw(InexactError)
    end
end

promote_rule{N,T,C}(::Type{HessianNum{N,T,C}}, ::Type{T}) = HessianNum{N,T,C}

function promote_rule{N,T,C,S}(::Type{HessianNum{N,T,C}}, ::Type{S})
    R = promote_type(T, S)
    return HessianNum{N,R,switch_eltype(C, R)}
end

function promote_rule{N,T1,C1,T2,C2}(::Type{HessianNum{N,T1,C1}}, ::Type{HessianNum{N,T2,C2}})
    R = promote_type(T1, T2)
    return HessianNum{N,R,switch_eltype(C1, R)}
end

#######################
# Math on HessianNums #
#######################
# Math on HessianNums is developed by examining hyperdual numbers
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
# Taylor series expansion of a univariate function `f` on a 
# HessianNum `h`:
#
#   f(h) = f(h₀ + h₁ϵ₁ + h₂ϵ₂ + h₃ϵ₁ϵ₂) 
#        = f(h₀) + f'(h₀)*(h₁ϵ₁ + h₂ϵ₂ + h₃ϵ₁ϵ₂) + f''(h₀)*h₁h₂ϵ₂ϵ₁
#
# The coefficients of ϵ₁ϵ₂ are what's stored by HessianNum's `hess` field:
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

function loadhess_deriv!{N}(h::HessianNum{N}, deriv1, deriv2, output)
    q = 1
    for i in 1:N
        for j in 1:i
            output[q] = deriv1*hess(h, q) + deriv2*grad(h, i)*grad(h, j)
            q += 1
        end
    end
    return output
end

# Bivariate functions on HessianNums #
#------------------------------------#
function loadhess_mul!{N}(a::HessianNum{N}, b::HessianNum{N}, output)
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

function loadhess_div!{N}(a::HessianNum{N}, b::HessianNum{N}, output)
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

function loadhess_div!{N}(x::Real, h::HessianNum{N}, output)
    hval = value(h)
    hval_sq = hval * hval
    inv_hval_sq = inv(hval_sq) * x
    inv_hval_cb = inv(hval_sq * hval)
    two_inv_hval_cb = (inv_hval_cb + inv_hval_cb) * x
    return loadhess_deriv!(h, -inv_hval_sq, two_inv_hval_cb, output)
end

function loadhess_exp!{N}(a::HessianNum{N}, b::HessianNum{N}, output)
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

function loadhess_exp!{N}(h::HessianNum{N}, p::Real, output)
    hval = value(h)
    p_coeff = p*hval^(p - 2)
    deriv1 = p_coeff * hval
    deriv2 = p_coeff * (p - 1)
    return loadhess_deriv!(h, deriv1, deriv2, output)
end

for (fsym, loadfsym) in [(:*, symbol("loadhess_mul!")),
                         (:/, symbol("loadhess_div!")), 
                         (:^, symbol("loadhess_exp!"))]
    @eval function $(fsym){N,A,B}(a::HessianNum{N,A}, b::HessianNum{N,B})
        new_hess = Array(promote_type(A, B), halfhesslen(N))
        return HessianNum($(fsym)(gradnum(a), gradnum(b)), $(loadfsym)(a, b, new_hess))
    end
end

function /{N,T}(x::Real, h::HessianNum{N,T})
    new_hess = Array(promote_type(T, typeof(x)), halfhesslen(N))
    return HessianNum(x / gradnum(h), loadhess_div!(x, h, new_hess))
end

for T in (:Rational, :Integer, :Real)
    @eval begin
        function ^{N}(h::HessianNum{N}, p::$(T))
            new_hess = Array(promote_type(eltype(h), typeof(p)), halfhesslen(N))
            return HessianNum(gradnum(h)^p, loadhess_exp!(h, p, new_hess))
        end
    end
end

+{N}(a::HessianNum{N}, b::HessianNum{N}) = HessianNum(gradnum(a) + gradnum(b), hess(a) + hess(b))
-{N}(a::HessianNum{N}, b::HessianNum{N}) = HessianNum(gradnum(a) - gradnum(b), hess(a) - hess(b))

for T in (:Bool, :Real)
    @eval begin
        *(h::HessianNum, x::$(T)) = HessianNum(gradnum(h) * x, hess(h) * x)
        *(x::$(T), h::HessianNum) = HessianNum(x * gradnum(h), x * hess(h))
    end
end

/(h::HessianNum, x::Real) = HessianNum(gradnum(h) / x, hess(h) / x)

# Univariate functions on HessianNums #
#-------------------------------------#
-(h::HessianNum) = HessianNum(-gradnum(h), -hess(h))

# the second derivatives of functions in unsupported_univar_hess_funcs involves differentiating 
# elementary functions that are unsupported by Calculus.jl
const unsupported_univar_hess_funcs = [:asec, :acsc, :asecd, :acscd, :acsch, :trigamma]
const univar_hess_funcs = filter!(sym -> !in(sym, unsupported_univar_hess_funcs), fad_supported_univar_funcs)

for fsym in univar_hess_funcs

    loadfsym = symbol(string("loadhess_", fsym, "!"))

    hval = :hval
    call_expr = :($(fsym)($hval))
    deriv1 = Calculus.differentiate(call_expr, hval)
    deriv2 = Calculus.differentiate(deriv1, hval)

    @eval function $(loadfsym){N}(h::HessianNum{N}, output)
        hval = value(h)
        return loadhess_deriv!(h, $deriv1, $deriv2, output)
    end

    expr = parse(""" 
        @generated function $(fsym){N,T}(h::HessianNum{N,T})
            ResultType = typeof($(fsym)(one(T)))
            return quote 
                new_hess = Array(\$ResultType, halfhesslen(N))
                return HessianNum($(fsym)(gradnum(h)), $(loadfsym)(h, new_hess))
            end
        end
    """)

    @eval $expr
end

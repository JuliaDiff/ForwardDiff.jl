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

# Bivariate functions on HessianNums #
#------------------------------------#
function loadhess_mul!{N}(a::HessianNum{N}, b::HessianNum{N}, output)
    aval = value(a)
    bval = value(b)
    k = 1
    for i in 1:N
        for j in 1:i
            output[k] = (hess(a,k)*bval
                         + grad(a,i)*grad(b,j)
                         + grad(a,j)*grad(b,i)
                         + aval*hess(b,k))
            k += 1
        end
    end
    return output
end

function loadhess_div!{N}(a::HessianNum{N}, b::HessianNum{N}, output)
    aval = value(a)
    two_aval = aval + aval
    bval = value(b)
    bval_sq = bval * bval
    bval_cb = bval_sq * bval
    k = 1
    for i in 1:N
        for j in 1:i
            grad_bi = grad(b, i)
            grad_bj = grad(b, j)
            term1 = two_aval*grad_bj*grad_bi + bval_sq*hess(a,k)
            term2 = grad(a,i)*grad_bj + grad(a,j)*grad_bi + aval*hess(b,k)
            output[k] = (term1 - bval*term2) / bval_cb
            k += 1
        end
    end
    return output
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
    k = 1
    for i in 1:N
        for j in 1:i
            grad_ai = grad(a, i)
            grad_aj = grad(a, j)
            grad_bi = grad(b, i)
            grad_bj = grad(b, j)
            output[k] = (aval_exp_bval*(
                              bval_sq*grad_ai*grad_aj
                            + bval*(
                                  grad_aj*(
                                      aval_x_logaval*grad_bi
                                    - grad_ai)
                                + aval*(
                                      log_aval*grad_ai*grad_bj
                                    + hess(a,k)))
                            + aval*(
                                  grad_aj*grad_bi
                                + aval_x_logaval*hess(b,k)
                                + grad_bj*(
                                      grad_ai
                                    + aval_x_logaval_x_logaval*grad_bi))))
            k += 1
        end
    end
    return output
end

for (fsym, loadfsym) in [(:*, symbol("loadhess_mul!")),
                         (:/, symbol("loadhess_div!")), 
                         (:^, symbol("loadhess_exp!"))]
    @eval function $(fsym){N,A,B}(a::HessianNum{N,A}, b::HessianNum{N,B})
        new_hess = Array(promote_type(A, B), halfhesslen(N))
        return HessianNum($(fsym)(gradnum(a), gradnum(b)), $(loadfsym)(a, b, new_hess))
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

function loadhess_div_real!{N}(x::Real, h::HessianNum{N}, output)
    hval = value(h)
    hval_sq = hval * hval
    hval_cb = hval_sq * hval
    k = 1
    for i in 1:N
        for j in 1:i
            output[k] = x * ((2*grad(h,i)*grad(h,j)/hval_cb) - (hess(h,k)/hval_sq))
            k += 1
        end
    end
    return output
end

function /{N,T}(x::Real, h::HessianNum{N,T})
    new_hess = Array(promote_type(T, typeof(x)), halfhesslen(N))
    return HessianNum(x / gradnum(h), loadhess_div_real!(x, h, new_hess))
end

for T in (:Rational, :Integer, :Real)
    @eval begin
        function ^{N}(h::HessianNum{N}, p::$(T))
            new_hess = Array(promote_type(eltype(h), typeof(p)), halfhesslen(N))
            k = 1
            for i in 1:N
                for j in 1:i
                    new_hess[k] = p*value(h)^(p-2)*((p-1)*grad(h,i)*grad(h,j)+value(h)*hess(h,k))
                    k += 1
                end
            end
            return HessianNum(gradnum(h)^p, new_hess)
        end
    end
end

# Univariate functions on HessianNums #
#-------------------------------------#
-(h::HessianNum) = HessianNum(-gradnum(h), -hess(h))

# the second derivative of functions in unsupported_univar_hess_funcs involves differentiating 
# elementary functions that are unsupported by Calculus.jl, e.g. abs(x) and polygamma(x)
const unsupported_univar_hess_funcs = [:asec, :acsc, :asecd, :acscd, :acsch, :trigamma]
const univar_hess_funcs = filter!(sym -> !in(sym, unsupported_univar_hess_funcs), map(first, Calculus.symbolic_derivatives_1arg()))

# Univariate function construction loop
for fsym in univar_hess_funcs
    loadfsym = symbol(string("loadhess_", fsym, "!"))

    hval = :hval
    call_expr = :($(fsym)($hval))
    deriv1 = differentiate(call_expr, hval)
    deriv2 = differentiate(deriv1, hval)

    @eval function $(loadfsym){N}(h::HessianNum{N}, output)
        hval = value(h)
        deriv1 = $deriv1
        deriv2 = $deriv2
        k = 1
        for i in 1:N
            for j in 1:i
                output[k] = deriv1*hess(h, k) + deriv2*grad(h, i)*grad(h, j)
                k += 1
            end
        end
        return output
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

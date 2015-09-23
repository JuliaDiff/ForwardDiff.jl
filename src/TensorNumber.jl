immutable TensorNumber{N,T,C} <: ForwardDiffNumber{N,T,C}
    hessnum::HessianNumber{N,T,C}
    tens::Vector{T}
    function TensorNumber(hessnum, tens)
        @assert length(tens) == halftenslen(N)
        return new(hessnum, tens)
    end
end

function TensorNumber{N,T,C}(hessnum::HessianNumber{N,T,C},
                             tens::Vector=zeros(T, halftenslen(N)))
    return TensorNumber{N,T,C}(hessnum, tens)
end

TensorNumber(value::Real) = TensorNumber(HessianNumber(value))

##############################
# Utility/Accessor Functions #
##############################
zero{N,T,C}(::Type{TensorNumber{N,T,C}}) = TensorNumber(zero(HessianNumber{N,T,C}))
one{N,T,C}(::Type{TensorNumber{N,T,C}}) = TensorNumber(one(HessianNumber{N,T,C}))
rand{N,T,C}(::Type{TensorNumber{N,T,C}}) = TensorNumber(rand(HessianNumber{N,T,C}), rand(T, halftenslen(N)))

@inline hessnum(t::TensorNumber) = t.hessnum
@inline gradnum(t::TensorNumber) = gradnum(hessnum(t))

@inline value(t::TensorNumber) = value(hessnum(t))
@inline grad(t::TensorNumber) = grad(hessnum(t))
@inline hess(t::TensorNumber) = hess(hessnum(t))
@inline tens(t::TensorNumber) = t.tens

@inline npartials{N,T,C}(::Type{TensorNumber{N,T,C}}) = N
@inline eltype{N,T,C}(::Type{TensorNumber{N,T,C}}) = T

#####################
# Generic Functions #
#####################
function isconstant(t::TensorNumber)
    zeroT = zero(eltype(t))
    return isconstant(hessnum(t)) && all(x -> x == zeroT, tens(t))
end

=={N}(a::TensorNumber{N}, b::TensorNumber{N}) = (hessnum(a) == hessnum(b)) && (tens(a) == tens(b))

isequal{N}(a::TensorNumber{N}, b::TensorNumber{N}) = isequal(hessnum(a), hessnum(b)) && isequal(tens(a), tens(b))

hash(t::TensorNumber) = isconstant(t) ? hash(value(t)) : hash(hessnum(t), hash(tens(t)))
hash(t::TensorNumber, hsh::Uint64) = hash(hash(t), hsh)

function read{N,T,C}(io::IO, ::Type{TensorNumber{N,T,C}})
    hessnum = read(io, HessianNumber{N,T,C})
    tens = [read(io, T) for i in 1:halftenslen(N)]
    return TensorNumber{N,T,C}(hessnum, tens)
end

function write(io::IO, t::TensorNumber)
    write(io, hessnum(t))
    for du in tens(t)
        write(io, du)
    end
end

########################
# Conversion/Promotion #
########################
convert{N,T,C}(::Type{TensorNumber{N,T,C}}, t::TensorNumber{N,T,C}) = t
convert{N,T,C}(::Type{TensorNumber{N,T,C}}, x::Real) = TensorNumber(HessianNumber{N,T,C}(x))

function convert{N,T,C}(::Type{TensorNumber{N,T,C}}, t::TensorNumber{N})
    return TensorNumber(convert(HessianNumber{N,T,C}, hessnum(t)), tens(t))
end

function convert{T<:Real}(::Type{T}, t::TensorNumber)
    if isconstant(t)
        return convert(T, value(t))
    else
        throw(InexactError)
    end
end

promote_rule{N,T<:Number,C}(::Type{TensorNumber{N,T,C}}, ::Type{T}) = TensorNumber{N,T,C}

function promote_rule{N,T,C,S<:Number}(::Type{TensorNumber{N,T,C}}, ::Type{S})
    R = promote_type(T, S)
    return TensorNumber{N,R,switch_eltype(C, R)}
end

function promote_rule{N,T1,C1,T2,C2}(::Type{TensorNumber{N,T1,C1}}, ::Type{TensorNumber{N,T2,C2}})
    R = promote_type(T1, T2)
    return TensorNumber{N,R,switch_eltype(C1, R)}
end

#########################
# Math on TensorNumbers #
#########################
function hess_inds(i, j)
    x, y = ifelse(i < j, (j, i), (i, j))
    return div(x*(x-1), 2) + y
end

function loadtens_deriv!{N}(t::TensorNumber{N}, deriv1, deriv2, deriv3, output)
    p = 1
    for i in 1:N
        for j in i:N
            for k in i:j
                qij, qik, qjk = hess_inds(i,j), hess_inds(i,k), hess_inds(j,k)
                bi, bj, bk = grad(t,i), grad(t,j), grad(t,k)
                output[p] = deriv1*tens(t,p) + deriv2*(bk*hess(t,qij) + bj*hess(t,qik) + bi*hess(t,qjk)) + deriv3*bi*bj*bk
                p += 1
            end
        end
    end
    return output
end

@inline function tensnum_from_deriv{N}(t::TensorNumber{N}, new_a, deriv1, deriv2, deriv3)
    new_h = hessnum_from_deriv(hessnum(t), new_a, deriv1, deriv2)
    new_tensvec = Array(eltype(new_h), halftenslen(N))
    loadtens_deriv!(t, deriv1, deriv2, deriv3, new_tensvec)
    return TensorNumber(new_h, new_tensvec)
end

# Addition/Subtraction #
#----------------------#
+{N}(t1::TensorNumber{N}, t2::TensorNumber{N}) = TensorNumber(hessnum(t1) + hessnum(t2), tens(t1) + tens(t2))
-{N}(t1::TensorNumber{N}, t2::TensorNumber{N}) = TensorNumber(hessnum(t1) - hessnum(t2), tens(t1) - tens(t2))
-(t::TensorNumber) = TensorNumber(-hessnum(t), -tens(t))

# Multiplication #
#----------------#
for T in (:Bool, :Real)
    @eval begin
        *(t::TensorNumber, x::$(T)) = TensorNumber(hessnum(t) * x, tens(t) * x)
        *(x::$(T), t::TensorNumber) = TensorNumber(x * hessnum(t), x * tens(t))
    end
end

function *{N}(t1::TensorNumber{N}, t2::TensorNumber{N})
    mul_h = hessnum(t1)*hessnum(t2)
    tensvec = Array(eltype(mul_h), halftenslen(N))

    a1, a2 = value(t1), value(t2)
    p = 1
    for i in 1:N
        for j in i:N
            for k in i:j
                qij, qik, qjk = hess_inds(i,j), hess_inds(i,k), hess_inds(j,k)
                tensvec[p] = (tens(t1,p)*a2 +
                              hess(t1,qjk)*grad(t2,i) +
                              hess(t1,qik)*grad(t2,j) +
                              hess(t1,qij)*grad(t2,k) +
                              grad(t1,k)*hess(t2,qij) +
                              grad(t1,j)*hess(t2,qik) +
                              grad(t1,i)*hess(t2,qjk) +
                              a1*tens(t2,p))
                p += 1
            end
        end
    end

    return TensorNumber(mul_h, tensvec)
end

# Division #
#----------#
/(t::TensorNumber, x::Real) = TensorNumber(hessnum(t) / x, tens(t) / x)

function /(x::Real, t::TensorNumber)
    a = value(t)
    div_a = x / a
    div_a_sq = div_a / a
    div_a_cb = div_a_sq / a

    deriv1 = -div_a_sq
    deriv2 = div_a_cb + div_a_cb
    deriv3 = -(deriv2 + deriv2 + deriv2)/a

    return tensnum_from_deriv(t, div_a, deriv1, deriv2, deriv3)
end

function /{N}(t1::TensorNumber{N}, t2::TensorNumber{N})
    div_h = hessnum(t1)/hessnum(t2)
    tensvec = Array(eltype(div_h), halftenslen(N))

    a1, a2 = value(t1), value(t2)
    inv_a2 = inv(a2)
    abs2_inv_a2 = abs2(inv_a2)

    coeff0 = inv_a2
    coeff1 = -abs2_inv_a2
    coeff2 = 2*abs2_inv_a2*inv_a2
    coeff3 = -2*abs2_inv_a2*(2*inv_a2*inv_a2 + abs2_inv_a2)

    p = 1
    for i in 1:N
        for j in i:N
            for k in i:j
                qij, qik, qjk = hess_inds(i,j), hess_inds(i,k), hess_inds(j,k)
                t1_bi, t1_bj, t1_bk = grad(t1,i), grad(t1,j), grad(t1,k)
                t2_bi, t2_bj, t2_bk = grad(t2,i), grad(t2,j), grad(t2,k)
                t1_cqij, t1_cqik, t1_cqjk = hess(t1,qij), hess(t1,qik), hess(t1,qjk)
                t2_cqij, t2_cqik, t2_cqjk = hess(t2,qij), hess(t2,qik), hess(t2,qjk)
                loop_coeff1 = (tens(t2,p)*a1 + t2_cqjk*t1_bi + t2_cqik*t1_bj + t2_cqij*t1_bk
                               + t2_bk*t1_cqij + t2_bj*t1_cqik + t2_bi*t1_cqjk)
                loop_coeff2 = (t2_bk*t2_cqij*a1 + t2_bj*t2_cqik*a1 + t2_bi*t2_cqjk*a1
                               + t2_bj*t2_bk*t1_bi + t2_bi*t2_bk*t1_bj + t2_bi*t2_bj*t1_bk)
                loop_coeff3 = (t2_bi*t2_bj*t2_bk*a1)
                tensvec[p] = coeff0*tens(t1,p) + coeff1*loop_coeff1 + coeff2*loop_coeff2 + coeff3*loop_coeff3
                p += 1
            end
        end
    end

    return TensorNumber(div_h, tensvec)
end

# Exponentiation #
#----------------#
^(::Base.Irrational{:e}, t::TensorNumber) = exp(t)

for T in (:Rational, :Integer, :Real)
    @eval begin
        function ^(t::TensorNumber, x::$(T))
            a = value(t)
            x_min_one = x - 1
            x_min_two = x - 2
            x_x_min_one = x * x_min_one

            exp_a = a^x
            deriv1 = x * a^x_min_one
            deriv2 = x_x_min_one * a^x_min_two
            deriv3 = x_x_min_one * x_min_two * a^(x - 3)

            return tensnum_from_deriv(t, exp_a, deriv1, deriv2, deriv3)
        end

        function ^(x::$(T), t::TensorNumber)
            log_x = log(x)

            exp_x = x^value(t)
            deriv1 = exp_x * log_x
            deriv2 = deriv1 * log_x
            deriv3 = deriv2 * log_x

            return tensnum_from_deriv(t, exp_x, deriv1, deriv2, deriv3)
        end
    end
end

function ^{N}(t1::TensorNumber{N}, t2::TensorNumber{N})
    exp_h = hessnum(t1)^hessnum(t2)
    tensvec = Array(eltype(exp_h), halftenslen(N))

    a1, a2 = value(t1), value(t2)
    inv_a1 = inv(a1)
    abs2_inv_a1 = abs2(inv_a1)

    f_0 = log(a1)
    f_1 = inv_a1
    f_2 = -abs2_inv_a1
    f_3 = 2*abs2_inv_a1*inv_a1

    deriv = a1^a2

    p = 1
    for i in 1:N
        for j in i:N
            for k in i:j
                qij, qik, qjk = hess_inds(i,j), hess_inds(i,k), hess_inds(j,k)

                t1_bi, t1_bj, t1_bk = grad(t1,i), grad(t1,j), grad(t1,k)
                t2_bi, t2_bj, t2_bk = grad(t2,i), grad(t2,j), grad(t2,k)
                t1_cqij, t1_cqik, t1_cqjk = hess(t1,qij), hess(t1,qik), hess(t1,qjk)
                t2_cqij, t2_cqik, t2_cqjk = hess(t2,qij), hess(t2,qik), hess(t2,qjk)

                d_1 = t1_bi*f_1
                d_2 = t1_bj*f_1
                d_3 = t1_bk*f_1
                d_4 = t1_cqij*f_1 + t1_bi*t1_bj*f_2
                d_5 = t1_cqik*f_1 + t1_bi*t1_bk*f_2
                d_6 = t1_cqjk*f_1 + t1_bj*t1_bk*f_2
                d_7 = (tens(t1,p)*f_1 + (t1_bk*t1_cqij + t1_bj*t1_cqik
                       + t1_bi*t1_cqjk)*f_2 + t1_bi*t1_bj*t1_bk*f_3)

                e_1 = t2_bi*f_0 + a2*d_1
                e_2 = t2_bj*f_0 + a2*d_2
                e_3 = t2_bk*f_0 + a2*d_3
                e_4 = t2_cqij*f_0 + t2_bj*d_1 + t2_bi*d_2 + a2*d_4
                e_5 = t2_cqik*f_0 + t2_bk*d_1 + t2_bi*d_3 + a2*d_5
                e_6 = t2_cqjk*f_0 + t2_bk*d_2 + t2_bj*d_3 + a2*d_6
                e_7 = (tens(t2,p)*f_0 + t2_cqjk*d_1 + t2_cqik*d_2 + t2_cqij*d_3
                       + t2_bk*d_4 + t2_bj*d_5 + t2_bi*d_6 + a2*d_7)

                tensvec[p] = deriv*(e_7 + e_3*e_4 + e_2*e_5 + e_1*e_6 + e_1*e_2*e_3)
                p += 1
            end
        end
    end

    return TensorNumber(exp_h, tensvec)
end

# Unary functions on TensorNumbers #
#----------------------------------#
# the third derivatives of functions in unsupported_unary_tens_funcs
# involve differentiating elementary functions that are unsupported
# by Calculus.jl
const unsupported_unary_tens_funcs = [:digamma]
const auto_defined_unary_tens_funcs = filter!(sym -> !in(sym, unsupported_unary_tens_funcs), ForwardDiff.auto_defined_unary_hess_funcs)

for fsym in auto_defined_unary_tens_funcs
    a = :a
    new_a = :($(fsym)($a))
    deriv1 = Calculus.differentiate(new_a, a)
    deriv2 = Calculus.differentiate(deriv1, a)
    deriv3 = Calculus.differentiate(deriv2, a)

    @eval function $(fsym){N}(t::TensorNumber{N})
        a = value(t)
        new_a = $new_a
        deriv1 = $deriv1
        deriv2 = $deriv2
        deriv3 = $deriv3
        return tensnum_from_deriv(t, new_a, deriv1, deriv2, deriv3)
    end
end

#################
# Special Cases #
#################

# Manually Optimized Functions #
#------------------------------#
@inline function exp(t::TensorNumber)
    exp_a = exp(value(t))
    return tensnum_from_deriv(t, exp_a, exp_a, exp_a, exp_a)
end

function sqrt(t::TensorNumber)
    sqrt_a = sqrt(value(t))
    deriv1 = 0.5 / sqrt_a
    sqrt_a_cb = a * sqrt_a
    deriv2 = -0.25 / sqrt_a_cb
    deriv3 = 0.375 / (a * sqrt_a_cb)
    return tensnum_from_deriv(t, sqrt_a, deriv1, deriv2, deriv3)
end

# Other Functions #
#-----------------#
@inline calc_atan2(y::TensorNumber, x::TensorNumber) = calc_atan2(hessnum(y), hessnum(x))
@inline calc_atan2(y::Real, x::TensorNumber) = calc_atan2(y, hessnum(x))
@inline calc_atan2(y::TensorNumber, x::Real) = calc_atan2(hessnum(y), x)

for Y in (:Real, :TensorNumber), X in (:Real, :TensorNumber)
    if !(Y == :Real && X == :Real)
        @eval begin
            function atan2(y::$Y, x::$X)
                z = y/x
                a = value(z)
                atan2_a = calc_atan2(y, x)
                deriv1 = inv(one(a) + a*a)
                abs2_deriv1 = -2 * abs2(deriv1)
                deriv2 = a * abs2_deriv1
                deriv3 = abs2_deriv1 - (4 * a * deriv1 * deriv2)
                return tensnum_from_deriv(z, atan2_a, deriv1, deriv2, deriv3)
            end
        end
    end
end

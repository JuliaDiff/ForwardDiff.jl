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

# Binary functions on TensorNumbers #
#-----------------------------------#
function loadtens_mul!{N}(t1::TensorNumber{N}, t2::TensorNumber{N}, output)
    t1_a = value(t1)
    t2_a = value(t2)
    p = 1
    for i in 1:N
        for j in i:N
            for k in i:j
                qij, qik, qjk = hess_inds(i,j), hess_inds(i,k), hess_inds(j,k)
                output[p] = (tens(t1,p)*t2_a +
                             hess(t1,qjk)*grad(t2,i) +
                             hess(t1,qik)*grad(t2,j) +
                             hess(t1,qij)*grad(t2,k) +
                             grad(t1,k)*hess(t2,qij) +
                             grad(t1,j)*hess(t2,qik) +
                             grad(t1,i)*hess(t2,qjk) +
                             t1_a*tens(t2,p))
                p += 1
            end
        end
    end
    return output
end

function loadtens_div!{N}(t1::TensorNumber{N}, t2::TensorNumber{N}, output)
    t1_a = value(t1)
    t2_a = value(t2)
    inv_t2_a = inv(t2_a)
    abs2_inv_t2_a = abs2(inv_t2_a)

    coeff0 = inv_t2_a
    coeff1 = -abs2_inv_t2_a
    coeff2 = 2*abs2_inv_t2_a*inv_t2_a
    coeff3 = -2*abs2_inv_t2_a*(2*inv_t2_a*inv_t2_a + abs2_inv_t2_a)

    p = 1
    for i in 1:N
        for j in i:N
            for k in i:j
                qij, qik, qjk = hess_inds(i,j), hess_inds(i,k), hess_inds(j,k)
                t1_bi, t1_bj, t1_bk = grad(t1,i), grad(t1,j), grad(t1,k)
                t2_bi, t2_bj, t2_bk = grad(t2,i), grad(t2,j), grad(t2,k)
                t1_cqij, t1_cqik, t1_cqjk = hess(t1,qij), hess(t1,qik), hess(t1,qjk)
                t2_cqij, t2_cqik, t2_cqjk = hess(t2,qij), hess(t2,qik), hess(t2,qjk)
                loop_coeff1 = (tens(t2,p)*t1_a + t2_cqjk*t1_bi + t2_cqik*t1_bj + t2_cqij*t1_bk + t2_bk*t1_cqij + t2_bj*t1_cqik + t2_bi*t1_cqjk)
                loop_coeff2 = (t2_bk*t2_cqij*t1_a + t2_bj*t2_cqik*t1_a + t2_bi*t2_cqjk*t1_a + t2_bj*t2_bk*t1_bi + t2_bi*t2_bk*t1_bj + t2_bi*t2_bj*t1_bk)
                loop_coeff3 = (t2_bi*t2_bj*t2_bk*t1_a)
                output[p] = coeff0*tens(t1,p) + coeff1*loop_coeff1 + coeff2*loop_coeff2 + coeff3*loop_coeff3
                p += 1
            end
        end
    end
    return output
end

function loadtens_div!{N}(x::Real, t::TensorNumber{N}, output)
    t_a = value(t)
    inv_t_a = inv(t_a)
    abs2_inv_t_a = abs2(inv_t_a)

    inv_deriv1 = -x*abs2_inv_t_a
    inv_deriv2 = 2*x*abs2_inv_t_a*inv_t_a
    inv_deriv3 = -2*x*abs2_inv_t_a*(2*inv_t_a*inv_t_a + abs2_inv_t_a)

    return loadtens_deriv!(t, inv_deriv1, inv_deriv2, inv_deriv3, output)
end

function loadtens_exp!{N}(t1::TensorNumber{N}, t2::TensorNumber{N}, output)
    t1_a = value(t1)
    t2_a = value(t2)
    
    inv_t1_a = inv(t1_a)
    abs2_inv_t1_a = abs2(inv_t1_a)

    f_0 = log(t1_a)
    f_1 = inv_t1_a
    f_2 = -abs2_inv_t1_a
    f_3 = 2*abs2_inv_t1_a*inv_t1_a

    deriv = t1_a^t2_a

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
                d_7 = tens(t1,p)*f_1 + (t1_bk*t1_cqij + t1_bj*t1_cqik + t1_bi*t1_cqjk)*f_2 + t1_bi*t1_bj*t1_bk*f_3

                e_1 = t2_bi*f_0 + t2_a*d_1
                e_2 = t2_bj*f_0 + t2_a*d_2
                e_3 = t2_bk*f_0 + t2_a*d_3
                e_4 = t2_cqij*f_0 + t2_bj*d_1 + t2_bi*d_2 + t2_a*d_4
                e_5 = t2_cqik*f_0 + t2_bk*d_1 + t2_bi*d_3 + t2_a*d_5
                e_6 = t2_cqjk*f_0 + t2_bk*d_2 + t2_bj*d_3 + t2_a*d_6
                e_7 = tens(t2,p)*f_0 + t2_cqjk*d_1 + t2_cqik*d_2 + t2_cqij*d_3 + t2_bk*d_4 + t2_bj*d_5 + t2_bi*d_6 + t2_a*d_7

                output[p] = deriv*(e_7 + e_3*e_4 + e_2*e_5 + e_1*e_6 + e_1*e_2*e_3)
                p += 1
            end
        end
    end
    return output
end

function loadtens_exp!{N}(t::TensorNumber{N}, x::Real, output)
    t_a = value(t)
    x_min_one = x - 1
    x_min_two = x - 2
    deriv1 = x * t_a^x_min_one
    deriv2 = x * x_min_one * t_a^x_min_two
    deriv3 = x * x_min_one * x_min_two * t_a^(x - 3)
    return loadtens_deriv!(t, deriv1, deriv2, deriv3, output)
end

function loadtens_exp!{N}(x::Real, t::TensorNumber{N}, output)
    log_x = log(x)
    deriv1 = x^value(t) * log_x
    deriv2 = deriv1 * log_x
    deriv3 = deriv2 * log_x
    return loadtens_deriv!(t, deriv1, deriv2, deriv3, output)
end

for (fsym, loadfsym) in [(:*, symbol("loadtens_mul!")),
                         (:/, symbol("loadtens_div!")), 
                         (:^, symbol("loadtens_exp!"))]
    @eval function $(fsym){N,A,B}(t1::TensorNumber{N,A}, t2::TensorNumber{N,B})
        new_tens = Array(promote_type(A, B), halftenslen(N))
        return TensorNumber($(fsym)(hessnum(t1), hessnum(t2)), $(loadfsym)(t1, t2, new_tens))
    end
end

^{N}(::Base.Irrational{:e}, t::TensorNumber{N}) = exp(t)

for T in (:Rational, :Integer, :Real)
    @eval begin
        function ^{N}(t::TensorNumber{N}, x::$(T))
            new_tens = Array(promote_type(eltype(t), typeof(x)), halftenslen(N))
            return TensorNumber(hessnum(t)^x, loadtens_exp!(t, x, new_tens))
        end

        function ^{N}(x::$(T), t::TensorNumber{N})
            new_tens = Array(promote_type(eltype(t), typeof(x)), halftenslen(N))
            return TensorNumber(x^hessnum(t), loadtens_exp!(x, t, new_tens))
        end
    end
end

function /{N,T}(x::Real, t::TensorNumber{N,T})
    new_tens = Array(promote_type(T, typeof(x)), halftenslen(N))
    return TensorNumber(x / hessnum(t), loadtens_div!(x, t, new_tens))
end

+{N}(t1::TensorNumber{N}, t2::TensorNumber{N}) = TensorNumber(hessnum(t1) + hessnum(t2), tens(t1) + tens(t2))
-{N}(t1::TensorNumber{N}, t2::TensorNumber{N}) = TensorNumber(hessnum(t1) - hessnum(t2), tens(t1) - tens(t2))

for T in (:Bool, :Real)
    @eval begin
        *(t::TensorNumber, x::$(T)) = TensorNumber(hessnum(t) * x, tens(t) * x)
        *(x::$(T), t::TensorNumber) = TensorNumber(x * hessnum(t), x * tens(t))
    end
end

/(t::TensorNumber, x::Real) = TensorNumber(hessnum(t) / x, tens(t) / x)

# Unary functions on TensorNumbers #
#----------------------------------#
-(t::TensorNumber) = TensorNumber(-hessnum(t), -tens(t))

# the third derivatives of functions in unsupported_univar_tens_funcs involves differentiating 
# elementary functions that are unsupported by Calculus.jl
const unsupported_univar_tens_funcs = [:digamma]
const univar_tens_funcs = filter!(sym -> !in(sym, unsupported_univar_tens_funcs), ForwardDiff.univar_hess_funcs)

for fsym in univar_tens_funcs
    t_a = :t_a
    new_a = :($(fsym)($t_a))
    deriv1 = Calculus.differentiate(new_a, t_a)
    deriv2 = Calculus.differentiate(deriv1, t_a)
    deriv3 = Calculus.differentiate(deriv2, t_a)

    @eval function $(fsym){N}(t::TensorNumber{N})
        t_a, t_g, t_h = value(t), gradnum(t), hessnum(t)

        new_a = $new_a
        deriv1 = $deriv1
        deriv2 = $deriv2
        deriv3 = $deriv3

        G = promote_typeof(t_g, deriv1, deriv2, deriv3)
        T = eltype(G)
        new_g = G(new_a, deriv1*partials(t_g))

        new_hessvec = Array(T, halfhesslen(N))
        loadhess_deriv!(t_h, deriv1, deriv2, new_hessvec)
        new_h = HessianNumber(new_g, new_hessvec)

        new_tensvec = Array(T, halftenslen(N))
        loadtens_deriv!(t, deriv1, deriv2, deriv3, new_tensvec)
        return TensorNumber(new_h, new_tensvec)
    end
end

# Special Cases #
#---------------#
@inline calc_atan2(y::TensorNumber, x::TensorNumber) = calc_atan2(hessnum(y), hessnum(x))
@inline calc_atan2(y::Real, x::TensorNumber) = calc_atan2(y, hessnum(x))
@inline calc_atan2(y::TensorNumber, x::Real) = calc_atan2(hessnum(y), x)

for Y in (:Real, :TensorNumber), X in (:Real, :TensorNumber)
    if !(Y == :Real && X == :Real)
        @eval begin
            function atan2(y::$Y, x::$X)
                z = y/x
                z_a, z_g, z_h = value(z), gradnum(z), hessnum(z)
                N, T = npartials(z), eltype(z)
                
                deriv1 = inv(one(z_a) + z_a^2)
                abs2_deriv1 = -2 * abs2(deriv1)
                deriv2 = z_a * abs2_deriv1
                deriv3 = abs2_deriv1 - (4 * z_a * deriv1 * deriv2)
                
                new_g = typeof(z_g)(calc_atan2(y, x), deriv1*partials(z_g))
                
                new_hessvec = Array(T, halfhesslen(N))
                loadhess_deriv!(z_h, deriv1, deriv2, new_hessvec)
                new_h = HessianNumber(new_g, new_hessvec)
                
                new_tensvec = Array(T, halftenslen(N))
                loadtens_deriv!(z, deriv1, deriv2, deriv3, new_tensvec)
                return TensorNumber(new_h, new_tensvec)
            end
        end
    end
end
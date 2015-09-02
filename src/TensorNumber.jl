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
# Math on TensorNumbers is developed by examining hyperdual numbers
# with 3 different infinitesmal parts (ϵ₁, ϵ₂, ϵ₃). These numbers
# can be formulated like the following:
#
#   t = t₀ + t₁ϵ₁ + t₂ϵ₂ + t₃ϵ₃ + t₄ϵ₁ϵ₂ + t₅ϵ₁ϵ₃ + t₆ϵ₂ϵ₃ + t₇ϵ₁ϵ₂ϵ₃
#
# where the t-components are real numbers, and the infinitesmal
# ϵ-components are defined as:
#
#   ϵ₁ != ϵ₂ != ϵ₃ != 0
#   ϵ₁² = ϵ₂² = ϵ₃² = (ϵ₁ϵ₂)² = (ϵ₁ϵ₃)² = (ϵ₂ϵ₃)² = (ϵ₁ϵ₂ϵ₃)² = 0
#
# Taylor series expansion of a univariate function `f` on a
# TensorNumber `t`:
#
#   f(t) = f(t₀ + t₁ϵ₁ + t₂ϵ₂ + t₃ϵ₃ + t₄ϵ₁ϵ₂ + t₅ϵ₁ϵ₃ + t₆ϵ₂ϵ₃ + t₇ϵ₁ϵ₂ϵ₃)
#        = f(t₀) +
#          f'(t₀)   * (t₁ϵ₁ + t₂ϵ₂ + t₃ϵ₃ + t₄ϵ₁ϵ₂ + t₅ϵ₁ϵ₃ + t₆ϵ₂ϵ₃ + t₇ϵ₁ϵ₂ϵ₃) +
#          f''(t₀)  * (t₁t₂ϵ₂ϵ₁ + t₁t₃ϵ₃ϵ₁ + t₃t₄ϵ₂ϵ₃ϵ₁ + t₂t₅ϵ₂ϵ₃ϵ₁ + t₁t₆ϵ₂ϵ₃ϵ₁ + t₂t₃ϵ₂ϵ₃) +
#          f'''(t₀) * (t₁t₂t₃ϵ₂ϵ₃ϵ₁)
#
# The coefficients of ϵ₁ϵ₂ϵ₃ are what's stored by TensorNumber's `tens` field:
#
#   f(t)_ϵ₁ϵ₂ϵ₃ = (f'(t₀)*t₇ + f''(t₀)*(t₃t₄ + t₂t₅ + t₁t₆) + f'''(t₀)*t₁t₂t₃
#
# where, in the loop code below:
#
#   t₀ = value(h)
#   t₁ = grad(t, i) # coeff of ϵ₁
#   t₂ = grad(t, j) # coeff of ϵ₂
#   t₃ = grad(t, k) # coeff of ϵ₃
#   t₄ = hess(t, a) = hess(t, hess_inds(i, j)) # coeff of ϵ₁ϵ₂
#   t₅ = hess(t, b) = hess(t, hess_inds(i, k)) # coeff of ϵ₁ϵ₃
#   t₆ = hess(t, c) = hess(t, hess_inds(j, k)) # coeff of ϵ₂ϵ₃
#   t₇ = tens(t, q) # coeff of ϵ₁ϵ₂ϵ₃
#
# see http://adl.stanford.edu/hyperdual/Fike_AIAA-2011-886.pdf for details.

function hess_inds(i, j)
    if i < j
        return div(j*(j-1), 2) + i
    else
        return div(i*(i-1), 2) + j
    end
end

function loadtens_deriv!{N}(t::TensorNumber{N}, deriv1, deriv2, deriv3, output)
    q = 1
    for i in 1:N
        for j in i:N
            for k in i:j
                a, b, c = hess_inds(i,j), hess_inds(i,k), hess_inds(j,k)
                g_i, g_j, g_k = grad(t,i), grad(t,j), grad(t,k)
                output[q] = deriv1*tens(t,q) + deriv2*(g_k*hess(t,a) + g_j*hess(t,b) + g_i*hess(t,c)) + deriv3*g_i*g_j*g_k
                q += 1
            end
        end
    end
    return output
end

# Bivariate functions on TensorNumbers #
#--------------------------------------#
function loadtens_mul!{N}(t1::TensorNumber{N}, t2::TensorNumber{N}, output)
    t1val = value(t1)
    t2val = value(t2)
    q = 1
    for i in 1:N
        for j in i:N
            for k in i:j
                a, b, c = hess_inds(i,j), hess_inds(i,k), hess_inds(j,k)
                output[q] = (tens(t1,q)*t2val +
                             hess(t1,c)*grad(t2,i) +
                             hess(t1,b)*grad(t2,j) +
                             hess(t1,a)*grad(t2,k) +
                             grad(t1,k)*hess(t2,a) +
                             grad(t1,j)*hess(t2,b) +
                             grad(t1,i)*hess(t2,c) +
                             t1val*tens(t2,q))
                q += 1
            end
        end
    end
    return output
end

function loadtens_div!{N}(t1::TensorNumber{N}, t2::TensorNumber{N}, output)
    t1val = value(t1)
    t2val = value(t2)
    inv_t2val = inv(t2val)
    abs2_inv_t2val = abs2(inv_t2val)

    coeff0 = inv_t2val
    coeff1 = -abs2_inv_t2val
    coeff2 = 2*abs2_inv_t2val*inv_t2val
    coeff3 = -2*abs2_inv_t2val*(2*inv_t2val*inv_t2val + abs2_inv_t2val)

    q = 1
    for i in 1:N
        for j in i:N
            for k in i:j
                a, b, c = hess_inds(i,j), hess_inds(i,k), hess_inds(j,k)
                t1_gi, t1_gj, t1_gk = grad(t1,i), grad(t1,j), grad(t1,k)
                t2_gi, t2_gj, t2_gk = grad(t2,i), grad(t2,j), grad(t2,k)
                t1_ha, t1_hb, t1_hc = hess(t1,a), hess(t1,b), hess(t1,c)
                t2_ha, t2_hb, t2_hc = hess(t2,a), hess(t2,b), hess(t2,c)
                loop_coeff1 = (tens(t2,q)*t1val + t2_hc*t1_gi + t2_hb*t1_gj + t2_ha*t1_gk + t2_gk*t1_ha + t2_gj*t1_hb + t2_gi*t1_hc)
                loop_coeff2 = (t2_gk*t2_ha*t1val + t2_gj*t2_hb*t1val + t2_gi*t2_hc*t1val + t2_gj*t2_gk*t1_gi + t2_gi*t2_gk*t1_gj + t2_gi*t2_gj*t1_gk)
                loop_coeff3 = (t2_gi*t2_gj*t2_gk*t1val)
                output[q] = coeff0*tens(t1,q) + coeff1*loop_coeff1 + coeff2*loop_coeff2 + coeff3*loop_coeff3
                q += 1
            end
        end
    end
    return output
end

function loadtens_div!{N}(x::Real, t::TensorNumber{N}, output)
    tval = value(t)
    inv_tval = inv(tval)
    abs2_inv_tval = abs2(inv_tval)

    inv_deriv1 = -x*abs2_inv_tval
    inv_deriv2 = 2*x*abs2_inv_tval*inv_tval
    inv_deriv3 = -2*x*abs2_inv_tval*(2*inv_tval*inv_tval + abs2_inv_tval)

    return loadtens_deriv!(t, inv_deriv1, inv_deriv2, inv_deriv3, output)
end

function loadtens_exp!{N}(t1::TensorNumber{N}, t2::TensorNumber{N}, output)
    t1val = value(t1)
    t2val = value(t2)
    
    inv_t1val = inv(t1val)
    abs2_inv_t1val = abs2(inv_t1val)

    f_0 = log(t1val)
    f_1 = inv_t1val
    f_2 = -abs2_inv_t1val
    f_3 = 2*abs2_inv_t1val*inv_t1val

    deriv = t1val^t2val

    q = 1
    for i in 1:N
        for j in i:N
            for k in i:j
                a, b, c = hess_inds(i,j), hess_inds(i,k), hess_inds(j,k)
                
                t1_gi, t1_gj, t1_gk = grad(t1,i), grad(t1,j), grad(t1,k)
                t2_gi, t2_gj, t2_gk = grad(t2,i), grad(t2,j), grad(t2,k)
                t1_ha, t1_hb, t1_hc = hess(t1,a), hess(t1,b), hess(t1,c)
                t2_ha, t2_hb, t2_hc = hess(t2,a), hess(t2,b), hess(t2,c)

                d_1 = t1_gi*f_1
                d_2 = t1_gj*f_1
                d_3 = t1_gk*f_1
                d_4 = t1_ha*f_1 + t1_gi*t1_gj*f_2
                d_5 = t1_hb*f_1 + t1_gi*t1_gk*f_2
                d_6 = t1_hc*f_1 + t1_gj*t1_gk*f_2
                d_7 = tens(t1,q)*f_1 + (t1_gk*t1_ha + t1_gj*t1_hb + t1_gi*t1_hc)*f_2 + t1_gi*t1_gj*t1_gk*f_3

                e_1 = t2_gi*f_0 + t2val*d_1
                e_2 = t2_gj*f_0 + t2val*d_2
                e_3 = t2_gk*f_0 + t2val*d_3
                e_4 = t2_ha*f_0 + t2_gj*d_1 + t2_gi*d_2 + t2val*d_4
                e_5 = t2_hb*f_0 + t2_gk*d_1 + t2_gi*d_3 + t2val*d_5
                e_6 = t2_hc*f_0 + t2_gk*d_2 + t2_gj*d_3 + t2val*d_6
                e_7 = tens(t2,q)*f_0 + t2_hc*d_1 + t2_hb*d_2 + t2_ha*d_3 + t2_gk*d_4 + t2_gj*d_5 + t2_gi*d_6 + t2val*d_7

                output[q] = deriv*(e_7 + e_3*e_4 + e_2*e_5 + e_1*e_6 + e_1*e_2*e_3)
                q += 1
            end
        end
    end
    return output
end

function loadtens_exp!{N}(t::TensorNumber{N}, x::Real, output)
    tval = value(t)
    x_min_one = x - 1
    x_min_two = x - 2
    deriv1 = x * tval^x_min_one
    deriv2 = x * x_min_one * tval^x_min_two
    deriv3 = x * x_min_one * x_min_two * tval^(x - 3)
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
    @eval function $(fsym){N,A,B}(a::TensorNumber{N,A}, b::TensorNumber{N,B})
        new_tens = Array(promote_type(A, B), halftenslen(N))
        return TensorNumber($(fsym)(hessnum(a), hessnum(b)), $(loadfsym)(a, b, new_tens))
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

+{N}(a::TensorNumber{N}, b::TensorNumber{N}) = TensorNumber(hessnum(a) + hessnum(b), tens(a) + tens(b))
-{N}(a::TensorNumber{N}, b::TensorNumber{N}) = TensorNumber(hessnum(a) - hessnum(b), tens(a) - tens(b))

for T in (:Bool, :Real)
    @eval begin
        *(t::TensorNumber, x::$(T)) = TensorNumber(hessnum(t) * x, tens(t) * x)
        *(x::$(T), t::TensorNumber) = TensorNumber(x * hessnum(t), x * tens(t))
    end
end

/(t::TensorNumber, x::Real) = TensorNumber(hessnum(t) / x, tens(t) / x)

# Univariate functions on TensorNumbers #
#---------------------------------------#
-(t::TensorNumber) = TensorNumber(-hessnum(t), -tens(t))

# the third derivatives of functions in unsupported_univar_tens_funcs involves differentiating 
# elementary functions that are unsupported by Calculus.jl
const unsupported_univar_tens_funcs = [:digamma]
const univar_tens_funcs = filter!(sym -> !in(sym, unsupported_univar_tens_funcs), ForwardDiff.univar_hess_funcs)

for fsym in univar_tens_funcs
    loadfsym = symbol(string("loadtens_", fsym, "!"))

    tval = :tval
    call_expr = :($(fsym)($tval))
    deriv1 = Calculus.differentiate(call_expr, tval)
    deriv2 = Calculus.differentiate(deriv1, tval)
    deriv3 = Calculus.differentiate(deriv2, tval)

    @eval function $(loadfsym){N}(t::TensorNumber{N}, output)
        tval = value(t)
        deriv1 = $deriv1
        deriv2 = $deriv2
        deriv3 = $deriv3
        return loadtens_deriv!(t, deriv1, deriv2, deriv3, output)
    end

    expr = parse(""" 
        @generated function $(fsym){N,T}(t::TensorNumber{N,T})
            ResultType = typeof($(fsym)(one(T)))
            return quote 
                new_tens = Array(\$ResultType, halftenslen(N))
                return TensorNumber($(fsym)(hessnum(t)), $(loadfsym)(t, new_tens))
            end
        end
    """)

    @eval $expr
end

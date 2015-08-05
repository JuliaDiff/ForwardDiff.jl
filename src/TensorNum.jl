immutable TensorNum{N,T,C} <: ForwardDiffNum{N,T,C}
    hessnum::HessianNum{N,T,C}
    tens::Vector{T}
    function TensorNum(hessnum, tens)
        @assert length(tens) == halftenslen(N)
        return new(hessnum, tens)
    end
end

function TensorNum{N,T,C}(hessnum::HessianNum{N,T,C},
                          tens::Vector=zeros(T, halftenslen(N)))
    return TensorNum{N,T,C}(hessnum, tens)
end

TensorNum(value::Real) = TensorNum(HessianNum(value))

##############################
# Utility/Accessor Functions #
##############################
zero{N,T,C}(::Type{TensorNum{N,T,C}}) = TensorNum(zero(HessianNum{N,T,C}))
one{N,T,C}(::Type{TensorNum{N,T,C}}) = TensorNum(one(HessianNum{N,T,C}))
rand{N,T,C}(::Type{TensorNum{N,T,C}}) = TensorNum(rand(HessianNum{N,T,C}), rand(T, halftenslen(N)))

hessnum(t::TensorNum) = t.hessnum

value(t::TensorNum) = value(hessnum(t))
grad(t::TensorNum) = grad(hessnum(t))
hess(t::TensorNum) = hess(hessnum(t))
tens(t::TensorNum) = t.tens

npartials{N,T,C}(::Type{TensorNum{N,T,C}}) = N
eltype{N,T,C}(::Type{TensorNum{N,T,C}}) = T

#####################
# Generic Functions #
#####################
function isconstant(t::TensorNum)
    zeroT = zero(eltype(t))
    return isconstant(hessnum(t)) && all(x -> x == zeroT, tens(t))
end

isconstant(t::TensorNum{0}) = true

=={N}(a::TensorNum{N}, b::TensorNum{N}) = (hessnum(a) == hessnum(b)) && (tens(a) == tens(b))

isequal{N}(a::TensorNum{N}, b::TensorNum{N}) = isequal(hessnum(a), hessnum(b)) && isequal(tens(a), tens(b))

hash(t::TensorNum) = isconstant(t) ? hash(value(t)) : hash(hessnum(t), hash(tens(t)))
hash(t::TensorNum, hsh::Uint64) = hash(hash(t), hsh)

function read{N,T,C}(io::IO, ::Type{TensorNum{N,T,C}})
    hessnum = read(io, HessianNum{N,T,C})
    tens = [read(io, T) for i in 1:halftenslen(N)]
    return TensorNum{N,T,C}(hessnum, tens)
end

function write(io::IO, t::TensorNum)
    write(io, hessnum(t))
    for du in tens(t)
        write(io, du)
    end
end

########################
# Conversion/Promotion #
########################
convert{N,T,C}(::Type{TensorNum{N,T,C}}, t::TensorNum{N,T,C}) = t
convert{N,T,C}(::Type{TensorNum{N,T,C}}, x::Real) = TensorNum(HessianNum{N,T,C}(x))

function convert{N,T,C}(::Type{TensorNum{N,T,C}}, t::TensorNum{N})
    return TensorNum(convert(HessianNum{N,T,C}, hessnum(t)), tens(t))
end

function convert{T<:Real}(::Type{T}, t::TensorNum)
    if isconstant(t)
        return convert(T, value(t))
    else
        throw(InexactError)
    end
end

promote_rule{N,T,C}(::Type{TensorNum{N,T,C}}, ::Type{T}) = TensorNum{N,T,C}

function promote_rule{N,T,C,S}(::Type{TensorNum{N,T,C}}, ::Type{S})
    R = promote_type(T, S)
    return TensorNum{N,R,switch_eltype(C, R)}
end

function promote_rule{N,T1,C1,T2,C2}(::Type{TensorNum{N,T1,C1}}, ::Type{TensorNum{N,T2,C2}})
    R = promote_type(T1, T2)
    return TensorNum{N,R,switch_eltype(C1, R)}
end

######################
# Math on TensorNums #
######################
# Math on TensorNums is developed by examining hyperdual numbers
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
# TensorNum `t`:
#
#   f(t) = f(t₀ + t₁ϵ₁ + t₂ϵ₂ + t₃ϵ₃ + t₄ϵ₁ϵ₂ + t₅ϵ₁ϵ₃ + t₆ϵ₂ϵ₃ + t₇ϵ₁ϵ₂ϵ₃)
#        = f(t₀) +
#          f'(t₀)   * (t₁ϵ₁ + t₂ϵ₂ + t₃ϵ₃ + t₄ϵ₁ϵ₂ + t₅ϵ₁ϵ₃ + t₆ϵ₂ϵ₃ + t₇ϵ₁ϵ₂ϵ₃) +
#          f''(t₀)  * (t₁t₂ϵ₂ϵ₁ + t₁t₃ϵ₃ϵ₁ + t₃t₄ϵ₂ϵ₃ϵ₁ + t₂t₅ϵ₂ϵ₃ϵ₁ + t₁t₆ϵ₂ϵ₃ϵ₁ + t₂t₃ϵ₂ϵ₃) +
#          f'''(t₀) * (t₁t₂t₃ϵ₂ϵ₃ϵ₁)
#
# The coefficients of ϵ₁ϵ₂ϵ₃ are what's stored by TensorNum's `tens` field:
#
#   f(t)_ϵ₁ϵ₂ϵ₃ = (f'(t₀)*t₇ + f''(t₀)*(t₃t₄ + t₂t₅ + t₁t₆) + f'''(t₀)*t₁t₂t₃
#
# where, in the loop code below:
#
#   t₀ = value(h)
#   t₁ = grad(t, i) # coeff of ϵ₁
#   t₂ = grad(t, j) # coeff of ϵ₂
#   t₃ = grad(t, k) # coeff of ϵ₃
#   t₄ = hess(t, a) = hess(t, t_inds_2_h_ind(i, j)) # coeff of ϵ₁ϵ₂
#   t₅ = hess(t, b) = hess(t, t_inds_2_h_ind(i, k)) # coeff of ϵ₁ϵ₃
#   t₆ = hess(t, c) = hess(t, t_inds_2_h_ind(j, k)) # coeff of ϵ₂ϵ₃
#   t₇ = tens(t, q) # coeff of ϵ₁ϵ₂ϵ₃
#
# see http://adl.stanford.edu/hyperdual/Fike_AIAA-2011-886.pdf for details.

function t_inds_2_h_ind(i, j)
    if i < j
        return div(j*(j-1), 2+i) + 1
    else
        return div(i*(i-1), 2+j) + 1
    end
end

function loadtens_deriv!{N}(t::TensorNum{N}, deriv1, deriv2, deriv3, output)
    q = 1
    for i in 1:N
        for j in i:N
            for k in i:j
                a, b, c = t_inds_2_h_ind(i,j), t_inds_2_h_ind(i,k), t_inds_2_h_ind(j,k)
                g_i, g_j, g_k = grad(t,i), grad(t,j), grad(t,k)
                output[q] = deriv1*tens(t,q) + deriv2*(g_k*hess(t,a) + g_j*hess(t,b) + g_i*hess(t,c)) + deriv3*g_i*g_j*g_k
                q += 1
            end
        end
    end
    return output
end

# Bivariate functions on TensorNums #
#-----------------------------------#
function loadtens_mul!{N}(t1::TensorNum{N}, t2::TensorNum{N}, output)
    t1val = value(t1)
    t2val = value(t2)
    q = 1
    for i in 1:N
        for j in i:N
            for k in i:j
                a, b, c = t_inds_2_h_ind(i,j), t_inds_2_h_ind(i,k), t_inds_2_h_ind(j,k)
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

function loadtens_div!{N}(t1::TensorNum{N}, t2::TensorNum{N}, output)
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
                a, b, c = t_inds_2_h_ind(i,j), t_inds_2_h_ind(i,k), t_inds_2_h_ind(j,k)
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

function loadtens_div!{N}(x::Real, t::TensorNum{N}, output)
    tval = value(t)
    inv_tval = inv(tval)
    abs2_inv_tval = abs2(inv_tval)

    inv_deriv1 = -x*abs2_inv_tval
    inv_deriv2 = 2*x*abs2_inv_tval*inv_tval
    inv_deriv3 = -2*x*abs2_inv_tval*(2*inv_tval*inv_tval + abs2_inv_tval)

    return loadtens_deriv!(t, inv_deriv1, inv_deriv2, inv_deriv3, output)
end

loadtens_exp!{N}(t1::TensorNum{N}, t2::TensorNum{N}, output) = error("loadtens_exp!(t1::TensorNum, t2::TensorNum, output) is not yet implemented.")
loadtens_exp!{N}(t::TensorNum{N}, p::Real, output) = error("loadtens_exp!(t::TensorNum, p::Real, output) is not yet implemented.")

for (fsym, loadfsym) in [(:*, symbol("loadtens_mul!")),
                         (:/, symbol("loadtens_div!")), 
                         (:^, symbol("loadtens_exp!"))]
    @eval function $(fsym){N,A,B}(a::TensorNum{N,A}, b::TensorNum{N,B})
        new_tens = Array(promote_type(A, B), halftenslen(N))
        return TensorNum($(fsym)(hessnum(a), hessnum(b)), $(loadfsym)(a, b, new_tens))
    end
end

function /{N,T}(x::Real, t::TensorNum{N,T})
    new_tens = Array(promote_type(T, typeof(x)), halftenslen(N))
    return TensorNum(x / hessnum(t), loadtens_div!(x, t, new_tens))
end

for T in (:Rational, :Integer, :Real)
    @eval begin
        function ^{N}(t::TensorNum{N}, p::$(T))
            new_tens = Array(promote_type(eltype(t), typeof(p)), halftenslen(N))
            return TensorNum(hessnum(t)^p, loadtens_exp!(t, p, new_tens))
        end
    end
end

+{N}(a::TensorNum{N}, b::TensorNum{N}) = TensorNum(hessnum(a) + hessnum(b), tens(a) + tens(b))
-{N}(a::TensorNum{N}, b::TensorNum{N}) = TensorNum(hessnum(a) - hessnum(b), tens(a) - tens(b))

for T in (:Bool, :Real)
    @eval begin
        *(t::TensorNum, x::$(T)) = TensorNum(hessnum(t) * x, tens(t) * x)
        *(x::$(T), t::TensorNum) = TensorNum(x * hessnum(t), x * tens(t))
    end
end

/(t::TensorNum, x::Real) = TensorNum(hessnum(t) / x, tens(t) / x)

# Univariate functions on TensorNums #
#------------------------------------#
-(t::TensorNum) = TensorNum(-hessnum(t), -tens(t))

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

    @eval function $(loadfsym){N}(t::TensorNum{N}, output)
        tval = value(t)
        deriv1 = $deriv1
        deriv2 = $deriv2
        deriv3 = $deriv3
        return loadtens_deriv!(t, deriv1, deriv2, deriv3, output)
    end

    expr = parse(""" 
        @generated function $(fsym){N,T}(t::TensorNum{N,T})
            ResultType = typeof($(fsym)(one(T)))
            return quote 
                new_tens = Array(\$ResultType, halftenslen(N))
                return TensorNum($(fsym)(hessnum(t)), $(loadfsym)(t, new_tens))
            end
        end
    """)

    @eval $expr
end

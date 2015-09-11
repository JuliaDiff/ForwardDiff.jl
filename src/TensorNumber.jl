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
    q = 1
    for i in 1:N
        for j in i:N
            for k in i:j
                x, y, z = hess_inds(i,j), hess_inds(i,k), hess_inds(j,k)
                g_i, g_j, g_k = grad(t,i), grad(t,j), grad(t,k)
                output[q] = deriv1*tens(t,q) + deriv2*(g_k*hess(t,x) + g_j*hess(t,y) + g_i*hess(t,z)) + deriv3*g_i*g_j*g_k
                q += 1
            end
        end
    end
    return output
end

# Binary functions on TensorNumbers #
#-----------------------------------#
function loadtens_mul!{N}(t1::TensorNumber{N}, t2::TensorNumber{N}, output)
    t1val = value(t1)
    t2val = value(t2)
    q = 1
    for i in 1:N
        for j in i:N
            for k in i:j
                x, y, z = hess_inds(i,j), hess_inds(i,k), hess_inds(j,k)
                output[q] = (tens(t1,q)*t2val +
                             hess(t1,z)*grad(t2,i) +
                             hess(t1,y)*grad(t2,j) +
                             hess(t1,x)*grad(t2,k) +
                             grad(t1,k)*hess(t2,x) +
                             grad(t1,j)*hess(t2,y) +
                             grad(t1,i)*hess(t2,z) +
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
                x, y, z = hess_inds(i,j), hess_inds(i,k), hess_inds(j,k)
                t1_gi, t1_gj, t1_gk = grad(t1,i), grad(t1,j), grad(t1,k)
                t2_gi, t2_gj, t2_gk = grad(t2,i), grad(t2,j), grad(t2,k)
                t1_hx, t1_hy, t1_hz = hess(t1,x), hess(t1,y), hess(t1,z)
                t2_hx, t2_hy, t2_hz = hess(t2,x), hess(t2,y), hess(t2,z)
                loop_coeff1 = (tens(t2,q)*t1val + t2_hz*t1_gi + t2_hy*t1_gj + t2_hx*t1_gk + t2_gk*t1_hx + t2_gj*t1_hy + t2_gi*t1_hz)
                loop_coeff2 = (t2_gk*t2_hx*t1val + t2_gj*t2_hy*t1val + t2_gi*t2_hz*t1val + t2_gj*t2_gk*t1_gi + t2_gi*t2_gk*t1_gj + t2_gi*t2_gj*t1_gk)
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
                x, y, z = hess_inds(i,j), hess_inds(i,k), hess_inds(j,k)
                
                t1_gi, t1_gj, t1_gk = grad(t1,i), grad(t1,j), grad(t1,k)
                t2_gi, t2_gj, t2_gk = grad(t2,i), grad(t2,j), grad(t2,k)
                t1_hx, t1_hy, t1_hz = hess(t1,x), hess(t1,y), hess(t1,z)
                t2_hx, t2_hy, t2_hz = hess(t2,x), hess(t2,y), hess(t2,z)

                d_1 = t1_gi*f_1
                d_2 = t1_gj*f_1
                d_3 = t1_gk*f_1
                d_4 = t1_hx*f_1 + t1_gi*t1_gj*f_2
                d_5 = t1_hy*f_1 + t1_gi*t1_gk*f_2
                d_6 = t1_hz*f_1 + t1_gj*t1_gk*f_2
                d_7 = tens(t1,q)*f_1 + (t1_gk*t1_hx + t1_gj*t1_hy + t1_gi*t1_hz)*f_2 + t1_gi*t1_gj*t1_gk*f_3

                e_1 = t2_gi*f_0 + t2val*d_1
                e_2 = t2_gj*f_0 + t2val*d_2
                e_3 = t2_gk*f_0 + t2val*d_3
                e_4 = t2_hx*f_0 + t2_gj*d_1 + t2_gi*d_2 + t2val*d_4
                e_5 = t2_hy*f_0 + t2_gk*d_1 + t2_gi*d_3 + t2val*d_5
                e_6 = t2_hz*f_0 + t2_gk*d_2 + t2_gj*d_3 + t2val*d_6
                e_7 = tens(t2,q)*f_0 + t2_hz*d_1 + t2_hy*d_2 + t2_hx*d_3 + t2_gk*d_4 + t2_gj*d_5 + t2_gi*d_6 + t2val*d_7

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

# Unary functions on TensorNumbers #
#----------------------------------#
-(t::TensorNumber) = TensorNumber(-hessnum(t), -tens(t))

# the third derivatives of functions in unsupported_univar_tens_funcs involves differentiating 
# elementary functions that are unsupported by Calculus.jl
const unsupported_univar_tens_funcs = [:digamma]
const univar_tens_funcs = filter!(sym -> !in(sym, unsupported_univar_tens_funcs), ForwardDiff.univar_hess_funcs)

for fsym in univar_tens_funcs
    tval = :tval
    new_val = :($(fsym)($tval))
    deriv1 = Calculus.differentiate(new_val, tval)
    deriv2 = Calculus.differentiate(deriv1, tval)
    deriv3 = Calculus.differentiate(deriv2, tval)

    @eval function $(fsym){N}(t::TensorNumber{N})
        tval, tg, th = value(t), gradnum(t), hessnum(t)

        new_val = $new_val
        deriv1 = $deriv1
        deriv2 = $deriv2
        deriv3 = $deriv3

        G = promote_typeof(tg, deriv1, deriv2, deriv3)
        T = eltype(G)
        new_g = G(new_val, deriv1*partials(tg))

        new_hessvec = Array(T, halfhesslen(N))
        loadhess_deriv!(th, deriv1, deriv2, new_hessvec)
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
                zval, zg, zh = value(z), gradnum(z), hessnum(z)
                N, T = npartials(z), eltype(z)
                
                deriv1 = inv(one(zval) + zval^2)
                abs2_deriv1 = -2 * abs2(deriv1)
                deriv2 = zval * abs2_deriv1
                deriv3 = abs2_deriv1 - (4 * zval * deriv1 * deriv2)
                
                new_g = typeof(zg)(calc_atan2(y, x), deriv1*partials(zg))
                
                new_hessvec = Array(T, halfhesslen(N))
                loadhess_deriv!(zh, deriv1, deriv2, new_hessvec)
                new_h = HessianNumber(new_g, new_hessvec)
                
                new_tensvec = Array(T, halftenslen(N))
                loadtens_deriv!(z, deriv1, deriv2, deriv3, new_tensvec)
                return TensorNumber(new_h, new_tensvec)
            end
        end
    end
end
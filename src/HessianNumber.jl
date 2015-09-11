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
function loadhess_mul!{N}(h1::HessianNumber{N}, h2::HessianNumber{N}, output)
    h1_a = value(h1)
    h2_a = value(h2)
    q = 1
    for i in 1:N
        for j in 1:i
            output[q] = (hess(h1,q)*h2_a
                         + grad(h1,i)*grad(h2,j)
                         + grad(h1,j)*grad(h2,i)
                         + h1_a*hess(h2,q))
            q += 1
        end
    end
    return output
end

function loadhess_div!{N}(h1::HessianNumber{N}, h2::HessianNumber{N}, output)
    h1_a = value(h1)
    two_h1_a = h1_a + h1_a
    h2_a = value(h2)
    h2_a_sq = h2_a * h2_a
    inv_h2_a_cb = inv(h2_a_sq * h2_a)
    q = 1
    for i in 1:N
        for j in 1:i
            h2_bi, h2_bj = grad(h2,i), grad(h2,j)
            term1 = two_h1_a*h2_bj*h2_bi + h2_a_sq*hess(h1,q)
            term2 = grad(h1,i)*h2_bj + grad(h1,j)*h2_bi + h1_a*hess(h2,q)
            output[q] = (term1 - h2_a*term2) * inv_h2_a_cb
            q += 1
        end
    end
    return output
end

function loadhess_div!{N}(x::Real, h::HessianNumber{N}, output)
    h_a = value(h)
    h_a_sq = h_a * h_a
    inv_h_a_sq = inv(h_a_sq) * x
    inv_h_a_cb = inv(h_a_sq * h_a)
    two_inv_h_a_cb = (inv_h_a_cb + inv_h_a_cb) * x
    return loadhess_deriv!(h, -inv_h_a_sq, two_inv_h_a_cb, output)
end

function loadhess_exp!{N}(h1::HessianNumber{N}, h2::HessianNumber{N}, output)
    h1_a = value(h1)
    h2_a = value(h2)
    h1_a_exp_h2_a = h1_a^(h2_a-2)
    h2_a_sq = h2_a * h2_a
    log_h1_a = log(h1_a)
    log_h2_a = log(h2_a)
    h1_a_x_logh1_a = h1_a * log_h1_a
    h1_a_x_logh1_a_x_logh1_a = h1_a_x_logh1_a * log_h1_a
    q = 1
    for i in 1:N
        for j in 1:i
            h1_bi, h1_bj = grad(h1, i), grad(h1, j)
            h2_bi, h2_bj = grad(h2, i), grad(h2, j)
            output[q] = (h1_a_exp_h2_a*(
                              h2_a_sq*h1_bi*h1_bj
                            + h2_a*(
                                  h1_bj*(
                                      h1_a_x_logh1_a*h2_bi
                                    - h1_bi)
                                + h1_a*(
                                      log_h1_a*h1_bi*h2_bj
                                    + hess(h1,q)))
                            + h1_a*(
                                  h1_bj*h2_bi
                                + h1_a_x_logh1_a*hess(h2,q)
                                + h2_bj*(
                                      h1_bi
                                    + h1_a_x_logh1_a_x_logh1_a*h2_bi))))
            q += 1
        end
    end
    return output
end

function loadhess_exp!{N}(h::HessianNumber{N}, x::Real, output)
    h_a = value(h)
    x_min_one = x - 1
    deriv1 = x * h_a^x_min_one
    deriv2 = x * x_min_one * h_a^(x - 2)
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
    @eval function $(fsym){N,A,B}(h1::HessianNumber{N,A}, h2::HessianNumber{N,B})
        new_hess = Array(promote_type(A, B), halfhesslen(N))
        return HessianNumber($(fsym)(gradnum(h1), gradnum(h2)), $(loadfsym)(h1, h2, new_hess))
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

+{N}(h1::HessianNumber{N}, h2::HessianNumber{N}) = HessianNumber(gradnum(h1) + gradnum(h2), hess(h1) + hess(h2))
-{N}(h1::HessianNumber{N}, h2::HessianNumber{N}) = HessianNumber(gradnum(h1) - gradnum(h2), hess(h1) - hess(h2))

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
    h_a = :h_a
    new_a = :($(fsym)($h_a))
    deriv1 = Calculus.differentiate(new_a, h_a)
    deriv2 = Calculus.differentiate(deriv1, h_a)

    @eval function $(fsym){N}(h::HessianNumber{N})
        h_a, hg = value(h), gradnum(h)

        new_a = $new_a
        deriv1 = $deriv1
        deriv2 = $deriv2

        G = promote_typeof(hg, deriv1, deriv2)
        new_g = G(new_a, deriv1*partials(hg))

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
                z_a, z_g = value(z), gradnum(z)
                N, T = npartials(z), eltype(z)

                deriv1 = inv(one(z_a) + z_a^2)
                deriv2 = 2 * z_a * -abs2(deriv1)

                new_g = typeof(z_g)(calc_atan2(y, x), deriv1*partials(z_g))

                new_hessvec = Array(T, halfhesslen(N))
                loadhess_deriv!(z, deriv1, deriv2, new_hessvec)

                return HessianNumber(new_g, new_hessvec)
            end
        end
    end
end

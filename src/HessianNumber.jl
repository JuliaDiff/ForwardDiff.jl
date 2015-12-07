################
# Constructors #
################
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
@inline containtype{N,T,C}(::Type{HessianNumber{N,T,C}}) = C

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
hash(h::HessianNumber, hsh::UInt64) = hash(hash(h), hsh)

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

##############
# Conversion #
##############
convert{N,T,C}(::Type{HessianNumber{N,T,C}}, x::ExternalReal) = HessianNumber(GradientNumber{N,T,C}(x))
convert{N,T,C}(::Type{HessianNumber{N,T,C}}, h::HessianNumber{N}) = HessianNumber(GradientNumber{N,T,C}(gradnum(h)), hess(h))
convert{N,T,C}(::Type{HessianNumber{N,T,C}}, h::HessianNumber{N,T,C}) = h
convert(::Type{HessianNumber}, h::HessianNumber) = h

####################################
# Math Functions on HessianNumbers #
####################################
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

@inline function hessnum_from_deriv{N}(h::HessianNumber{N}, new_a, deriv1, deriv2)
    new_g = gradnum_from_deriv(gradnum(h), new_a, deriv1)
    new_hessvec = Array(eltype(new_g), halfhesslen(N))
    loadhess_deriv!(h, deriv1, deriv2, new_hessvec)
    return HessianNumber(new_g, new_hessvec)
end

# Addition/Subtraction #
#----------------------#
+{N}(h1::HessianNumber{N}, h2::HessianNumber{N}) = HessianNumber(gradnum(h1) + gradnum(h2), hess(h1) + hess(h2))
-{N}(h1::HessianNumber{N}, h2::HessianNumber{N}) = HessianNumber(gradnum(h1) - gradnum(h2), hess(h1) - hess(h2))

-(h::HessianNumber) = HessianNumber(-gradnum(h), -hess(h))

# Multiplication #
#----------------#
for T in (:Bool, :Real)
    @eval begin
        *(h::HessianNumber, x::$(T)) = HessianNumber(gradnum(h) * x, hess(h) * x)
        *(x::$(T), h::HessianNumber) = HessianNumber(x * gradnum(h), x * hess(h))
    end
end

function *{N}(h1::HessianNumber{N}, h2::HessianNumber{N})
    mul_g = gradnum(h1)*gradnum(h2)
    hessvec = Array(eltype(mul_g), halfhesslen(N))

    a1, a2 = value(h1), value(h2)
    q = 1
    for i in 1:N
        for j in 1:i
            hessvec[q] = (hess(h1,q)*a2
                          + grad(h1,i)*grad(h2,j)
                          + grad(h1,j)*grad(h2,i)
                          + a1*hess(h2,q))
            q += 1
        end
    end

    return HessianNumber(mul_g, hessvec)
end

# Division #
#----------#
/(h::HessianNumber, x::Real) = HessianNumber(gradnum(h) / x, hess(h) / x)

function /(x::Real, h::HessianNumber)
    a = value(h)

    div_a = x / a
    div_a_sq = div_a / a
    div_a_cb = div_a_sq / a

    deriv1 = -div_a_sq
    deriv2 = div_a_cb + div_a_cb

    return hessnum_from_deriv(h, div_a, deriv1, deriv2)
end

function /{N}(h1::HessianNumber{N}, h2::HessianNumber{N})
    div_g = gradnum(h1)/gradnum(h2)
    hessvec = Array(eltype(div_g), halfhesslen(N))

    a1, a2 = value(h1), value(h2)
    two_a1 = a1 + a1
    a2_sq = a2 * a2
    inv_a2_cb = inv(a2_sq * a2)
    q = 1
    for i in 1:N
        for j in 1:i
            h2_bi, h2_bj = grad(h2,i), grad(h2,j)
            term1 = two_a1*h2_bj*h2_bi + a2_sq*hess(h1,q)
            term2 = grad(h1,i)*h2_bj + grad(h1,j)*h2_bi + a1*hess(h2,q)
            hessvec[q] = (term1 - a2*term2) * inv_a2_cb
            q += 1
        end
    end

    return HessianNumber(div_g, hessvec)
end

# Exponentiation #
#----------------#
^(::Base.Irrational{:e}, h::HessianNumber) = exp(h)

for T in (:Rational, :Integer, :Real)
    @eval begin
        function ^(h::HessianNumber, x::$(T))
            a = value(h)
            x_min_one = x - 1
            exp_a = a^x
            deriv1 = x * a^x_min_one
            deriv2 = x * x_min_one * a^(x - 2)
            return hessnum_from_deriv(h, exp_a, deriv1, deriv2)
        end

        function ^(x::$(T), h::HessianNumber)
            log_x = log(x)
            exp_x = x^value(h)
            deriv1 = exp_x * log_x
            deriv2 = deriv1 * log_x
            return hessnum_from_deriv(h, exp_x, deriv1, deriv2)
        end
    end
end

function ^{N}(h1::HessianNumber{N}, h2::HessianNumber{N})
    exp_g = gradnum(h1)^gradnum(h2)
    hessvec = Array(eltype(exp_g), halfhesslen(N))

    a1, a2 = value(h1), value(h2)
    a1_exp_a2 = a1^(a2 - 2)
    a2_sq = a2 * a2
    log_a1, log_a2  = log(a1), log(a2)
    a1_x_loga1 = a1 * log_a1
    a1_x_loga1_x_loga1 = a1_x_loga1 * log_a1
    q = 1
    for i in 1:N
        for j in 1:i
            h1_bi, h1_bj = grad(h1, i), grad(h1, j)
            h2_bi, h2_bj = grad(h2, i), grad(h2, j)
            h1_q, h2_q = hess(h1, q), hess(h2, q)
            term1 = (h1_bj*(a1_x_loga1*h2_bi - h1_bi)
                     + a1*(log_a1*h1_bi*h2_bj + h1_q))
            term2 = (h1_bj*h2_bi
                     + a1_x_loga1*h2_q
                     + h2_bj*(h1_bi + a1_x_loga1_x_loga1*h2_bi))
            term3 = a2_sq*h1_bi*h1_bj + a2*term1 + a1*term2
            hessvec[q] = a1_exp_a2*term3
            q += 1
        end
    end

    return HessianNumber(exp_g, hessvec)
end

# Unary functions on HessianNumbers #
#-----------------------------------#
# the second derivatives of functions in
# unsupported_unary_hess_funcs involve
# differentiating elementary functions
# that are unsupported by Calculus.jl
const unsupported_unary_hess_funcs = [:asec, :acsc, :asecd, :acscd, :acsch, :trigamma]
const auto_defined_unary_hess_funcs = filter!(sym -> !in(sym, unsupported_unary_hess_funcs), auto_defined_unary_funcs)

for fsym in auto_defined_unary_hess_funcs
    a = :a
    new_a = :($(fsym)($a))
    deriv1 = Calculus.differentiate(new_a, a)
    deriv2 = Calculus.differentiate(deriv1, a)

    @eval function $(fsym){N}(h::HessianNumber{N})
        a = value(h)
        new_a = $new_a
        deriv1 = $deriv1
        deriv2 = $deriv2
        return hessnum_from_deriv(h, new_a, deriv1, deriv2)
    end
end

#################
# Special Cases #
#################

# Manually Optimized Functions #
#------------------------------#
@inline function exp(h::HessianNumber)
    exp_a = exp(value(h))
    return hessnum_from_deriv(h, exp_a, exp_a, exp_a)
end

@inline function sqrt(h::HessianNumber)
    sqrt_a = sqrt(value(h))
    deriv1 = 0.5 / sqrt_a
    deriv2 = -0.25 / (a * sqrt_a)
    return hessnum_from_deriv(h, sqrt_a, deriv1, deriv2)
end

# Other Functions #
#-----------------#
@inline calc_atan2(y::HessianNumber, x::HessianNumber) = calc_atan2(gradnum(y), gradnum(x))
@inline calc_atan2(y::Real, x::HessianNumber) = calc_atan2(y, gradnum(x))
@inline calc_atan2(y::HessianNumber, x::Real) = calc_atan2(gradnum(y), x)

for Y in (:Real, :HessianNumber), X in (:Real, :HessianNumber)
    if !(Y == :Real && X == :Real)
        @eval begin
            function atan2(y::$Y, x::$X)
                z = y/x
                a = value(z)
                atan2_a = calc_atan2(y, x)
                deriv1 = inv(one(a) + a*a)
                deriv2 = (a + a) * -abs2(deriv1)
                return hessnum_from_deriv(z, atan2_a, deriv1, deriv2)
            end
        end
    end
end

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
# In the code-generating loops below (see "Bivariate function construction loop"
# and "Univariate function construction loop"), we build definitions for math functions
# on HessianNums in a consistent, uniform manner by utilizing the `h_bivar_funcs` and
# `h_univar_funcs` arrays. These arrays hold multiple Tuples, each of which provides the
# necessary information to define a different function. The description of these
# Tuples' formats can be found in comments above their respective arrays.

# Bivariate functions on HessianNums #
#------------------------------------#

# The Tuples in `h_bivar_funcs` have the following format:
#
# (:function_name,
#  :(expression defining the kth entry of the hessian vector, using any available variables))
const h_bivar_funcs = Tuple{Symbol, Expr}[
    (:*, :(hess(a,k)*value(b)+grad(a,i)*grad(b,j)+grad(a,j)*grad(b,i)+value(a)*hess(b,k))),
    (:/, :(((2*value(a)*grad(b,j)*grad(b,i)+(value(b)^2)*hess(a,k))-(value(b)*(grad(a,i)*grad(b,j)
           +grad(a,j)*grad(b,i)+value(a)*hess(b,k))))/(value(b)^3))),
    (:^, :((value(a)^(value(b)-2))*((value(b)^2)*grad(a,i)*grad(a,j)+value(b)*(grad(a,j)*(-grad(a,i)
           +value(a)*log(value(a))*grad(b,i))+value(a)*(log(value(a))*grad(a,i)*grad(b,j)
           +hess(a,k)))+value(a)*(grad(a,j)*grad(b,i)+grad(b,j)*(grad(a,i)+value(a)*log(value(a))
           *log(value(a))*grad(b,i))+value(a)*log(value(a))*hess(b,k)))))
]

# Bivariate function construction loop
for (fsym, term) in h_bivar_funcs
    loadfsym = symbol(string("loadhess_", fsym, "!"))
    @eval begin
        function $(loadfsym){N}(a::HessianNum{N}, b::HessianNum{N}, output)
            k = 1
            for i in 1:N
                for j in 1:i
                    output[k] = $(term)
                    k += 1
                end
            end
            return output
        end

        function $(fsym){N,A,B}(a::HessianNum{N,A}, b::HessianNum{N,B})
            new_hess = Array(promote_type(A, B), halfhesslen(N))
            return HessianNum($(fsym)(gradnum(a), gradnum(b)), $(loadfsym)(a, b, new_hess))
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
#/(x::Real, h::HessianNum) = ?

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

# The Tuples in `h_univar_funcs` have the following format:
#
# (:function_name,
#  :(expression defining the kth entry of the hessian vector, using any available variables))
const h_univar_funcs = Tuple{Symbol, Expr}[
    (:sqrt, :((-grad(h,i)*grad(h,j)+2*value(h)*hess(h,i)) / (4*(value(h)^(1.5))))),
    (:cbrt, :((-2*grad(h,i)*grad(h,j)+3*value(h)*hess(h,k)) / (9*cbrt(value(h)^5)))),
    (:exp, :(exp(value(h))*(grad(h,i)*grad(h,j)+hess(h,k)))),
    (:log, :((value(h)*hess(h,k)-grad(h,i)*grad(h,j))/(value(h)^2))),
    (:log2, :((value(h)*hess(h,k)-grad(h,i)*grad(h,j)) / ((value(h)^2)*0.6931471805599453))),
    (:log10, :((value(h)*hess(h,k)-grad(h,i)*grad(h,j)) / ((value(h)^2)*2.302585092994046))),
    (:sin, :(-sin(value(h))*grad(h,i)*grad(h,j)+cos(value(h))*hess(h,k))),
    (:cos, :(-cos(value(h))*grad(h,i)*grad(h,j)-sin(value(h))*hess(h,k))),
    (:tan, :((sec(value(h))^2)*(2*tan(value(h))*grad(h,i)*grad(h,j)+hess(h,k)))),
    (:asin, :((value(h)*grad(h,i)*grad(h,j)-((value(h)^2)-1)*new_hess[k]) / ((1-(value(h)^2))^1.5))),
    (:acos, :((-value(h)*grad(h,i)*grad(h,j)+((value(h)^2)-1)*new_hess[k]) / ((1-(value(h)^2))^1.5))),
    (:atan, :((-2*value(h)*grad(h,i)*grad(h,j)+((value(h)^2)+1)*new_hess[k]) / ((1+(value(h)^2))^2))),
    (:sinh, :(sinh(value(h))*grad(h,i)*grad(h,j)+cosh(value(h))*hess(h,k))),
    (:cosh, :(cosh(value(h))*grad(h,i)*grad(h,j)+sinh(value(h))*hess(h,k))),
    (:tanh, :((sech(value(h))^2)*(-2*tanh(value(h))*grad(h,i)*grad(h,j)+hess(h,k)))),
    (:asinh, :((-value(h)*grad(h,i)*grad(h,j)+((1+(value(h)^2))*hess(h,k))) / ((1+(value(h)^2))^1.5))),
    (:acosh, :((-value(h)*grad(h,i)*grad(h,j)+(((value(h)^2)-1)*hess(h,k))) / (((1+value(h))^1.5)*((value(h)-1)^1.5)))),
    (:atanh, :((2*value(h)*grad(h,i)*grad(h,j)-(((value(h)^2)-1)*hess(h,k)))/(((value(h)^2)-1)^2))),
    (:lgamma, :(digamma(value(h))*hess(h,k)+trigamma(value(h))*grad(h,i)*grad(h,j))),
    (:digamma, :(trigamma(value(h))*hess(h,k)+polygamma(2,value(h))*grad(h,i)*grad(h,j)))
]

# Univariate function construction loop
for (fsym, term) in h_univar_funcs
    loadfsym = symbol(string("loadhess_", fsym, "!"))
    @eval begin
        function $(loadfsym){N}(h::HessianNum{N}, output)
            k = 1
            for i in 1:N
                for j in 1:i
                    output[k] = $(term)
                    k += 1
                end
            end
            return output
        end

        function $(fsym){N,T}(h::HessianNum{N,T})
            ResultType = typeof($(fsym)(one(T)))
            new_hess = Array(ResultType, halfhesslen(N))
            return HessianNum($(fsym)(gradnum(h)), $(loadfsym)(h, new_hess))
        end
    end
end

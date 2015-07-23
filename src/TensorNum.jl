immutable TensorNum{N,T,C} <: ForwardDiffNum{N,T,C}
    hessnum::HessianNum{N,T,C}
    tens::Vector{T}
    function TensorNum(hessnum::HessianNum{N,T,C}, tens::Vector{T})
        @assert length(tens) == halftenslen(N)
        return new(hessnum, tens)
    end
end

function TensorNum{N,T,C}(hessnum::HessianNum{N,T,C},
                          tens::Vector=zeros(T, halftenslen(N)))
    return TensorNum{N,T,C}(hessnum, tens)
end

##############################
# Utility/Accessor Functions #
##############################
zero{N,T,C}(::Type{TensorNum{N,T,C}}) = TensorNum(zero(HessianNum{N,T,C}), zeros(T, halftenslen(n)))
one{N,T,C}(::Type{TensorNum{N,T,C}}) = TensorNum(one(HessianNum{N,T,C}), zeros(T, halftenslen(n)))

hessnum(t::TensorNum) = t.hessnum

value(t::TensorNum) = value(hessnum(t))
grad(t::TensorNum) = grad(hessnum(t))
hess(t::TensorNum) = hess(hessnum(t))
tens(t::TensorNum) = t.tens

npartials{N,T,C}(::Type{TensorNum{N,T,C}}) = N
eltype{N,T,C}(::Type{TensorNum{N,T,C}}) = T

function isconstant(t::TensorNum)
    zeroT = zero(eltype(t))
    return isconstant(hess(t)) && all(x -> x == zeroT, tens(h))
end

=={N}(a::TensorNum{N}, b::TensorNum{N}) = (hessnum(a) == hessnum(b)) && (tens(a) == tens(b))

########################
# Conversion/Promotion #
########################
convert{N,T,C}(::Type{TensorNum{N,T,C}}, t::TensorNum{N,T,C}) = t
convert{N,T,C}(::Type{TensorNum{N,T,C}}, x::Real) = TensorNum(HessianNum{N,T,C}(x), zeros(T, halftenslen(N)))

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
# In the code-generating loops below (see "Bivariate function construction loop"
# and "Univariate function construction loop"), we build definitions for math functions
# on TensorNums in a consistent, uniform manner by utilizing the `t_bivar_funcs` and
# `t_univar_funcs` arrays. These arrays hold multiple Tuples, each of which provides the
# necessary information to define a different function. The description of these
# Tuples' formats can be found in comments above their respective arrays.

function t2h(i, j)
    if i < j
        return div(j*(j-1), 2+i) + 1
    else
        return div(i*(i-1), 2+j) + 1
    end
end

const noexpr = Expr(:quote, nothing)

# Bivariate functions on TensorNums #
#-----------------------------------#

# The Tuples in `t_bivar_funcs` have the following format:
#
# (:function_name,
#  :(expression defining function-level variables, or `noexpr` if none),
#  :(expression defining the qth entry of the tensor vector, using any available variables))
const t_bivar_funcs = Tuple{Symbol, Expr, Expr}[
    (:*, noexpr, :(grad(t2,a)*hess(t1,r)+grad(t1,j)*hess(t2,r)+grad(t2,i)*hess(t1,m)+grad(t1,i)*hess(t2,m)+grad(t2,j)*hess(t1,l)
                   +grad(t1,j)*hess(t2,l)+value(t2)*tens(t1,q)+value(t1)*tens(t2,q))),
    (:/, noexpr, :((tens(t1,q)+((-(grad(t1,j)*hess(t2,r)+grad(t1,i)*hess(t2,m)+grad(t1,j)*hess(t2,l)+grad(t2,a)*hess(t1,r)+grad(t2,i)
                   *hess(t1,m)+grad(t2,j)*hess(t1,l)+value(t1)*tens(t2,q))+(2*(grad(t1,j)*grad(t2,i)*grad(t2,j)+grad(t2,a)*grad(t1,i)
                   *grad(t2,j)+grad(t2,a)*grad(t2,i)*grad(t1,j)+(value(t1)*(grad(t2,a)*hess(t2,r)+grad(t2,i)*hess(t2,m)+grad(t2,j)
                   *hess(t2,l)))-(3*value(t1)*grad(t2,a)*grad(t2,i)*grad(t2,j)/value(t2)))/value(t2)))/value(t2)))/value(t2))),
    (:^, :(logt1 = log(value(t1)); logt1sq = logt1^2; t1logt1 = value(t1)*logt1; t1logt1sq = value(t1)*logt1sq),
         :(value(t1)^(value(t2)-3)*(value(t2)^3*grad(t1,j)*grad(t1,i)*grad(t1,j)+value(t2)^2*(value(t1)*((logt1*grad(t2,j)
           *grad(t1,i)+hess(t1,r))*grad(t1,j)+grad(t1,i)*hess(t1,m))+grad(t1,j)*(grad(t1,i)*(-3*grad(t1,j)+t1logt1*grad(t2,a))
           +value(t1)*(logt1*grad(t2,i)*grad(t1,j)+hess(t1,l))))+value(t2)*(grad(t1,j)*(grad(t1,i)*(2*grad(t1,j)-value(t1)
           *(-2+logt1)*grad(t2,a))+value(t1)*(grad(t2,i)*(-(-2+logt1)*grad(t1,j)+t1logt1sq*grad(t2,a))-hess(t1,l)+t1logt1*hess(t2,l)))
           +value(t1)*(hess(t1,r)*(-grad(t1,j)+t1logt1*grad(t2,a))-grad(t1,i)*hess(t1,m)+grad(t2,j)*(grad(t1,i)*(-(-2+logt1)*grad(t1,j)
           +t1logt1sq*grad(t2,a))+t1logt1*(logt1*grad(t1,j)*grad(t2,i)+hess(t1,l)))+value(t1)*(logt1*(grad(t1,j)*hess(t2,r)+grad(t2,i)
           *hess(t1,m)+grad(t1,i)*hess(t2,m))+tens(t1,q))))+value(t1)*(grad(t1,j)*(-grad(t1,i)*grad(t2,a)-grad(t2,i)*(grad(t1,j)
           -2*t1logt1*grad(t2,a))+value(t1)*hess(t2,l))+grad(t2,j)*(-grad(t1,i)*(grad(t1,j)-2*t1logt1*grad(t2,a))+value(t1)
           *(hess(t1,l)+logt1*(grad(t2,i)*(2*grad(t1,j)+t1logt1sq*grad(t2,a))+t1logt1*hess(t2,l))))+value(t1)*(grad(t2,a)*hess(t1,r)
           +hess(t2,r)*(grad(t1,j)+t1logt1sq*grad(t2,a))+grad(t1,i)*hess(t2,m)+grad(t2,i)*(hess(t1,m)+t1logt1sq*hess(t2,m))
           +t1logt1*tens(t2,q))))))
]

# Bivariate function construction loop
for (fsym, vars, term) in t_bivar_funcs
    loadfsym = symbol(string("loadtens_", fsym, "!"))
    @eval begin
        function $(loadfsym){N}(t1::TensorNum{N}, t2::TensorNum{N}, output)
            q = 1
            $(vars)
            for a in 1:N
                for i in a:N
                    for j in a:i
                        l, m, r = t2h(a, i), t2h(a, j), t2h(i, j)
                        output[q] = $(term)
                        q += 1
                    end
                end
            end
            return output
        end

        function $(fsym){N,A,B}(t1::TensorNum{N,A}, t2::TensorNum{N,B})
            new_tens = Array(promote_type(A, B), halftenslen(N))
            return TensorNum($(fsym)(hessnum(t1), hessnum(t2)), $(loadfsym)(t1, t2, new_tens))
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
#/(x::Real, t::TensorNum) = ?

for T in (:Rational, :Integer, :Real)
    @eval begin
        function ^{N}(t::TensorNum{N}, p::$(T))
            new_tens = Array(promote_type(eltype(t), typeof(p)), halftenslen(N))
            q = 1
            for a in 1:N
                for i in a:N
                    for j in a:i
                        l, m, r = t2h(a, i), t2h(a, j), t2h(i, j)
                        new_tens[q] = (p*((p-1)*value(t)^(p-3)*((p-2)*grad(t,a)*grad(t,i)*grad(t,j)+value(t)
                                      *(grad(t,a)*hess(t,r)+grad(t,i)*hess(t,m)+grad(t,j)*hess(t,l)))+value(t)^2*tens(t,q)))
                        q += 1
                    end
                end
            end
            return TensorNum(hessnum(t)^p, new_tens)
        end
    end
end

# Univariate functions on TensorNums #
#------------------------------------#

# The Tuples in `t_univar_funcs` have the following format:
#
# (:function_name,
#  :(expression defining function-level variables, or `noexpr` if none),
#  :(expression defining inner-loop variables, or `noexpr` if none),
#  :(expression defining the qth entry of the tensor vector, using any available variables))
const t_univar_funcs = Tuple{Symbol, Expr, Expr, Expr}[
    (:sqrt, noexpr, noexpr, :(((0.375*grad(t,a)*grad(t,i)*grad(t,j)/value(t)-0.25*(grad(t,a)*hess(t,r)+grad(t,i)*hess(t,m)+grad(t,j)*hess(t,l)))/value(t)+0.5*tens(t,q))/sqrt(value(t)))),
    (:cbrt, noexpr, noexpr, :(((10*grad(t,a)*grad(t,i)*grad(t,j)/(3*value(t))-2*(grad(t,a)*hess(t,r)+grad(t,i)*hess(t,m)+grad(t,j)*hess(t,l)))/(3*value(t))+tens(t,q))/(3*value(t)^(2/3)))),
    (:exp, noexpr, noexpr, :(exp(value(t))*(grad(t,a)*grad(t,i)*grad(t,j)+grad(t,a)*hess(t,r)+grad(t,i)*hess(t,m)+grad(t,j)*hess(t,l)+tens(t,q)))),
    (:log, noexpr, noexpr, :(((2*grad(t,a)*grad(t,i)*grad(t,j)/value(t)-(grad(t,a)*hess(t,r)+grad(t,i)*hess(t,m)+grad(t,j)*hess(t,l)))/value(t)+tens(t,q))/value(t))),
    (:log2, noexpr, noexpr, :(((2*grad(t,a)*grad(t,i)*grad(t,j)/value(t)-(grad(t,a)*hess(t,r)+grad(t,i)*hess(t,m)+grad(t,j)*hess(t,l)))/value(t)+tens(t,q))/(value(t)*convert(T, 0.6931471805599453)))),
    (:log10, noexpr, noexpr, :(((2*grad(t,a)*grad(t,i)*grad(t,j)/value(t)-(grad(t,a)*hess(t,r)+grad(t,i)*hess(t,m)+grad(t,j)*hess(t,l)))/value(t)+tens(t,q))/(value(t)*convert(T, 2.302585092994046)))),
    (:sin, noexpr, noexpr, :(cos(value(t))*(tens(t,q)-grad(t,a)*grad(t,i)*grad(t,j))-sin(value(t))*(grad(t,a)*hess(t,r)+grad(t,i)*hess(t,m)+grad(t,j)*hess(t,l)))),
    (:cos, noexpr, noexpr, :(sin(value(t))*(grad(t,a)*grad(t,i)*grad(t,j)-tens(t,q))-cos(value(t))*(grad(t,a)*hess(t,r)+grad(t,i)*hess(t,m)+grad(t,j)*hess(t,l)))),
    (:tan, :(tanx = tan(value(t)); secxsq = sec(value(t))^2), noexpr, :(secxsq*(2*tanx*(grad(t,a)*hess(t,r)+grad(t,i)*hess(t,m))+2*grad(t,j)*((3*secxsq-2)*grad(t,a)*grad(t,i)+tanx*hess(t,l))+tens(t,q)))),
    (:asin, :(xsq = value(t)^2; oneminusxsq = 1-xsq), :(gprod = grad(t,a)*grad(t,i)*grad(t,j)), :(((3*xsq*gprod/oneminusxsq+gprod+(+grad(t,a)*hess(t,r)+grad(t,i)*hess(t,m)+grad(t,j)*hess(t,l))*value(t))/oneminusxsq+tens(t,q))/oneminusxsq^0.5)),
    (:acos,  :(xsq = value(t)^2; oneminusxsq = 1-xsq), :(gprod = grad(t,a)*grad(t,i)*grad(t,j)), :(-((3*xsq*gprod/oneminusxsq+gprod+(+grad(t,a)*hess(t,r)+grad(t,i)*hess(t,m)+grad(t,j)*hess(t,l))*value(t))/oneminusxsq+tens(t,q))/oneminusxsq^0.5)),
    (:atan, :(xsq = value(t)^2; oneplusxsq = 1+xsq), :(gprod = grad(t,a)*grad(t,i)*grad(t,j)), :(((4*xsq*gprod/oneplusxsq-gprod-(+grad(t,a)*hess(t,r)+grad(t,i)*hess(t,m)+grad(t,j)*hess(t,l))*value(t))*2/oneplusxsq+tens(t,q))/oneplusxsq)),
    (:sinh, noexpr, noexpr, :(cosh(value(t))*(grad(t,a)*grad(t,i)*grad(t,j)+tens(t,q))+sinh(value(t))*(+grad(t,a)*hess(t,r)+grad(t,i)*hess(t,m)+grad(t,j)*hess(t,l)))),
    (:cosh, noexpr, noexpr, :(sinh(value(t))*(grad(t,a)*grad(t,i)*grad(t,j)+tens(t,q))+cosh(value(t))*(+grad(t,a)*hess(t,r)+grad(t,i)*hess(t,m)+grad(t,j)*hess(t,l)))),
    (:tanh, :(sechxsq = sech(value(t))^2; tanhx = tanh(value(t))), noexpr, :(sechxsq*(-2*(tanhx*(grad(t,a)*hess(t,r)+grad(t,i)*hess(t,m))+grad(t,j)*((3*sechxsq-2)*grad(t,a)*grad(t,i)+tanhx*hess(t,l)))+tens(t,q)))),
    (:asinh, :(xsq = value(t)^2; oneplusxsq = 1+xsq), :(gprod = grad(t,a)*grad(t,i)*grad(t,j)), :(((3*xsq*gprod/oneplusxsq-gprod-(+grad(t,a)*hess(t,r)+grad(t,i)*hess(t,m)+grad(t,j)*hess(t,l))*value(t))/oneplusxsq+tens(t,q))/oneplusxsq^0.5)),
    (:acosh, :(xsq = value(t)^2), noexpr, :((grad(t,j)*((2*xsq+1)*grad(t,a)*grad(t,i)-value(t)*(xsq-1)*hess(t,l))+(xsq-1)*(-value(t)*(grad(t,a)*hess(t,r)+grad(t,i)*hess(t,m))+tens(t,q)*(xsq-1)))/(xsq-1)^2.5)),
    (:atanh, :(xsq = value(t)^2; oneminusxsq = 1-xsq), :(gprod = grad(t,a)*grad(t,i)*grad(t,j)), :(((4*xsq*gprod/oneminusxsq+gprod+(+grad(t,a)*hess(t,r)+grad(t,i)*hess(t,m)+grad(t,j)*hess(t,l))*value(t))*2/oneminusxsq+tens(t,q))/oneminusxsq))
]

# Univariate function construction loop
for (fsym, funcvars, loopvars, term) in t_univar_funcs
    loadfsym = symbol(string("loadtens_", fsym, "!"))
    @eval begin
        function $(loadfsym){N}(t::TensorNum{N}, output)
            q = 1
            $(funcvars)
            for a in 1:N
                for i in a:N
                    for j in a:i
                        l, m, r = t2h(a, i), t2h(a, j), t2h(i, j)
                        $(loopvars)
                        output[q] = $(term)
                        q += 1
                    end
                end
            end
            return output
        end

        function $(fsym){N,T}(t::TensorNum{N,T})
            ResultType = typeof($(fsym)(one(T)))
            new_tens = Array(ResultType, halftenslen(N))
            return TensorNum($(fsym)(hessnum(t)), $(loadfsym)(t, new_tens))
        end
    end
end

-(t::TensorNum) = TensorNum(-hessnum(t), -tens(t))


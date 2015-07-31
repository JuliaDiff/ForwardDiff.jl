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
# The coefficients of ϵ₁ϵ₂ are what's stored by TensorNum's `tens` field:
#
#   f(t)_ϵ₁ϵ₂ϵ₃ = (f'(t₀)*t₇ + f''(t₀)*(t₃t₄ + t₂t₅ + t₁t₆) + f'''(t₀)*t₁t₂t₃
#
# where, in loop code:
#
#   a, b, c = t_inds_2_h_ind(i, j), t_inds_2_h_ind(j, k), t_inds_2_h_ind(k, i)
#   t₀ = value(h)
#   t₁ = grad(t, i)
#   t₂ = grad(t, j)
#   t₃ = grad(t, k)
#   t₄ = hess(t, a)
#   t₅ = hess(t, b)
#   t₆ = hess(t, c)
#   t₇ = tens(t, q)
#
# see http://adl.stanford.edu/hyperdual/Fike_AIAA-2011-886.pdf for details.

function t_inds_2_h_ind(i, j)
    if i < j
        return div(j*(j-1), 2+i) + 1
    else
        return div(i*(i-1), 2+j) + 1
    end
end

# Bivariate functions on TensorNums #
#-----------------------------------#
# function loadtens_mul!{N}(t1::TensorNum{N}, t2::TensorNum{N}, output)
#     q = 1
#     for i in 1:N
#         for j in i:N
#             for k in i:j
#                 a, b, c = t_inds_2_h_ind(i, j), t_inds_2_h_ind(j, k), t_inds_2_h_ind(k, i)
#                 output[q] = 
#                 q += 1
#             end
#         end
#     end
#     return output
# end

# function loadtens_div!{N}(t1::TensorNum{N}, t2::TensorNum{N}, output)
#     q = 1
#     for i in 1:N
#         for j in i:N
#             for k in i:j
#                 a, b, c = t_inds_2_h_ind(i, j), t_inds_2_h_ind(j, k), t_inds_2_h_ind(k, i)
#                 output[q] = 
#                 q += 1
#             end
#         end
#     end
#     return output
# end

# function loadtens_div!{N}(x::Real, h::TensorNum{N}, output)
#     q = 1
#     for i in 1:N
#         for j in i:N
#             for k in i:j
#                 a, b, c = t_inds_2_h_ind(i, j), t_inds_2_h_ind(j, k), t_inds_2_h_ind(k, i)
#                 output[q] = 
#                 q += 1
#             end
#         end
#     end
#     return output
# end

# function loadtens_exp!{N}(t1::TensorNum{N}, t2::TensorNum{N}, output)
#     q = 1
#     for i in 1:N
#         for j in i:N
#             for k in i:j
#                 a, b, c = t_inds_2_h_ind(i, j), t_inds_2_h_ind(j, k), t_inds_2_h_ind(k, i)
#                 output[q] = 
#                 q += 1
#             end
#         end
#     end
#     return output
# end

# function loadtens_exp!{N}(h::TensorNum{N}, p::Real, output)
#     q = 1
#     for i in 1:N
#         for j in i:N
#             for k in i:j
#                 a, b, c = t_inds_2_h_ind(i, j), t_inds_2_h_ind(j, k), t_inds_2_h_ind(k, i)
#                 output[q] = 
#                 q += 1
#             end
#         end
#     end
#     return output
# end

for (fsym, loadfsym) in [(:*, symbol("loadtens_mul!")),
                         (:/, symbol("loadtens_div!")), 
                         (:^, symbol("loadtens_exp!"))]
    @eval function $(fsym){N,A,B}(a::TensorNum{N,A}, b::TensorNum{N,B})
        new_tens = Array(promote_type(A, B), halftenslen(N))
        return TensorNum($(fsym)(hessnum(a), hessnum(b)), $(loadfsym)(a, b, new_tens))
    end
end

function /{N,T}(x::Real, t::TensorNum{N,T})
    new_hess = Array(promote_type(T, typeof(x)), halftenslen(N))
    return TensorNum(x / hessnum(t), loadtens_div!(x, t, new_tens))
end

for T in (:Rational, :Integer, :Real)
    @eval begin
        function ^{N}(t::TensorNum{N}, p::$(T))
            new_hess = Array(promote_type(eltype(t), typeof(p)), halftenslen(N))
            return TensorNum(hessnum(t)^p, loadtens_exp!(h, p, new_tens))
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

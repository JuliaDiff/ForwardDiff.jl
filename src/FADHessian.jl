immutable FADHessian{N,T,D} <: FADNumber{N,T,D}
    grad::D 
    hess::Vector{T}
    function FADHessian(grad::NDual{N,T}, hess::Vector)
        @assert length(hess) == halfhesslen(N)
        return new(grad, hess)
    end
end

function FADHessian{N,T}(grad::NDual{N,T}, 
                         hess::Vector=zeros(T, halfhesslen(N)))
    return FADHessian{N,T,typeof(grad)}(grad, hess)
end

##############################
# Utility/Accessor Functions #
##############################
zero{N,T,D}(::Type{FADHessian{N,T,D}}) = FADHessian(zero(D), zeros(T, halfhesslen(N)))
one{N,T,D}(::Type{FADHessian{N,T,D}}) = FADHessian(one(D), zeros(T, halfhesslen(N)))

hess(h::FADHessian) = h.hess
tens(h::FADHessian) = error("FADHessians do not store tensor values")
grad(h::FADHessian) = h.grad

neps{N,T,D}(::Type{FADHessian{N,T,D}}) = N
eltype{N,T,D}(::Type{FADHessian{N,T,D}}) = T

function isconstant(h::FADHessian)
    zeroT = zero(eltype(h))
    return isreal(grad(h)) && all(x -> x == zeroT, hess(h))
end

function isfinite(h::FADHessian)
    oneT = one(eltype(h))
    return isfinite(grad(h)) && all(x -> x == oneT, hess(h))
end

=={N}(a::FADHessian{N}, b::FADHessian{N}) = (grad(a) == grad(b)) && (hess(a) == hess(b))

########################
# Conversion/Promotion #
########################
convert{N,T,D}(::Type{FADHessian{N,T,D}}, h::FADHessian{N,T,D}) = h
convert{N,T,D}(::Type{FADHessian{N,T,D}}, x::Real) = FADHessian(D(x), zeros(T, halfhesslen(N)))

function convert{N,T,D}(::Type{FADHessian{N,T,D}}, h::FADHessian{N})
    return FADHessian(convert(D, grad(h)), hess(h))
end

function convert{T<:Real}(::Type{T}, h::FADHessian)
    if isconstant(h)
        return convert(T, value(h))
    else
        throw(InexactError)
    end
end

promote_rule{N,T,D}(::Type{FADHessian{N,T,D}}, ::Type{T}) = FADHessian{N,T,D}
promote_rule{N,T,D,S}(::Type{FADHessian{N,T,D}}, ::Type{S}) = FADHessian{N,promote_type(T, S),promote_type(D, S)}
function promote_rule{N,T1,D1,T2,D2}(::Type{FADHessian{N,T1,D1}}, ::Type{FADHessian{N,T2,D2}})
    return FADHessian{N,promote_type(T1, T2),promote_type(D1, D2)}
end

#######################
# Math on FADHessians #
#######################

## Bivariate functions on FADHessians ##
##------------------------------------##

const h_bivar_funcs = Tuple{Symbol, Expr}[
    (:*, :(hess(a,k)*value(b)+grad(a,i)*grad(b,j)+grad(a,j)*grad(b,i)+value(a)*hess(b,k))),
    (:/, :(((2*value(a)*grad(b,j)*grad(b,i)+(value(b)^2)*hess(a,k))-(value(b)*(grad(a,i)*grad(b,j)
           +grad(a,j)*grad(b,i)+value(a)*hess(b,k))))/(value(b)^3))),
    (:^, :((value(a)^(value(b)-2))*((value(b)^2)*grad(a,i)*grad(a,j)+value(b)*(grad(a,j)*(-grad(a,i)
           +value(a)*log(value(a))*grad(b,i))+value(a)*(log(value(a))*grad(a,i)*grad(b,j)
           +hess(a,k)))+value(a)*(grad(a,j)*grad(b,i)+grad(b,j)*(grad(a,i)+value(a)*log(value(a))
           *log(value(a))*grad(b,i))+value(a)*log(value(a))*hess(b,k)))))
]

for (fsym, term) in h_bivar_funcs
    loadfsym = symbol(string("loadhess_", fsym, "!"))
    @eval begin
        function $(loadfsym){N}(a::FADHessian{N}, b::FADHessian{N}, output)
            k = 1
            for i in 1:N
                for j in 1:i
                    output[k] = $(term)
                    k += 1
                end
            end
            return output
        end

        function $(fsym){N,A,B}(a::FADHessian{N,A}, b::FADHessian{N,B})
            new_hess = Array(promote_type(A, B), halfhesslen(N))
            return FADHessian($(fsym)(grad(a), grad(b)), $(loadfsym)(a, b, new_hess))
        end

    end
end

+{N}(a::FADHessian{N}, b::FADHessian{N}) = FADHessian(grad(a) + grad(b), hess(a) + hess(b))
-{N}(a::FADHessian{N}, b::FADHessian{N}) = FADHessian(grad(a) - grad(b), hess(a) - hess(b))

for T in (:Bool, :Real)
    @eval begin
        *(h::FADHessian, x::$(T)) = FADHessian(grad(h) * x, hess(h) * x)
        *(x::$(T), h::FADHessian) = FADHessian(x * grad(h), x * hess(h))
    end
end

/(h::FADHessian, x::Real) = FADHessian(grad(h) / x, hess(h) / x)
#/(x::Real, h::FADHessian) = ?

for T in (:Rational, :Integer, :Real)
    @eval begin
        function ^{N}(h::FADHessian{N}, p::$(T))
            new_hess = Array(promote_type(eltype(h), typeof(p)), halfhesslen(N))
            k = 1
            for i in 1:N
                for j in 1:i
                    new_hess[k] = p*value(h)^(p-2)*((p-1)*grad(h,i)*grad(h,j)+value(h)*hess(h,k))
                    k += 1
                end
            end
            return FADHessian(grad(h)^p, new_hess)
        end
    end
end

## Univariate functions on FADHessians ##
##-------------------------------------##

-(h::FADHessian) = FADHessian(-grad(h), -hess(h))

const h_univar_funcs = Tuple{Symbol, Expr}[
    (:sqrt, :((-grad(h,i)*grad(h,j)+2*value(h)*hess(h,i)) / (4*(value(h)^(1.5))))),
    (:cbrt, :((-2*grad(h,i)*grad(h,j)+3*value(h)*hess(h,k)) / (9*cbrt(value(h)^5)))),
    (:ehp, :(ehp(value(h))*(grad(h,i)*grad(h,j)+hess(h,k)))),
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
    (:atanh, :((2*value(h)*grad(h,i)*grad(h,j)-(((value(h)^2)-1)*hess(h,k)))/(((value(h)^2)-1)^2)))
]

for (fsym, term) in h_univar_funcs
    loadfsym = symbol(string("loadhess_", fsym, "!"))
    @eval begin
        function $(loadfsym){N}(h::FADHessian{N}, output)
            k = 1
            for i in 1:N
                for j in 1:i
                    output[k] = $(term)
                    k += 1
                end
            end
            return output
        end

        function $(fsym){N,T}(h::FADHessian{N,T})
            ResultType = typeof($(fsym)(one(T)))
            new_hess = Array(ResultType, halfhesslen(N))
            return FADHessian($(fsym)(grad(h)), $(loadfsym)(h, new_hess))
        end
    end
end

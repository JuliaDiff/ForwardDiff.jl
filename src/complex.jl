Base.prevfloat(x::Dual{T,V}) where {T,V<:AbstractFloat} = prevfloat(x.value)
Base.nextfloat(x::Dual{T,V}) where {T,V<:AbstractFloat} = nextfloat(x.value)

function Base.log(z::Complex{T}) where {A, FT<:AbstractFloat, T<:Dual{A,FT}}
    T1::T  = 1.25
    T2::T  = 3
    ln2::T = log(convert(T,2))  #0.6931471805599453
    x, y = reim(z)
    ρ, k = Base.ssqs(x,y)
    ax = abs(x)
    ay = abs(y)
    if ax < ay
        θ, β = ax, ay
    else
        θ, β = ay, ax
    end
    if k==0 && (0.5 < β*β) && (β <= T1 || ρ < T2)
        ρρ = log1p((β-1)*(β+1)+θ*θ)/2
    else
        ρρ = log(ρ)/2 + k*ln2
    end
    Complex(ρρ, angle(z))
end
function Base.tanh(z::Complex{T}) where {A, FT<:AbstractFloat, T<:Dual{A,FT}}
    Ω = prevfloat(typemax(T))
    ξ, η = reim(z)
    if isnan(ξ) && η==0 return Complex(ξ, η) end
    if 4*abs(ξ) > asinh(Ω) #Overflow?
        Complex(copysign(one(T),ξ),
                copysign(zero(T),η*(isfinite(η) ? sin(2*abs(η)) : one(η))))
    else
        t = tan(η)
        β = 1+t*t #sec(η)^2
        s = sinh(ξ)
        ρ = sqrt(1 + s*s) #cosh(ξ)
        if isinf(t)
            Complex(ρ/s,1/t)
        else
            Complex(β*ρ*s,t)/(1+β*s*s)
        end
    end
end

_convert(T, x::Dual) = convert(T, x.value)
function Base._cpow(z::Union{Dual{A,T}, Complex{<:Dual{A,T}}}, p::Union{Dual{B,T}, Complex{<:Dual{B,T}}}) where {T,A,B}
    if isreal(p)
        pᵣ = real(p)
        if isinteger(pᵣ) && abs(pᵣ) < typemax(Int32)
            # |p| < typemax(Int32) serves two purposes: it prevents overflow
            # when converting p to Int, and it also turns out to be roughly
            # the crossover point for exp(p*log(z)) or similar to be faster.
            if iszero(pᵣ) # fix signs of imaginary part for z^0
                zer = flipsign(copysign(zero(T),pᵣ), imag(z))
                return Complex(one(T), zer)
            end
            ip = _convert(Int, pᵣ)
            if isreal(z)
                zᵣ = real(z)
                if ip < 0
                    iszero(z) && return Complex(T(NaN),T(NaN))
                    re = Base.power_by_squaring(inv(zᵣ), -ip)
                    im = -imag(z)
                else
                    re = Base.power_by_squaring(zᵣ, ip)
                    im = imag(z)
                end
                # slightly tricky to get the correct sign of zero imag. part
                return Complex(re, ifelse(iseven(ip) & signbit(zᵣ), -im, im))
            else
                return ip < 0 ? Base.power_by_squaring(inv(z), -ip) : Base.power_by_squaring(z, ip)
            end
        elseif isreal(z)
            # (note: if both z and p are complex with ±0.0 imaginary parts,
            #  the sign of the ±0.0 imaginary part of the result is ambiguous)
            if iszero(real(z))
                return pᵣ > 0 ? complex(z) : Complex(T(NaN),T(NaN)) # 0 or NaN+NaN*im
            elseif real(z) > 0
                return Complex(real(z)^pᵣ, z isa Real ? ifelse(real(z) < 1, -imag(p), imag(p)) : flipsign(imag(z), pᵣ))
            else
                zᵣ = real(z)
                rᵖ = (-zᵣ)^pᵣ
                if isfinite(pᵣ)
                    # figuring out the sign of 0.0 when p is a complex number
                    # with zero imaginary part and integer/2 real part could be
                    # improved here, but it's not clear if it's worth it…
                    return rᵖ * complex(cospi(pᵣ), flipsign(sinpi(pᵣ),imag(z)))
                else
                    iszero(rᵖ) && return zero(Complex{T}) # no way to get correct signs of 0.0
                    return Complex(T(NaN),T(NaN)) # non-finite phase angle or NaN input
                end
            end
        else
            rᵖ = abs(z)^pᵣ
            ϕ = pᵣ*angle(z)
        end
    elseif isreal(z)
        iszero(z) && return real(p) > 0 ? complex(z) : Complex(T(NaN),T(NaN)) # 0 or NaN+NaN*im
        zᵣ = real(z)
        pᵣ, pᵢ = reim(p)
        if zᵣ > 0
            rᵖ = zᵣ^pᵣ
            ϕ = pᵢ*log(zᵣ)
        else
            r = -zᵣ
            θ = copysign(T(π),imag(z))
            rᵖ = r^pᵣ * exp(-pᵢ*θ)
            ϕ = pᵣ*θ + pᵢ*log(r)
        end
    else
        pᵣ, pᵢ = reim(p)
        r = abs(z)
        θ = angle(z)
        rᵖ = r^pᵣ * exp(-pᵢ*θ)
        ϕ = pᵣ*θ + pᵢ*log(r)
    end

    if isfinite(ϕ)
        return rᵖ * cis(ϕ)
    else
        iszero(rᵖ) && return zero(Complex{T}) # no way to get correct signs of 0.0
        return Complex(T(NaN),T(NaN)) # non-finite phase angle or NaN input
    end
end

function Base.ssqs(x::T, y::T) where T<:Dual
    k::Int = 0
    ρ = x*x + y*y
    if !isfinite(ρ) && (isinf(x) || isinf(y))
        ρ = convert(T, Inf)
    elseif isinf(ρ) || (ρ==0 && (x!=0 || y!=0)) || ρ<nextfloat(zero(T))/(2*eps(T)^2)
        m::T = max(abs(x), abs(y))
        k = m==0 ? m : exponent(m)
        xk, yk = ldexp(x,-k), ldexp(y,-k)
        ρ = xk*xk + yk*yk
    end
    ρ, k
end

function Base.sqrt(z::Complex{T}) where {T<:Dual{<:Any,<:AbstractFloat}}
    x, y = reim(z)
    if x==y==0
        return Complex(zero(x),y)
    end
    ρ, k::Int = Base.ssqs(x, y)
    if isfinite(x) ρ=_ldexp(abs(x),-k)+sqrt(ρ) end
    if isodd(k)
        k = div(k-1,2)
    else
        k = div(k,2)-1
        ρ += ρ
    end
    ρ = _ldexp(sqrt(ρ),k) #sqrt((abs(z)+abs(x))/2) without over/underflow
    ξ = ρ
    η = y
    if ρ != 0
        if isfinite(η) η=(η/ρ)/2 end
        if x<0
            ξ = abs(η)
            η = copysign(ρ,y)
        end
    end
    Complex(ξ,η)
end

# TODO: polish this ldexp function.
function _ldexp(x::T, e::Integer) where T<:Dual
    if e >=0
        x * (1<<e)
    else
        x / (1<<-e)
    end
end

module ForwardDiffAbstractFFTsExt

using ForwardDiff, AbstractFFTs

import AbstractFFTs: plan_fft, plan_ifft, plan_bfft, plan_rfft, plan_brfft, plan_irfft, Plan
using ForwardDiff: array2dual, dual2array
import LinearAlgebra: mul!

for P in (:Plan, :ScaledPlan)  # need ScaledPlan to avoid ambiguities
    @eval begin
        Base.:*(p::AbstractFFTs.$P, x::AbstractArray{DT}) where DT<:Dual = array2dual(DT, p * dual2array(x))
        Base.:*(p::AbstractFFTs.$P, x::AbstractArray{<:Complex{DT}}) where DT<:Dual = array2dual(DT, p * dual2array(x))
    end
end

mul!(y::AbstractArray{<:Union{Dual,Complex{<:Dual}}}, p::Plan, x::AbstractArray{<:Union{Dual,Complex{<:Dual}}}) = copyto!(y, p*x)

AbstractFFTs.complexfloat(x::AbstractArray{<:Dual}) = AbstractFFTs.complexfloat.(x)
AbstractFFTs.complexfloat(d::Dual{T,V,N}) where {T,V,N} = convert(Dual{T,float(V),N}, d) + 0im

AbstractFFTs.realfloat(x::AbstractArray{<:Dual}) = AbstractFFTs.realfloat.(x)
AbstractFFTs.realfloat(d::Dual{T,V,N}) where {T,V,N} = convert(Dual{T,float(V),N}, d)

for plan in (:plan_fft, :plan_ifft, :plan_bfft, :plan_rfft)
    @eval begin
        $plan(x::AbstractArray{<:Dual}, dims=1:ndims(x)) = $plan(dual2array(x), 1 .+ dims)
        $plan(x::AbstractArray{<:Complex{<:Dual}}, dims=1:ndims(x)) = $plan(dual2array(x), 1 .+ dims)
    end
end

for plan in (:plan_irfft, :plan_brfft)  # these take an extra argument, only when complex?
    @eval begin
        $plan(x::AbstractArray{<:Dual}, dims=1:ndims(x)) = $plan(dual2array(x), 1 .+ dims)
        $plan(x::AbstractArray{<:Complex{<:Dual}}, d::Integer, dims=1:ndims(x)) = $plan(dual2array(x), d, 1 .+ dims)
    end
end


end
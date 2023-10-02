module ForwardDiffFFTWExt

using ForwardDiff, FFTW

import FFTW: r2r, r2r!, plan_r2r


plan_r2r(x::AbstractArray{<:Dual}, FLAG, dims=1:ndims(x)) = plan_r2r(dual2array(x), FLAG, 1 .+ dims)
plan_r2r(x::AbstractArray{<:Complex{<:Dual}}, FLAG, dims=1:ndims(x)) = plan_r2r(dual2array(x), FLAG, 1 .+ dims)

r2r(x::AbstractArray{<:Dual}, kinds, region...) = plan_r2r(x, kinds, region...) * x
r2r(x::AbstractArray{<:Complex{<:Dual}}, kinds, region...) = plan_r2r(x, kinds, region...) * x


end
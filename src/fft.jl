
value(x::Complex{<:Dual}) =
    Complex(x.re.value, x.im.value)

partials(x::Complex{<:Dual}, n::Int) =
    Complex(partials(x.re, n), partials(x.im, n))

npartials(x::Complex{<:Dual{T,V,N}}) where {T,V,N} = N
npartials(::Type{<:Complex{<:Dual{T,V,N}}}) where {T,V,N} = N

# AbstractFFTs.complexfloat(x::AbstractArray{<:Dual}) = float.(x .+ 0im)
AbstractFFTs.complexfloat(x::AbstractArray{<:Dual}) = AbstractFFTs.complexfloat.(x)
AbstractFFTs.complexfloat(d::Dual{T,V,N}) where {T,V,N} = convert(Dual{T,float(V),N}, d) + 0im

AbstractFFTs.realfloat(x::AbstractArray{<:Dual}) = AbstractFFTs.realfloat.(x)
AbstractFFTs.realfloat(d::Dual{T,V,N}) where {T,V,N} = convert(Dual{T,float(V),N}, d)

for plan in [:plan_fft, :plan_ifft, :plan_bfft]
    @eval begin

        AbstractFFTs.$plan(x::AbstractArray{<:Dual}, region=1:ndims(x)) =
            AbstractFFTs.$plan(value.(x), region)

        AbstractFFTs.$plan(x::AbstractArray{<:Complex{<:Dual}}, region=1:ndims(x)) =
            AbstractFFTs.$plan(value.(x), region)

    end
end

# rfft only accepts real arrays
AbstractFFTs.plan_rfft(x::AbstractArray{<:Dual}, region=1:ndims(x)) =
    AbstractFFTs.plan_rfft(value.(x), region)

for plan in [:plan_irfft, :plan_brfft]  # these take an extra argument, only when complex?
    @eval begin

        AbstractFFTs.$plan(x::AbstractArray{<:Dual}, region=1:ndims(x)) =
            AbstractFFTs.$plan(value.(x), region)

        AbstractFFTs.$plan(x::AbstractArray{<:Complex{<:Dual}}, d::Integer, region=1:ndims(x)) =
            AbstractFFTs.$plan(value.(x), d, region)

    end
end

# for f in (:dct, :idct)
#     pf = Symbol("plan_", f)
#     @eval begin
#         AbstractFFTs.$f(x::AbstractArray{<:Dual}) = $pf(x) * x
#         AbstractFFTs.$f(x::AbstractArray{<:Dual}, region) = $pf(x, region) * x
#         AbstractFFTs.$pf(x::AbstractArray{<:Dual}, region; kws...) = $pf(value.(x), region; kws...)
#         AbstractFFTs.$pf(x::AbstractArray{<:Complex}, region; kws...) = $pf(value.(x), region; kws...)
#     end
# end


for P in [:Plan, :ScaledPlan]  # need ScaledPlan to avoid ambiguities
    @eval begin

        Base.:*(p::AbstractFFTs.$P, x::AbstractArray{<:Dual}) =
            _apply_plan(p, x)

        Base.:*(p::AbstractFFTs.$P, x::AbstractArray{<:Complex{<:Dual}}) =
            _apply_plan(p, x)

    end
end

function _apply_plan(p::AbstractFFTs.Plan, x::AbstractArray)
    xtil = p * value.(x)
    dxtils = ntuple(npartials(eltype(x))) do n
        p * partials.(x, n)
    end
    __apply_plan(tagtype(eltype(x)), xtil, dxtils)
end

function __apply_plan(T, xtil, dxtils)
    map(xtil, dxtils...) do val, parts...
        Complex(
            Dual{T}(real(val), map(real, parts)),
            Dual{T}(imag(val), map(imag, parts)),
        )
    end
end
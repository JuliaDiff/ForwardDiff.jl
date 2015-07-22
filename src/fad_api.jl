####################
# Taking Gradients #
####################

# Gradient from ForwardDiffNum #
#------------------------------#
function gradient!(n::ForwardDiffNum, output)
    @assert npartials(n) == length(output)
    @inbounds @simd for i in eachindex(output)
        output[i] = grad(n, i)
    end
    return output
end

gradient{N,T,C}(n::ForwardDiffNum{N,T,C}) = gradient!(n, Array(T, N))

# Gradient from function #
#------------------------#
function gradient!{N,T,C,S}(f::Function,
                            x::Vector{T},
                            output::Vector{S},
                            gradvec::Vector{GradientNum{N,T,C}}) 
    xlen = length(x)
    Grad = eltype(gradvec)
    ResultGrad = GradientNum{N,S,switch_eltype(C,S)}

    @assert xlen == length(output) "The output array must be the same length as x"
    @assert xlen == length(gradvec) "The GradientNum vector must be the same length as the input vector"
    @assert xlen % N == 0 "Length of input vector is indivisible by the number of partials components (length(x) = $k, npartials(eltype(gradvec)) = $N)"

    pchunk = partials_chunk(Grad)

    # We can do less work filling and
    # zeroing out gradvec if xlen == N
    if xlen == N
        @inbounds @simd for i in 1:xlen
            gradvec[i] = Grad(x[i], pchunk[i])
        end

        result::ResultGrad = f(gradvec)
        
        gradient!(result, output)
    else
        zpartials = zero_partials(Grad)

        # load x[i]-valued GradientNums into gradvec 
        @inbounds @simd for i in 1:xlen
            gradvec[i] = Grad(x[i], zpartials)
        end

        for i in 1:N:xlen
            # load GradientNums with single
            # partial components into current 
            # chunk of gradvec
            @inbounds @simd for j in 0:(N-1)
                m = i+j
                gradvec[m] = Grad(x[m], pchunk[j+1])
            end

            chunk_result::ResultGrad = f(gradvec)

            # load resultant partials components
            # into output, replacing them with 
            # zeros in gradvec
            @inbounds @simd for j in 0:(N-1)
                m = i+j
                output[m] = grad(chunk_result, j+1)
                gradvec[m] = Grad(x[m], zpartials)
            end
        end
    end

    return output
end

function gradient!{N,T,C}(f::Function, x::Vector{T}, output::Vector, Grad::Type{GradientNum{N,T,C}})
    return gradient!(f, x, output, similar(x, Grad))
end

function gradient!{N,T,C}(f::Function, x::Vector{T}, gradvec::Vector{GradientNum{N,T,C}})
    return gradient!(f, x, similar(x), gradvec)
end

function gradient{N,T,C}(f::Function, x::Vector{T}, Grad::Type{GradientNum{N,T,C}})
    return gradient!(f, x, similar(x), Grad)
end

function gradient_func{G<:GradientNum}(f::Function, xlen::Integer, ::Type{G}, mutates=true)
    gradvec = Vector{G}(xlen)
    T = eltype(G)
    if mutates
        gradf!{T}(x::Vector{T}, output::Vector) = gradient!(f, x, output, gradvec)
        return gradf!
    else
        gradf{T}(x::Vector{T}) = gradient!(f, x, gradvec)
        return gradf
    end
end

####################
# Taking Jacobians #
####################

# Jacobian from Vector{F<:ForwardDiffNum} #
#-----------------------------------------#
function jacobian!{F<:ForwardDiffNum}(v::Vector{F}, output)
    N = npartials(F)
    @assert (length(v), N) == size(output)
    for i in 1:length(v), j in 1:N
        output[i,j] = grad(v[i], j)
    end
    return output
end

jacobian{F<:ForwardDiffNum}(v::Vector{F}) = jacobian!(v, Array(eltype(F), length(v), npartials(F)))


# Jacobian from function #
#------------------------#
function jacobian!{N,T,C,S}(f::Function,
                            x::Vector{T},
                            output::Matrix{S},
                            gradvec::Vector{GradientNum{N,T,C}})
    xlen = length(x)
    Grad = eltype(gradvec)
    ResultGradVec = Vector{GradientNum{N,S,switch_eltype(C,S)}}

    @assert (xlen, N) == size(output) "The output matrix must have size (length(input), npartials(eltype(gradvec)))"
    @assert xlen == length(gradvec) "The GradientNum vector must be the same length as the input vector"
    @assert xlen % N == 0 "Length of input vector is indivisible by the number of partials components (length(x) = $k, npartials(eltype(gradvec)) = $N)"

    pchunk = partials_chunk(Grad)

    # We can do less work filling and
    # zeroing out gradvec if xlen == N
    if xlen == N
        @inbounds @simd for i in 1:xlen
            gradvec[i] = Grad(x[i], pchunk[i])
        end

        result::ResultGradVec = f(gradvec)

        jacobian!(result, output)
    else
        zpartials = zero_partials(Grad)

        # load x[i]-valued GradientNums into gradvec
        @inbounds @simd for i in 1:xlen
            gradvec[i] = Grad(x[i], zpartials)
        end

        for i in 1:N:xlen
            # load GradientNums with single
            # partial components into current
            # chunk of gradvec
            @inbounds @simd for j in 0:(N-1)
                m = i+j
                gradvec[m] = Grad(x[m], pchunk[j+1])
            end

            chunk_result::ResultGradVec = f(gradvec)

            # load resultant partials components
            # into output, replacing them with
            # zeros in gradvec
            @inbounds @simd for j in 0:(N-1)
                m = i+j
                output[i,j] = grad(chunk_result[i], j)
                gradvec[m] = Grad(x[m], zpartials)
            end
        end
    end

    return output
end

function jacobian!{N,T,C}(f::Function, x::Vector{T}, output::Matrix, Grad::Type{GradientNum{N,T,C}})
    return jacobian!(f, x, output, similar(x, Grad))
end

function jacobian!{N,T,C}(f::Function, x::Vector{T}, gradvec::Vector{GradientNum{N,T,C}})
    return jacobian!(f, x, Array(T, length(x), N), gradvec)
end

function jacobian{N,T,C}(f::Function, x::Vector{T}, Grad::Type{GradientNum{N,T,C}})
    return jacobian!(f, x, Array(T, length(x), N), Grad)
end

function jacobian_func{G<:GradientNum}(f::Function, xlen::Int, ::Type{G}, mutates=true)
    gradvec = Vector{G}(xlen)
    T = eltype(G)
    if mutates
        jacf!{T}(x::Vector{T}, output::Matrix) = jacobian!(f, x, output, gradvec)
        return jacf!
    else
        jacf{T}(x::Vector{T}) = jacobian!(f, x, gradvec)
        return jacf
    end
end

###################
# Taking Hessians #
###################

# Hessian from ForwardDiffNum #
#-----------------------------#
function hessian!{N}(n::ForwardDiffNum{N}, output)
    @assert (N, N) == size(output)
    q = 1
    for i in 1:N
        for j in 1:i
            val = hess(n, q)
            output[i, j] = val
            output[j, i] = val
            q += 1
        end
    end
    return output
end

hessian{N,T}(n::ForwardDiffNum{N,T}) = hessian!(n, Array(T, N, N))

# Hessian from function #
#-----------------------#
function hessian!{N,T,C,S}(f::Function,
                           x::Vector{T},
                           output::Matrix{S},
                           hessvec::Vector{HessianNum{N,T,C}}) 
    xlen = length(x)
    Grad = GradientNum{N,T,C}
    ResultHessian = HessianNum{N,S,switch_eltype(C,S)}

    @assert (xlen, xlen) == size(output) "The output matrix must have size (length(input), length(input))"
    @assert xlen == length(hessvec) "The HessianNum vector must be the same length as the input vector"
    @assert xlen % N == 0 "Length of input is indivisible by the number of partials components (length(x) = $k, npartials(eltype(hessvec)) = $N)"

    pchunk = partials_chunk(Grad)
    zhess = zero_partials(eltype(hessvec))

    # We can do less work filling and
    # zeroing out hessvec if xlen == N
    if xlen == N
        @inbounds @simd for i in 1:xlen
            hessvec[i] = HessianNum(Grad(x[i], pchunk[i]), zhess)
        end

        result::ResultHessian = f(hessvec)

        hessian!(result, output)
    else
        zpartials = zero_partials(Grad)

        # load x[i]-valued HessianNums into hessvec 
        @inbounds @simd for i in 1:xlen
            hessvec[i] = HessianNum(Grad(x[i], zpartials), zhess)
        end

        for m in 1:N:xlen
            # load HessianNums with single
            # partials components into current
            # chunk of hessvec
            @inbounds @simd for i in 0:(N-1)
                k = m + i
                hessvec[k] = HessianNum(Grad(x[k], pchunk[i+1]), zhess)
            end

            chunk_result::ResultHessian = f(hessvec)

            # load resultant hessian components
            # into output, replacing them with 
            # zeros in hessvec
            q = 1
            for i in m:(m+N)
                for j in 1:i
                    k = m + q
                    val = hess(chunk_result, q)
                    output[i, j] = val
                    output[j, i] = val
                    hessvec[k] = HessianNum(Grad(x[k], zpartials), zhess)
                    q += 1
                end
            end
        end
    end

    return output
end

function hessian!{N,T,C}(f::Function, x::Vector{T}, output::Matrix, ::Type{GradientNum{N,T,C}})
    return hessian!(f, x, output, similar(x, HessianNum{N,T,C}))
end

function hessian!{N,T,C}(f::Function, x::Vector{T}, hessvec::Vector{HessianNum{N,T,C}})
    xlen = length(x)
    return hessian!(f, x, Array(T, xlen, xlen), hessvec)
end

function hessian{N,T,C}(f::Function, x::Vector{T}, ::Type{GradientNum{N,T,C}})
    xlen = length(x)
    return hessian!(f, x, Array(T, xlen, xlen), GradientNum{N,T,C})
end

function hessian_func{N,T,C}(f::Function, xlen::Int, ::Type{GradientNum{N,T,C}}, mutates=true)
    hessvec = Vector{HessianNum{N,T,C}}(xlen)
    if mutates
        hessf!{T}(x::Vector{T}, output::Matrix) = hessian!(f, x, output, hessvec)
        return hessf!
    else
        hessf{T}(x::Vector{T}) = hessian!(f, x, hessvec)
        return hessf
    end
end

##################
# Taking Tensors #
##################

# Tensor from ForwardDiffNum #
#----------------------------#
function tensor!{N,T,C}(n::ForwardDiffNum{N,T,C}, output)
    @assert (N, N, N) == size(output)
    q = 1
    for k in 1:N
        for i in k:N
            for j in k:i 
                val = tens(n, q)
                output[i, j, k] = val
                output[j, i, k] = val
                output[j, k, i] = val
                q += 1
            end
        end
    end
    return output
end

tensor{N,T,C}(n::ForwardDiffNum{N,T,C}) = tensor!(n, Array(T, N, N, N))

# Tensor from function #
#----------------------#
function tensor!{N,T,C,S}(f::Function,
                          x::Vector{T},
                          output::Array{S,3},
                          tensvec::Vector{TensorNum{N,T,C}}) 
    xlen = length(x)
    Grad = GradientNum{N,T,C}
    ResultTensor = TensorNum{N,S,switch_eltype(C,S)}

    @assert (xlen,xlen,xlen) == size(output) "The output array must have size (length(input), length(input), length(input))"
    @assert xlen == length(tensvec) "The TensorNum vector must be the same length as the input"
    @assert xlen % N == 0 "Length of input is indivisible by the number of partials components (length(x) = $k, npartials(eltype(tensvec)) = $N)"

    pchunk = partials_chunk(Grad)
    zhess = zero_partials(HessianNum{N,T,C})
    ztens = zero_partials(eltype(tensvec))

    # We can do less work filling and
    # zeroing out tensvec if xlen == N
    if xlen == N
        @inbounds @simd for i in 1:xlen
            tensvec[i] = TensorNum(HessianNum(Grad(x[i], pchunk[i]), zhess), ztens)
        end

        result::ResultTensor = f(tensvec)

        tensor!(result, output)
    else
        zpartials = zero_partials(Grad)

        # load x[i]-valued TensorNums into tensvec 
        @inbounds @simd for i in 1:xlen
            tensvec[i] = TensorNum(HessianNum(Grad(x[i], zpartials[i]), zhess), ztens)
        end

        for m in 1:N:xlen
            # load TensorNums with single
            # partials components into current
            # chunk of tensvec
            @inbounds @simd for i in 0:(N-1)
                k = m + i
                tensvec[k] = TensorNum(HessianNum(Grad(x[k], pchunk[i+1]), zhess), ztens)
            end

            chunk_result::ResultTensor = f(tensvec)

            # load resultant tensor components
            # into output, replacing them with 
            # zeros in tensvec
            q = 0
            for k in m:(m+N)
                for i in k:(m+N)
                    for j in k:i 
                        n = m + q
                        val = tens(chunk_result, q+1)
                        output[i, j, k] = val
                        output[j, i, k] = val
                        output[j, k, i] = val
                        tensvec[n] = HessianNum(Grad(x[n], zpartials), zhess)
                        q += 1
                    end
                end
            end
        end
    end

    return output
end

function tensor!{N,T,C,S}(f::Function, x::Vector{T}, output::Array{S,3}, ::Type{GradientNum{N,T,C}})
    return tensor!(f, x, output, similar(x, TensorNum{N,T,C}))
end

function tensor!{N,T,C}(f::Function, x::Vector{T}, tensvec::Vector{TensorNum{N,T,C}})
    xlen = length(x)
    return tensor!(f, x, Array(T, xlen, xlen, xlen), tensvec)
end

function tensor{N,T,C}(f::Function, x::Vector{T}, Grad::Type{GradientNum{N,T,C}})
    xlen = length(x)
    return tensor!(f, x, Array(T, xlen, xlen, xlen), Grad)
end

function tensor_func{N,T,C}(f::Function, xlen::Int, ::Type{GradientNum{N,T,C}}, mutates=true)
    tensvec = Vector{TensorNum{N,T,C}}(xlen)
    if mutates
        tensf!{T,S}(x::Vector{T}, output::Array{S,3}) = tensor!(f, x, output, tensvec)
        return tensf!
    else
        tensf{T}(x::Vector{T}) = tensor!(f, x, tensvec)
        return tensf
    end
end

####################
# Helper Functions #
####################
# Use @generated functions to essentially cache the
# zeros/partial components generated by the input type.
# This caching allows for a higher degree of efficieny 
# when calculating derivatives of f at multiple points, 
# as these values get reused rather than instantiated 
# every time.
#
# This method has the potential to incur a large memory 
# cost (and could even be considered a leak) if a 
# downstream program use many *different* partial components, 
# though I can't think of any use cases in which that would be 
# relevant.

@generated function zero_partials{N,T}(::Type{GradNumVec{N,T}})
    result = zeros(T, N)
    return :($result)
end

@generated function zero_partials{N,T}(::Type{GradNumTup{N,T}})
    z = zero(T)
    result = ntuple(i->z, Val{N})
    return :($result)
end

@generated function zero_partials{N,T,C}(::Type{HessianNum{N,T,C}})
    result = zeros(T, halfhesslen(N))
    return :($result)
end

@generated function zero_partials{N,T,C}(::Type{TensorNum{N,T,C}})
    result = zeros(T, halftenslen(N))
    return :($result) 
end

@generated function partials_chunk{N,T}(::Type{GradNumVec{N,T}})
    dus_arr = Array(Vector{T}, N)
    for i in 1:length(dus_arr)
        dus_arr[i] = setindex!(zeros(T,N), one(T), i)
    end
    return :($dus_arr)
end

@generated function partials_chunk{N,T}(::Type{GradNumTup{N,T}})
    dus_arr = Array(NTuple{N,T}, N)
    z = zero(T)
    o = one(T)
    for i in 1:length(dus_arr)
        dus_arr[i] = ntuple(x -> ifelse(x == i, o, z), Val{N})
    end
    return :($dus_arr)
end
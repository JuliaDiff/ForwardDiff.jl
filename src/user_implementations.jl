# This file contains the macros and functions that enables
# user to define custom derivatives for arbitrary functions.
# This is useful is when calling out to external libraries,
# when the function is implicitly defined or when an optimized
# version of the derivative is available.

##############
# Derivative #
##############

_propagate_user_derivative{D <: Dual}(f, Df, x::D) = Dual(f(value(x)), Df(value(x)) * partials(x))

macro implement_derivative(f, Df)
    return :($(esc(f))(x :: Dual) = _propagate_user_derivative($(esc(f)), $(esc(Df)), x))
end


############
# Gradient #
############

immutable GradientImplementConfig{T}
    input::Vector{T}
    user_gradient::Vector{T}
    current_jacobian::Matrix{T}
    updated_gradient::Vector{T}
end

getchunksize(gic::GradientImplementConfig) = length(gic.updated_gradient)

function GradientImplementConfig{T}(::Type{T}, chunk_size::Int, n_input::Int)
    GradientImplementConfig(
        zeros(T, n_input),
        zeros(T, n_input),
        zeros(T, n_input, chunk_size),
        zeros(T, chunk_size)
    )
end

GradientImplementConfig{T}(chunk_size::Int, x::Vector{T}) = GradientImplementConfig(T, chunk_size, length(x))

function _propagate_user_gradient!{D <: Dual}(f, Df!, x::Vector{D}, cfig::GradientImplementConfig)
    for i in eachindex(x)
        cfig.input[i] = value(x[i])
    end
    fv = f(cfig.input)
    Df!(cfig.user_gradient, cfig.input)
    extract_jacobian!(cfig.current_jacobian, x, npartials(D))
    At_mul_B!(cfig.updated_gradient, cfig.current_jacobian, cfig.user_gradient)
    return D(fv, Partials(totuple(D, cfig.updated_gradient)))
end

@generated function totuple{D <: Dual}(::Type{D}, v)
    return tupexpr(k -> :(v[$k]), npartials(D))
end

macro implement_gradient(f, Df)
    return quote
        function $(esc(f)){D <: Dual}(x::Vector{D})
            cfig =  GradientImplementConfig(valtype(D), npartials(D), length(x))
            Df! = (G, x) -> copy!(G, $(esc(Df))(x))
            _propagate_user_gradient!($(esc(f)), Df!, x, cfig)
        end
    end
end

macro implement_gradient!(f, Df!, cfig)
    return quote
        function $(esc(f)){D <: Dual}(x::Vector{D})
            _propagate_user_gradient!($(esc(f)), $(esc(Df!)), x, $(esc(cfig)))
        end
    end
end


#############
# Jacobian #
############

immutable JacobianImplementConfig{T}
    input::Vector{T}
    output::Vector{T}
    user_jacobian::Matrix{T}
    current_jacobian::Matrix{T}
    new_jacobian::Matrix{T}
end

function JacobianImplementConfig(::Type{T}, chunk_size::Int, n_output::Int, n_input::Int) where {T}
    JacobianImplementConfig(
        zeros(T, n_input),
        zeros(T, n_output),
        zeros(T, n_output, n_input),
        zeros(T, n_input, chunk_size),
        zeros(T, n_output, chunk_size)
    )
end

function JacobianImplementConfig(chunk_size::Int, y::Vector{T}, x::Vector{T}) where {T}
    JacobianImplementConfig(T, chunk_size, length(y), length(x))
end

function _propagate_user_jacobian!(f!, Df!, y::Vector{D}, x::Vector{D}, cfig::JacobianImplementConfig) where {D <: Dual}
    for i in eachindex(x)
        cfig.input[i] = value(x[i])
    end
    f!(cfig.output, cfig.input)
    Df!(cfig.user_jacobian, cfig.input)
    extract_jacobian!(cfig.current_jacobian, x, npartials(D))
    A_mul_B!(cfig.new_jacobian, cfig.user_jacobian, cfig.current_jacobian)
    for i in eachindex(cfig.output)
        y[i] = D(cfig.output[i], Partials(getrowpartials(D, cfig.new_jacobian, i)))
    end
    return y
end

function _propagate_user_jacobian(f, Df, x::Vector{D}) where {D <: Dual}
    input = zeros(valtype(D), length(x))
    for i in eachindex(x)
        input[i] = value(x[i])
    end
    output = f(input)
    user_jacobian = Df(input)
    current_jacobian = zeros(valtype(D), length(x), npartials(D))
    extract_jacobian!(current_jacobian, x, npartials(D))
    new_jacobian = user_jacobian * current_jacobian
    y = similar(x, length(output))
    for i in eachindex(output)
        y[i] = D(output[i], Partials(getrowpartials(D, new_jacobian, i)))
    end
    return y
end

@generated function getrowpartials{D <: Dual}(::Type{D}, J, i)
    return tupexpr(k -> :(J[i, $k]), npartials(D))
end

macro implement_jacobian(f, Df)
    return quote
        function $(esc(f)){D <: Dual}(x::Vector{D})
            _propagate_user_jacobian($(esc(f)), $(esc(Df)), x)
        end
    end
end

macro implement_jacobian!(f!, Df!, cfig)
    return quote
        function $(esc(f!)){D <: Dual}(y::Vector{D}, x::Vector{D})
            _propagate_user_jacobian!($(esc(f!)), $(esc(Df!)), y, x, $(esc(cfig)))
        end
    end
end

###########
# Hessian #
###########

# Currently not able to support this since the implementation Currently
# assumed that it is possible to evaluate gradients and jacobians of
# any functions passed to ForwardDiff.

#=
# Note that since we only ask the user to provide the Hessian it is impossible
# to know what the gradient should be. Extracting lower order results will thus
# give wrong values.

immutable HessianImplementConfig{T}
    input::Vector{T}
    user_hessian::Matrix{T}
    current_hessian::Matrix{T}
    new_hessian::Matrix{T}
end

function HessianImplementConfig{T}(::Type{T}, chunk_size::Int, n_input::Int)
    HessianImplementConfig(
        zeros(T, n_input),
        zeros(T, n_input, n_input),
        zeros(T, n_input, chunk_size),
        zeros(T, n_input, chunk_size)
    )
end

function HessianImplementConfig{T}(chunk_size::Int, x::Vector{T})
    HessianImplementConfig(T, chunk_size, length(x), length(y))
end

function _propagate_user_hessian!{D <: Dual}(f, Hf!, x::Vector{D}, cfig::HessianImplementConfig)
    for i in eachindex(x)
        cfig.input[i] = value(x[i])
    end
    fv = f(cfig.input)
    Hf!(cfig.user_hessian, cfig.input)
    extract_hessian!(cfig.current_hessian, x, npartials(D))
    A_mul_B!(cfig.new_hessian, cfig.user_hessian, cfig.current_hessian)
    return Dual(fv, create_duals(D, cfig_new_hessian)
end

function extract_hessian!(out::AbstractArray, ydual::Vector)
    for col in 1:length(ydual)
        parts = partials(ydual[col])
        for row in 1:length(ydual)
            out[col, row] = parts[row]
        end
    end
    return out
end

=#

#############################################
# ForwardDiffResult --> DiffBase.DiffResult #
#############################################

Base.@deprecate DerivativeResult(x, y) DiffBase.DiffResult(x, y)
Base.@deprecate DerivativeResult(x) DiffBase.DiffResult(copy(x), copy(x))

Base.@deprecate GradientResult(x, y) DiffBase.DiffResult(x, y)
Base.@deprecate GradientResult(x) DiffBase.GradientResult(x)

Base.@deprecate JacobianResult(x, y) DiffBase.DiffResult(x, y)
Base.@deprecate JacobianResult(x) DiffBase.JacobianResult(x)

Base.@deprecate HessianResult(x, y, z) DiffBase.DiffResult(x, y, z)
Base.@deprecate HessianResult(x) DiffBase.HessianResult(x)

immutable Chunk{N}
    function Chunk()
        Base.depwarn("Chunk{N}() is deprecated, use the ForwardDiff.Config API instead.", :Chunk)
        return new()
    end
end

export Chunk

######################
# gradient/gradient! #
######################

function gradient{N}(f, x, chunk::Chunk{N}; multithread = false, kwargs...)
    if multithread
        Base.depwarn("ForwardDiff.gradient(f, x, ::ForwardDiff.Chunk{N}; multithread = true) is deprecated" *
                     ", use ForwardDiff.gradient(f, x, ForwardDiff.Multithread(ForwardDiff.Config{N}(x))) instead.",
                     :gradient)
        return gradient(f, x, Multithread(Config{N}(x)))
    else
        Base.depwarn("ForwardDiff.gradient(f, x, ::ForwardDiff.Chunk{N}) is deprecated, use " *
                     "ForwardDiff.gradient(f, x, ForwardDiff.Config{N}(x)) instead.",
                     :gradient)
        return gradient(f, x, Config{N}(x))
    end
end

function gradient!{N}(out, f, x, chunk::Chunk{N}; multithread = false, kwargs...)
    if multithread
        Base.depwarn("ForwardDiff.gradient!(out, f, x, ::ForwardDiff.Chunk{N}; multithread = true) is deprecated" *
                     ", use ForwardDiff.gradient!(out, f, x, ForwardDiff.Multithread(ForwardDiff.Config{N}(x))) instead.",
                     :gradient!)
        return gradient!(out, f, x, Multithread(Config{N}(x)))
    else
        Base.depwarn("ForwardDiff.gradient!(out, f, x, ::ForwardDiff.Chunk{N}) is deprecated, use " *
                     "ForwardDiff.gradient!(out, f, x, ForwardDiff.Config{N}(x)) instead.",
                     :gradient!)
        return gradient!(out, f, x, Config{N}(x))
    end
end

######################
# jacobian/jacobian! #
######################

function jacobian{N}(f, x, chunk::Chunk{N}; kwargs...)
    Base.depwarn("ForwardDiff.jacobian(f, x, ::ForwardDiff.Chunk{N}) is deprecated, use " *
                 "ForwardDiff.jacobian(f, x, ForwardDiff.Config{N}(x)) instead.",
                 :jacobian)
    return jacobian(f, x, Config{N}(x))
end

function jacobian{N}(f!, y, x, chunk::Chunk{N}; kwargs...)
    Base.depwarn("ForwardDiff.jacobian(f!, y, x, ::ForwardDiff.Chunk{N}) is deprecated, use " *
                 "ForwardDiff.jacobian(f!, y, x, ForwardDiff.Config{N}(x)) instead.",
                 :jacobian)
    return jacobian(f!, y, x, Config{N}(y, x))
end

function jacobian!{N}(out, f, x, chunk::Chunk{N}; kwargs...)
    Base.depwarn("ForwardDiff.jacobian!(out, f, x, ::ForwardDiff.Chunk{N}) is deprecated, use " *
                 "ForwardDiff.jacobian!(out, f, x, ForwardDiff.Config{N}(x)) instead.",
                 :jacobian!)
    return jacobian!(out, f, x, Config{N}(x))
end

function jacobian!{N}(out, f!, y, x, chunk::Chunk{N}; kwargs...)
    Base.depwarn("ForwardDiff.jacobian!(out, f, y, x, ::ForwardDiff.Chunk{N}) is deprecated, use " *
                 "ForwardDiff.jacobian!(out, f, y, x, ForwardDiff.Config{N}(x)) instead.",
                 :jacobian!)
    return jacobian!(out, f!, y, x, Config{N}(y, x))
end

####################
# hessian/hessian! #
####################

function hessian{N}(f, x, chunk::Chunk{N}; multithread = false, kwargs...)
    if multithread
        Base.depwarn("ForwardDiff.hessian(f, x, ::ForwardDiff.Chunk{N}; multithread = true) is deprecated" *
                     ", use ForwardDiff.hessian(f, x, ForwardDiff.Multithread(ForwardDiff.Config{N}(x))) instead.",
                     :hessian)
        return hessian(f, x, Multithread(HessianConfig{N}(x)))
    else
        Base.depwarn("ForwardDiff.hessian(f, x, ::ForwardDiff.Chunk{N}) is deprecated, use " *
                     "ForwardDiff.hessian(f, x, ForwardDiff.Config{N}(x)) instead.",
                     :hessian)
        return hessian(f, x, HessianConfig{N}(x))
    end
end

function hessian!{N}(out, f, x, chunk::Chunk{N}; multithread = false, kwargs...)
    return deprecated_hessian!(out, f, x, chunk; multithread = multithread)
end

function hessian!{N}(out::DiffResult, f, x, chunk::Chunk{N}; multithread = false, kwargs...)
    return deprecated_hessian!(out, f, x, chunk; multithread = multithread)
end

function deprecated_hessian!{N}(out, f, x, chunk::Chunk{N}; multithread = false)
    if isa(out, DiffBase.DiffResult)
        out_str = "out::DiffBase.DiffResult"
        cfg_str = "ForwardDiff.HessianConfig{N}(out, x)"
        cfg = HessianConfig{N}(out, x)
    else
        out_str = "out"
        cfg_str = "ForwardDiff.HessianConfig{N}(x)"
        cfg = HessianConfig{N}(x)
    end
    if multithread
        Base.depwarn("ForwardDiff.hessian!($(out_str), f, x, ::Chunk{N}; multithread = true) is deprecated" *
                     ", use ForwardDiff.hessian!($(out_str), f, x, ForwardDiff.Multithread($(cfg_str))) instead.",
                     :hessian!)
        return hessian!(out, f, x, Multithread(cfg))
    else
        Base.depwarn("ForwardDiff.hessian!($(out_str), f, x, ::Chunk{N}) is deprecated, use " *
                     "ForwardDiff.hessian!($(out_str), f, x, $(cfg_str)) instead.",
                     :hessian!)
        return hessian!(out, f, x, cfg)
    end
end

#########################################################
# Config{N}(args...) --> Config(f, args..., Chunk{N}()) #
#########################################################

function GradientConfig{N}(x) where N
    msg = "GradientConfig{N}(x) is deprecated; use GradientConfig(nothing, x, Chunk{N}()) instead."
    Base.depwarn(msg, :GradientConfig)
    return GradientConfig(nothing, x, Chunk{N}())
end

function JacobianConfig{N}(x) where N
    msg = "JacobianConfig{N}(x) is deprecated; use JacobianConfig(nothing, x, Chunk{N}()) instead."
    Base.depwarn(msg, :JacobianConfig)
    return JacobianConfig(nothing, x, Chunk{N}())
end

function JacobianConfig{N}(y, x) where N
    msg = "JacobianConfig{N}(y, x) is deprecated; use JacobianConfig(nothing, y, x, Chunk{N}()) instead."
    Base.depwarn(msg, :JacobianConfig)
    return JacobianConfig(nothing, y, x, Chunk{N}())
end

function HessianConfig{N}(x) where N
    msg = "HessianConfig{N}(x) is deprecated; use HessianConfig(nothing, x, Chunk{N}()) instead."
    Base.depwarn(msg, :HessianConfig)
    return HessianConfig(nothing, x, Chunk{N}())
end

function HessianConfig{N}(out, x) where N
    msg = "HessianConfig{N}(out, x) is deprecated; use HessianConfig(nothing, out, x, Chunk{N}()) instead."
    Base.depwarn(msg, :HessianConfig)
    return HessianConfig(nothing, out, x, Chunk{N}())
end

function MultithreadConfig(cfg::AbstractConfig)
    msg = "MultithreadConfig(cfg) is deprecated; use cfg instead (ForwardDiff no longer implements experimental multithreading)."
    Base.depwarn(msg, :MultithreadConfig)
    return cfg
end

###############
# Derivatives #
###############


const INPUT_TYPES_DERIVATIVE = (Float32, Float64)
const FUNCTIONS_DERIVATIVE = (sawtooth, taylor_sin)


@track TRACKER "derivatives" begin
    @setup begin
        xs = [samerand(T) for T in INPUT_TYPES_DERIVATIVE]
        gs = Any[ForwardDiff.@derivative(F) for F in FUNCTIONS_DERIVATIVE]
        func_names = Dict{Function, AbstractString}()
        for F in FUNCTIONS_DERIVATIVE
            func_names[F] = string(F)
        end
    end

    @benchmarks begin
        [(:derivative, func_names[g.f], string(typeof(x))) => g(x) for g in gs, x in xs]
    end
    @tags "derivatives" "differentiation"
end

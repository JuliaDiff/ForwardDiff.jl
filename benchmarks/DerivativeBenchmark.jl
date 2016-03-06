###############
# Derivatives #
###############


const INPUT_TYPES_DERIVATIVE = (Float32, Float64)

@track TRACKER "derivatives" begin
    @setup begin
        xs = [samerand(T) for T in INPUT_TYPES_DERIVATIVE]
        gs = Any[ForwardDiff.@derivative(F) for F in TestFuncs.NUMBER_TO_NUMBER_FUNCS]
        f_strings = [string(FUNCTIONS_GRADIENT[i]) for i in 1:length(FUNCTIONS_DERIVATIVE), CS in CHUNK_SIZES_GRADIENT]
    end

    @benchmarks begin
        [(:derivative, f_strings[i], string(typeof(x))) => g(x) for (i, g) in enumerate(gs), x in xs]
    end
    @constraints time_limit=>1
    @tags "differentiation"
end

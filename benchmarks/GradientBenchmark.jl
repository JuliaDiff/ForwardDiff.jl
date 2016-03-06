#############
# Gradients #
#############

const INPUT_SIZES_GRADIENT = (2, ForwardDiff.AUTO_CHUNK_THRESHOLD, ForwardDiff.AUTO_CHUNK_THRESHOLD+1,
                              50)
const INPUT_TYPES_GRADIENT = (Float32, Float64)
const CHUNK_SIZES_GRADIENT = (1, 5, ForwardDiff.AUTO_CHUNK_THRESHOLD)

@track TRACKER "gradients" begin
    @setup begin
        vecs = [samerand(T, S) for T in INPUT_TYPES_GRADIENT, S in INPUT_SIZES_GRADIENT]
        outputs = [samerand(T, S) for T in INPUT_TYPES_GRADIENT, S in INPUT_SIZES_GRADIENT]
        gs = [ForwardDiff.@gradient(F, chunk=CS) for F in TestFuncs.VECTOR_TO_NUMBER_FUNCS, CS in CHUNK_SIZES_GRADIENT]
        gs! = [ForwardDiff.@gradient(F, chunk=CS, output_mutates = true) for F in FUNCTIONS_GRADIENT, CS in CHUNK_SIZES_GRADIENT]
        f_strings = [string(FUNCTIONS_GRADIENT[i]) for i in 1:length(FUNCTIONS_GRADIENT), CS in CHUNK_SIZES_GRADIENT]
    end

    @benchmarks begin
        [(:grad_, f_strings[i], string(typeof(vec)), size(vec)) => g(vec) for (i, g) in enumerate(gs), vec in vecs]
        [(:grad!, f_strings[i], string(typeof(vec)), size(vec)) => g!(output, vec) for (i, g!) in enumerate(gs!),
                                                                     (output, vec) in zip(outputs, vecs)]
    end
    @constraints time_limit=>1
    @tags "differentiation"
end

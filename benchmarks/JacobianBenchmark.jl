# TODO

const INPUT_SIZES_GRADIENT = (2, ForwardDiff.AUTO_CHUNK_THRESHOLD, ForwardDiff.AUTO_CHUNK_THRESHOLD+1,
                              50)
const INPUT_TYPES_GRADIENT = (Float32, Float64)
const CHUNK_SIZES_GRADIENT = (1, 5, ForwardDiff.AUTO_CHUNK_THRESHOLD)

@track TRACKER "jacobians" begin
    @setup begin
        vecs = [samerand(T, S) for T in INPUT_TYPES_GRADIENT, S in INPUT_SIZES_GRADIENT]
        outputs = [samerand(T, S) for T in INPUT_TYPES_GRADIENT, S in INPUT_SIZES_GRADIENT]
        js = [ForwardDiff.@jacobian(F, chunk=CS) for F in TestFuncs.VECTOR_TO_VECTOR_FUNCS, CS in CHUNK_SIZES_GRADIENT]
        js! = [ForwardDiff.@jacobian(F, chunk=CS, output_mutates = true) for F in TestFuncs.VECTOR_TO_VECTOR_INPLACE_FUNCS , CS in CHUNK_SIZES_GRADIENT]
        f_strings = [string(FUNCTIONS_GRADIENT[i]) for i in 1:length(FUNCTIONS_GRADIENT), CS in CHUNK_SIZES_GRADIENT]
    end

    @benchmarks begin
        [(:jac_, f_strings[i], string(typeof(vec)), size(vec)) => j(vec) for (i, j) in enumerate(js), vec in vecs]
        [(:jac!, f_strings[i], string(typeof(vec)), size(vec)) => j!(output, vec) for (i, j!) in enumerate(js!),
                                                                     (output, vec) in zip(outputs, vecs)]
    end
    @constraints time_limit=>1
    @tags "differentiation"
end

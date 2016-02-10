#############
# Gradients #
#############

const INPUT_SIZES_GRADIENT = (2, ForwardDiff.AUTO_CHUNK_THRESHOLD, ForwardDiff.AUTO_CHUNK_THRESHOLD+1,
                              50)
const INPUT_TYPES_GRADIENT = (Float32, Float64)
const CHUNK_SIZES_GRADIENT = (1, 5, ForwardDiff.AUTO_CHUNK_THRESHOLD)
const FUNCTIONS_GRADIENT = (rosenbrock, ackley, self_weighted_logit)


@track TRACKER "gradients" begin
    @setup begin
        vecs = [samerand(T, S) for T in INPUT_TYPES_GRADIENT, S in INPUT_SIZES_GRADIENT]
        gs = [ForwardDiff.@gradient(F, chunk=CS) for F in FUNCTIONS_GRADIENT, CS in CHUNK_SIZES_GRADIENT]
        names = Dict{Function, AbstractString}()
        for F in FUNCTIONS_GRADIENT
            names[F] = string(F)
        end
    end

    @benchmarks begin
        [(:grad_unmod, names[g.f], string(typeof(vec)), size(vec)) => g(vec) for g in gs, vec in vecs]
    end
    @tags "gradients" "differentiation"
end

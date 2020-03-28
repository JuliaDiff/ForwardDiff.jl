import ForwardDiff
using ForwardDiff: DEFAULT_CHUNK_THRESHOLD
using Test
using Random

# seed RNG, thus making result inaccuracies deterministic
# so we don't have to retune EPS for arbitrary inputs
Random.seed!(1)

const XLEN = DEFAULT_CHUNK_THRESHOLD + 1
const YLEN = div(DEFAULT_CHUNK_THRESHOLD, 2) + 1
const X, Y = rand(XLEN), rand(YLEN)
const CHUNK_SIZES = (1, div(DEFAULT_CHUNK_THRESHOLD, 3), div(DEFAULT_CHUNK_THRESHOLD, 2), DEFAULT_CHUNK_THRESHOLD, DEFAULT_CHUNK_THRESHOLD + 1)
const HESSIAN_CHUNK_SIZES = (1, 2, 3)
const FINITEDIFF_ERROR = 3e-5

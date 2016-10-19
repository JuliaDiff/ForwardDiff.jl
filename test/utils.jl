import ForwardDiff
using ForwardDiff.MAX_CHUNK_SIZE
using Base.Test

const XLEN = MAX_CHUNK_SIZE
const YLEN = div(MAX_CHUNK_SIZE, 2) + 1
const X, Y = rand(XLEN), rand(YLEN)
const CHUNK_SIZES = (1, div(MAX_CHUNK_SIZE, 3), div(MAX_CHUNK_SIZE, 2), MAX_CHUNK_SIZE)
const FINITEDIFF_ERROR = 1e-5

# used to test against results calculated via finite difference
test_approx_eps(a::Array, b::Array) = @test_approx_eq_eps a b EPS

# seed RNG, thus making result inaccuracies deterministic
# so we don't have to retune EPS for arbitrary inputs
srand(1)

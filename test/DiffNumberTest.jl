module DiffNumberTest

using Base.Test
using ForwardDiff
using ForwardDiff: Partials, DiffNumber, value, partials

import NaNMath
import Calculus

const N = 3
const M = 4
const T = Float32

const PARTIALS = Partials(ntuple(n -> rand(T), Val{N}))
const PRIMAL = rand(T)
const FDNUM = DiffNumber(PRIMAL, PARTIALS)

const PARTIALS2 = Partials(ntuple(n -> rand(T), Val{N}))
const PRIMAL2 = rand(T)
const FDNUM2 = DiffNumber(PRIMAL2, PARTIALS2)

const M_PARTIALS = Partials(ntuple(m -> rand(T), Val{M}))
const NESTED_PARTIALS = convert(Partials{N,DiffNumber{M,T}}, PARTIALS)
const NESTED_FDNUM = DiffNumber(DiffNumber(PRIMAL, M_PARTIALS), NESTED_PARTIALS)

const M_PARTIALS2 = Partials(ntuple(m -> rand(T), Val{M}))
const NESTED_PARTIALS2 = convert(Partials{N,DiffNumber{M,T}}, PARTIALS2)
const NESTED_FDNUM2 = DiffNumber(DiffNumber(PRIMAL2, M_PARTIALS2), NESTED_PARTIALS2)

samerng() = MersenneTwister(1)

################
# Constructors #
################

@test DiffNumber(PRIMAL, PARTIALS...) === FDNUM
@test typeof(DiffNumber(widen(T)(PRIMAL), PARTIALS)) === DiffNumber{N,widen(T)}
@test typeof(DiffNumber(widen(T)(PRIMAL), PARTIALS.values)) === DiffNumber{N,widen(T)}
@test typeof(DiffNumber(widen(T)(PRIMAL), PARTIALS...)) === DiffNumber{N,widen(T)}
@test typeof(NESTED_FDNUM) == DiffNumber{N,DiffNumber{M,T}}

#############
# Accessors #
#############

@test value(PRIMAL) == PRIMAL
@test value(FDNUM) == PRIMAL
@test value(NESTED_FDNUM) === DiffNumber(PRIMAL, M_PARTIALS)

@test partials(PRIMAL) == Partials{0,T}(tuple())
@test partials(FDNUM) == PARTIALS
@test partials(NESTED_FDNUM) === NESTED_PARTIALS

for i in 1:N
    @test partials(FDNUM, i) == PARTIALS[i]
    for j in 1:M
        @test partials(NESTED_FDNUM, i, j) == partials(NESTED_PARTIALS[i], j)
    end
end

@test ForwardDiff.npartials(FDNUM) == N
@test ForwardDiff.npartials(typeof(FDNUM)) == N
@test ForwardDiff.npartials(NESTED_FDNUM) == N
@test ForwardDiff.npartials(typeof(NESTED_FDNUM)) == N

@test ForwardDiff.numtype(FDNUM) == T
@test ForwardDiff.numtype(typeof(FDNUM)) == T
@test ForwardDiff.numtype(NESTED_FDNUM) == DiffNumber{M,T}
@test ForwardDiff.numtype(typeof(NESTED_FDNUM)) == DiffNumber{M,T}

#####################
# Generic Functions #
#####################

@test FDNUM === copy(FDNUM)
@test NESTED_FDNUM === copy(NESTED_FDNUM)

@test eps(FDNUM) === eps(PRIMAL)
@test eps(typeof(FDNUM)) === eps(T)
@test eps(NESTED_FDNUM) === eps(PRIMAL)
@test eps(typeof(NESTED_FDNUM)) === eps(T)

@test floor(Int, FDNUM) === floor(Int, PRIMAL)
@test floor(Int, FDNUM2) === floor(Int, PRIMAL2)
@test floor(Int, NESTED_FDNUM) === floor(Int, PRIMAL)

@test ceil(Int, FDNUM) === ceil(Int, PRIMAL)
@test ceil(Int, FDNUM2) === ceil(Int, PRIMAL2)
@test ceil(Int, NESTED_FDNUM) === ceil(Int, PRIMAL)

@test trunc(Int, FDNUM) === trunc(Int, PRIMAL)
@test trunc(Int, FDNUM2) === trunc(Int, PRIMAL2)
@test trunc(Int, NESTED_FDNUM) === trunc(Int, PRIMAL)

@test round(Int, FDNUM) === round(Int, PRIMAL)
@test round(Int, FDNUM2) === round(Int, PRIMAL2)
@test round(Int, NESTED_FDNUM) === round(Int, PRIMAL)

@test hash(FDNUM) === hash(PRIMAL)
@test hash(FDNUM, hash(PRIMAL)) === hash(PRIMAL, hash(PRIMAL))
@test hash(NESTED_FDNUM) === hash(PRIMAL)
@test hash(NESTED_FDNUM, hash(PRIMAL)) === hash(PRIMAL, hash(PRIMAL))

const TMPIO = IOBuffer()
write(TMPIO, FDNUM)
seekstart(TMPIO)
@test read(TMPIO, typeof(FDNUM)) === FDNUM
seekstart(TMPIO)
write(TMPIO, FDNUM2)
seekstart(TMPIO)
@test read(TMPIO, typeof(FDNUM2)) === FDNUM2
seekstart(TMPIO)
write(TMPIO, NESTED_FDNUM)
seekstart(TMPIO)
@test read(TMPIO, typeof(NESTED_FDNUM)) === NESTED_FDNUM
close(TMPIO)

@test zero(FDNUM) === DiffNumber(zero(PRIMAL), zero(PARTIALS))
@test zero(typeof(FDNUM)) === DiffNumber(zero(T), zero(Partials{N,T}))
@test zero(NESTED_FDNUM) === DiffNumber(DiffNumber(zero(PRIMAL), zero(M_PARTIALS)), zero(NESTED_PARTIALS))
@test zero(typeof(NESTED_FDNUM)) === DiffNumber(DiffNumber(zero(T), zero(Partials{M,T})), zero(Partials{N,DiffNumber{M,T}}))

@test one(FDNUM) === DiffNumber(one(PRIMAL), zero(PARTIALS))
@test one(typeof(FDNUM)) === DiffNumber(one(T), zero(Partials{N,T}))
@test one(NESTED_FDNUM) === DiffNumber(DiffNumber(one(PRIMAL), zero(M_PARTIALS)), zero(NESTED_PARTIALS))
@test one(typeof(NESTED_FDNUM)) === DiffNumber(DiffNumber(one(T), zero(Partials{M,T})), zero(Partials{N,DiffNumber{M,T}}))

@test rand(samerng(), FDNUM) === DiffNumber(rand(samerng(), T), zero(PARTIALS))
@test rand(samerng(), typeof(FDNUM)) === DiffNumber(rand(samerng(), T), zero(Partials{N,T}))
@test rand(samerng(), NESTED_FDNUM) === DiffNumber(DiffNumber(rand(samerng(), T), zero(M_PARTIALS)), zero(NESTED_PARTIALS))
@test rand(samerng(), typeof(NESTED_FDNUM)) === DiffNumber(DiffNumber(rand(samerng(), T), zero(Partials{M,T})), zero(Partials{N,DiffNumber{M,T}}))

# Predicates #
#------------#

@test ForwardDiff.isconstant(zero(FDNUM))
@test ForwardDiff.isconstant(rand(FDNUM))
@test ForwardDiff.isconstant(one(FDNUM))
@test !(ForwardDiff.isconstant(FDNUM))

@test ForwardDiff.isconstant(zero(NESTED_FDNUM))
@test ForwardDiff.isconstant(rand(NESTED_FDNUM))
@test ForwardDiff.isconstant(one(NESTED_FDNUM))
@test !(ForwardDiff.isconstant(NESTED_FDNUM))

@test isequal(FDNUM, DiffNumber(PRIMAL, PARTIALS2))
@test !(isequal(FDNUM, FDNUM2))

@test isequal(NESTED_FDNUM, DiffNumber(DiffNumber(PRIMAL, M_PARTIALS2), NESTED_PARTIALS2))
@test !(isequal(NESTED_FDNUM, NESTED_FDNUM2))

@test FDNUM == DiffNumber(PRIMAL, PARTIALS2)
@test FDNUM != FDNUM2
@test NESTED_FDNUM != NESTED_FDNUM2

@test isless(DiffNumber(1, PARTIALS), DiffNumber(2, PARTIALS2))
@test !(isless(DiffNumber(1, PARTIALS), DiffNumber(1, PARTIALS2)))
@test !(isless(DiffNumber(2, PARTIALS), DiffNumber(1, PARTIALS2)))

@test isless(DiffNumber(DiffNumber(1, M_PARTIALS), NESTED_PARTIALS), DiffNumber(DiffNumber(2, M_PARTIALS2), NESTED_PARTIALS2))
@test !(isless(DiffNumber(DiffNumber(1, M_PARTIALS), NESTED_PARTIALS), DiffNumber(DiffNumber(1, M_PARTIALS2), NESTED_PARTIALS2)))
@test !(isless(DiffNumber(DiffNumber(2, M_PARTIALS), NESTED_PARTIALS), DiffNumber(DiffNumber(1, M_PARTIALS2), NESTED_PARTIALS2)))

@test DiffNumber(1, PARTIALS) < DiffNumber(2, PARTIALS2)
@test !(DiffNumber(1, PARTIALS) < DiffNumber(1, PARTIALS2))
@test !(DiffNumber(2, PARTIALS) < DiffNumber(1, PARTIALS2))

@test DiffNumber(DiffNumber(1, M_PARTIALS), NESTED_PARTIALS) < DiffNumber(DiffNumber(2, M_PARTIALS2), NESTED_PARTIALS2)
@test !(DiffNumber(DiffNumber(1, M_PARTIALS), NESTED_PARTIALS) < DiffNumber(DiffNumber(1, M_PARTIALS2), NESTED_PARTIALS2))
@test !(DiffNumber(DiffNumber(2, M_PARTIALS), NESTED_PARTIALS) < DiffNumber(DiffNumber(1, M_PARTIALS2), NESTED_PARTIALS2))

@test DiffNumber(1, PARTIALS) <= DiffNumber(2, PARTIALS2)
@test DiffNumber(1, PARTIALS) <= DiffNumber(1, PARTIALS2)
@test !(DiffNumber(2, PARTIALS) <= DiffNumber(1, PARTIALS2))

@test DiffNumber(DiffNumber(1, M_PARTIALS), NESTED_PARTIALS) <= DiffNumber(DiffNumber(2, M_PARTIALS2), NESTED_PARTIALS2)
@test DiffNumber(DiffNumber(1, M_PARTIALS), NESTED_PARTIALS) <= DiffNumber(DiffNumber(1, M_PARTIALS2), NESTED_PARTIALS2)
@test !(DiffNumber(DiffNumber(2, M_PARTIALS), NESTED_PARTIALS) <= DiffNumber(DiffNumber(1, M_PARTIALS2), NESTED_PARTIALS2))

@test DiffNumber(2, PARTIALS) > DiffNumber(1, PARTIALS2)
@test !(DiffNumber(1, PARTIALS) > DiffNumber(1, PARTIALS2))
@test !(DiffNumber(1, PARTIALS) > DiffNumber(2, PARTIALS2))

@test DiffNumber(DiffNumber(2, M_PARTIALS), NESTED_PARTIALS) > DiffNumber(DiffNumber(1, M_PARTIALS2), NESTED_PARTIALS2)
@test !(DiffNumber(DiffNumber(1, M_PARTIALS), NESTED_PARTIALS) > DiffNumber(DiffNumber(1, M_PARTIALS2), NESTED_PARTIALS2))
@test !(DiffNumber(DiffNumber(1, M_PARTIALS), NESTED_PARTIALS) > DiffNumber(DiffNumber(2, M_PARTIALS2), NESTED_PARTIALS2))

@test DiffNumber(2, PARTIALS) >= DiffNumber(1, PARTIALS2)
@test DiffNumber(1, PARTIALS) >= DiffNumber(1, PARTIALS2)
@test !(DiffNumber(1, PARTIALS) >= DiffNumber(2, PARTIALS2))

@test DiffNumber(DiffNumber(2, M_PARTIALS), NESTED_PARTIALS) >= DiffNumber(DiffNumber(1, M_PARTIALS2), NESTED_PARTIALS2)
@test DiffNumber(DiffNumber(1, M_PARTIALS), NESTED_PARTIALS) >= DiffNumber(DiffNumber(1, M_PARTIALS2), NESTED_PARTIALS2)
@test !(DiffNumber(DiffNumber(1, M_PARTIALS), NESTED_PARTIALS) >= DiffNumber(DiffNumber(2, M_PARTIALS2), NESTED_PARTIALS2))

@test isnan(DiffNumber(NaN, PARTIALS))
@test !(isnan(FDNUM))

@test isnan(DiffNumber(DiffNumber(NaN, M_PARTIALS), NESTED_PARTIALS))
@test !(isnan(NESTED_FDNUM))

@test isfinite(FDNUM)
@test !(isfinite(DiffNumber(Inf, PARTIALS)))

@test isfinite(NESTED_FDNUM)
@test !(isfinite(DiffNumber(DiffNumber(NaN, M_PARTIALS), NESTED_PARTIALS)))

@test isinf(DiffNumber(Inf, PARTIALS))
@test !(isinf(FDNUM))

@test isinf(DiffNumber(DiffNumber(Inf, M_PARTIALS), NESTED_PARTIALS))
@test !(isinf(NESTED_FDNUM))

@test isreal(FDNUM)
@test isreal(NESTED_FDNUM)

@test isinteger(DiffNumber(1.0, PARTIALS))
@test !(isinteger(FDNUM))

@test isinteger(DiffNumber(DiffNumber(1.0, M_PARTIALS), NESTED_PARTIALS))
@test !(isinteger(NESTED_FDNUM))

@test iseven(DiffNumber(2))
@test !(iseven(DiffNumber(1)))

@test iseven(DiffNumber(DiffNumber(2)))
@test !(iseven(DiffNumber(DiffNumber(1))))

@test isodd(DiffNumber(1))
@test !(isodd(DiffNumber(2)))

@test isodd(DiffNumber(DiffNumber(1)))
@test !(isodd(DiffNumber(DiffNumber(2))))

########################
# Promotion/Conversion #
########################

const WIDE_T = widen(T)

@test promote_type(DiffNumber{N,T}, T) == DiffNumber{N,T}
@test promote_type(DiffNumber{N,T}, WIDE_T) == DiffNumber{N,WIDE_T}
@test promote_type(DiffNumber{N,WIDE_T}, T) == DiffNumber{N,WIDE_T}
@test promote_type(DiffNumber{N,T}, DiffNumber{N,T}) == DiffNumber{N,T}
@test promote_type(DiffNumber{N,T}, DiffNumber{N,WIDE_T}) == DiffNumber{N,WIDE_T}
@test promote_type(DiffNumber{N,WIDE_T}, DiffNumber{N,DiffNumber{M,T}}) == DiffNumber{N,DiffNumber{M,WIDE_T}}

const WIDE_FDNUM = convert(DiffNumber{N,WIDE_T}, FDNUM)
const WIDE_NESTED_FDNUM = convert(DiffNumber{N,DiffNumber{M,WIDE_T}}, NESTED_FDNUM)

@test typeof(WIDE_FDNUM) == DiffNumber{N,WIDE_T}
@test typeof(WIDE_NESTED_FDNUM) == DiffNumber{N,DiffNumber{M,WIDE_T}}

@test value(WIDE_FDNUM) == PRIMAL
@test value(WIDE_NESTED_FDNUM) == PRIMAL

@test convert(DiffNumber, FDNUM) === FDNUM
@test convert(DiffNumber, NESTED_FDNUM) === NESTED_FDNUM
@test convert(DiffNumber{N,T}, FDNUM) === FDNUM
@test convert(DiffNumber{N,DiffNumber{M,T}}, NESTED_FDNUM) === NESTED_FDNUM
@test convert(DiffNumber{N,WIDE_T}, PRIMAL) === DiffNumber(WIDE_T(PRIMAL), zero(Partials{N,WIDE_T}))
@test convert(DiffNumber{N,DiffNumber{M,WIDE_T}}, PRIMAL) === DiffNumber(DiffNumber(WIDE_T(PRIMAL), zero(Partials{M,WIDE_T})), zero(Partials{N,DiffNumber{M,T}}))
@test convert(DiffNumber{N,DiffNumber{M,T}}, FDNUM) === DiffNumber(DiffNumber{M,T}(PRIMAL), convert(Partials{N,DiffNumber{M,T}}, PARTIALS))
@test convert(DiffNumber{N,DiffNumber{M,WIDE_T}}, FDNUM) === DiffNumber(DiffNumber{M,WIDE_T}(PRIMAL), convert(Partials{N,DiffNumber{M,WIDE_T}}, PARTIALS))

########
# Math #
########

test_approx_diffnums(a::Real, b::Real) = @test_approx_eq a b

function test_approx_diffnums{N}(a::DiffNumber{N}, b::DiffNumber{N})
    test_approx_diffnums(value(a), value(b))
    for i in 1:N
        test_approx_diffnums(partials(a)[i], partials(b)[i])
    end
end

# Arithmetic #
#------------#

@test FDNUM + FDNUM2 === DiffNumber(value(FDNUM) + value(FDNUM2), partials(FDNUM) + partials(FDNUM2))
@test FDNUM + PRIMAL === DiffNumber(value(FDNUM) + PRIMAL, partials(FDNUM))
@test PRIMAL + FDNUM === DiffNumber(value(FDNUM) + PRIMAL, partials(FDNUM))

@test NESTED_FDNUM + NESTED_FDNUM2 === DiffNumber(value(NESTED_FDNUM) + value(NESTED_FDNUM2), partials(NESTED_FDNUM) + partials(NESTED_FDNUM2))
@test NESTED_FDNUM + PRIMAL === DiffNumber(value(NESTED_FDNUM) + PRIMAL, partials(NESTED_FDNUM))
@test PRIMAL + NESTED_FDNUM === DiffNumber(value(NESTED_FDNUM) + PRIMAL, partials(NESTED_FDNUM))

@test FDNUM - FDNUM2 === DiffNumber(value(FDNUM) - value(FDNUM2), partials(FDNUM) - partials(FDNUM2))
@test FDNUM - PRIMAL === DiffNumber(value(FDNUM) - PRIMAL, partials(FDNUM))
@test PRIMAL - FDNUM === DiffNumber(PRIMAL - value(FDNUM), -(partials(FDNUM)))
@test -(FDNUM) === DiffNumber(-(value(FDNUM)), -(partials(FDNUM)))

@test NESTED_FDNUM - NESTED_FDNUM2 === DiffNumber(value(NESTED_FDNUM) - value(NESTED_FDNUM2), partials(NESTED_FDNUM) - partials(NESTED_FDNUM2))
@test NESTED_FDNUM - PRIMAL === DiffNumber(value(NESTED_FDNUM) - PRIMAL, partials(NESTED_FDNUM))
@test PRIMAL - NESTED_FDNUM === DiffNumber(PRIMAL - value(NESTED_FDNUM), -(partials(NESTED_FDNUM)))
@test -(NESTED_FDNUM) === DiffNumber(-(value(NESTED_FDNUM)), -(partials(NESTED_FDNUM)))

@test FDNUM * FDNUM2 === DiffNumber(value(FDNUM) * value(FDNUM2), ForwardDiff._mul_partials(partials(FDNUM), partials(FDNUM2), value(FDNUM2), value(FDNUM)))
@test FDNUM * PRIMAL === DiffNumber(value(FDNUM) * PRIMAL, partials(FDNUM) * PRIMAL)
@test PRIMAL * FDNUM === DiffNumber(value(FDNUM) * PRIMAL, partials(FDNUM) * PRIMAL)

@test NESTED_FDNUM * NESTED_FDNUM2 === DiffNumber(value(NESTED_FDNUM) * value(NESTED_FDNUM2), ForwardDiff._mul_partials(partials(NESTED_FDNUM), partials(NESTED_FDNUM2), value(NESTED_FDNUM2), value(NESTED_FDNUM)))
@test NESTED_FDNUM * PRIMAL === DiffNumber(value(NESTED_FDNUM) * PRIMAL, partials(NESTED_FDNUM) * PRIMAL)
@test PRIMAL * NESTED_FDNUM === DiffNumber(value(NESTED_FDNUM) * PRIMAL, partials(NESTED_FDNUM) * PRIMAL)

test_approx_diffnums(FDNUM / FDNUM2, DiffNumber(value(FDNUM) / value(FDNUM2), ForwardDiff._div_partials(partials(FDNUM), partials(FDNUM2), value(FDNUM), value(FDNUM2))))
test_approx_diffnums(FDNUM / PRIMAL, DiffNumber(value(FDNUM) / PRIMAL, partials(FDNUM) / PRIMAL))
test_approx_diffnums(PRIMAL / FDNUM, DiffNumber(PRIMAL / value(FDNUM), (-(PRIMAL) / value(FDNUM)^2) * partials(FDNUM)))

test_approx_diffnums(NESTED_FDNUM / NESTED_FDNUM2, DiffNumber(value(NESTED_FDNUM) / value(NESTED_FDNUM2), ForwardDiff._div_partials(partials(NESTED_FDNUM), partials(NESTED_FDNUM2), value(NESTED_FDNUM), value(NESTED_FDNUM2))))
test_approx_diffnums(NESTED_FDNUM / PRIMAL, DiffNumber(value(NESTED_FDNUM) / PRIMAL, partials(NESTED_FDNUM) / PRIMAL))
test_approx_diffnums(PRIMAL / NESTED_FDNUM, DiffNumber(PRIMAL / value(NESTED_FDNUM), (-(PRIMAL) / value(NESTED_FDNUM)^2) * partials(NESTED_FDNUM)))

test_approx_diffnums(FDNUM^FDNUM2, exp(FDNUM2 * log(FDNUM)))
test_approx_diffnums(FDNUM^PRIMAL, exp(PRIMAL * log(FDNUM)))
test_approx_diffnums(PRIMAL^FDNUM, exp(FDNUM * log(PRIMAL)))

test_approx_diffnums(NESTED_FDNUM^NESTED_FDNUM2, exp(NESTED_FDNUM2 * log(NESTED_FDNUM)))
test_approx_diffnums(NESTED_FDNUM^PRIMAL, exp(PRIMAL * log(NESTED_FDNUM)))
test_approx_diffnums(PRIMAL^NESTED_FDNUM, exp(NESTED_FDNUM * log(PRIMAL)))

@test partials(NaNMath.pow(DiffNumber(-2.0, 1.0), DiffNumber(2.0, 0.0)), 1) == -4.0

# Unary Functions #
#-----------------#

@test conj(FDNUM) === FDNUM
@test conj(NESTED_FDNUM) === NESTED_FDNUM
@test transpose(FDNUM) === FDNUM
@test transpose(NESTED_FDNUM) === NESTED_FDNUM
@test ctranspose(FDNUM) === FDNUM
@test ctranspose(NESTED_FDNUM) === NESTED_FDNUM

@test abs(-FDNUM) === FDNUM
@test abs(FDNUM) === FDNUM
@test abs(-NESTED_FDNUM) === NESTED_FDNUM
@test abs(NESTED_FDNUM) === NESTED_FDNUM

const UNSUPPORTED_NESTED_FUNCS = (:trigamma, :airyprime, :besselj1, :bessely1)
const DOMAIN_ERR_FUNCS = (:asec, :acsc, :asecd, :acscd, :acoth, :acosh)

for fsym in ForwardDiff.AUTO_DEFINED_UNARY_FUNCS
    try
        v = :v
        deriv = Calculus.differentiate(:($(fsym)($v)), v)
        @eval begin
            is_domain_err_func = $(fsym in DOMAIN_ERR_FUNCS)
            is_nanmath_func = $(fsym in ForwardDiff.NANMATH_FUNCS)
            is_unsupported_nested_func = $(fsym in UNSUPPORTED_NESTED_FUNCS)

            fdnum = is_domain_err_func ? FDNUM + 1 : FDNUM
            $(v) = value(fdnum)
            test_approx_diffnums($(fsym)(fdnum), DiffNumber($(fsym)($v), $(deriv) * partials(fdnum)))
            if is_nanmath_func
                test_approx_diffnums(NaNMath.$(fsym)(fdnum), DiffNumber(NaNMath.$(fsym)($v), $(deriv) * partials(fdnum)))
            end

            if !(is_unsupported_nested_func)
                nested_fdnum = is_domain_err_func ? NESTED_FDNUM + 1 : NESTED_FDNUM
                $(v) = value(nested_fdnum)
                test_approx_diffnums($(fsym)(nested_fdnum), DiffNumber($(fsym)($v), $(deriv) * partials(nested_fdnum)))
                if is_nanmath_func
                    test_approx_diffnums(NaNMath.$(fsym)(nested_fdnum), DiffNumber(NaNMath.$(fsym)($v), $(deriv) * partials(nested_fdnum)))
                end
            end
        end
    catch err
        warn("Encountered error when testing $(fsym)(::DiffNumber):")
        throw(err)
    end
end

end # module

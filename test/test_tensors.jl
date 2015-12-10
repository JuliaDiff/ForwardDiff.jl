using Base.Test
using Calculus
using ForwardDiff
using ForwardDiff:
        GradientNumber,
        HessianNumber,
        TensorNumber,
        value,
        grad,
        hess,
        tens,
        npartials,
        isconstant,
        hessnum,
        hess_inds

floatrange = .01:.01:.99
intrange = 1:10
N = 4
T = Float64
C = NTuple{N,T}
hessveclen = ForwardDiff.halfhesslen(N)
tensveclen = ForwardDiff.halftenslen(N)

test_val = rand(floatrange)
test_partials = tuple(rand(floatrange, N)...)
test_hessvec = rand(floatrange, hessveclen)
test_tensvec = rand(floatrange, tensveclen)

test_grad = GradientNumber(test_val, test_partials)
test_hess = HessianNumber(test_grad, test_hessvec)
test_tens = TensorNumber(test_hess, test_tensvec)

######################
# Accessor Functions #
######################
@test value(test_tens) == test_val
@test grad(test_tens) == test_partials
@test hess(test_tens) == test_hessvec
@test tens(test_tens) == test_tensvec

for i in 1:N
    @test grad(test_tens, i) == test_partials[i]
end

for i in 1:hessveclen
    @test hess(test_tens, i) == test_hessvec[i]
end

for i in 1:tensveclen
    @test tens(test_tens, i) == test_tensvec[i]
end

@test npartials(test_tens) == npartials(typeof(test_tens)) == N

##################################
# Value Representation Functions #
##################################
@test eps(test_tens) == eps(test_val)
@test eps(typeof(test_tens)) == eps(T)

tens_zero = TensorNumber(zero(test_hess), map(zero, test_tensvec))
tens_one = TensorNumber(one(test_hess), map(zero, test_tensvec))

@test zero(test_tens) == tens_zero
@test zero(typeof(test_tens)) == tens_zero

@test one(test_tens) == tens_one
@test one(typeof(test_tens)) == tens_one

#########################################
# Conversion/Promotion/Hashing/Equality #
#########################################
int_val = round(Int, test_val)
int_partials = map(x -> round(Int, x), test_partials)
int_hessvec = map(x -> round(Int, x), test_hessvec)
int_tensvec = map(x -> round(Int, x), test_tensvec)

float_val = float(int_val)
float_partials = map(float, int_partials)
float_hessvec= map(float, int_hessvec)
float_tensvec= map(float, int_tensvec)

int_tens = TensorNumber(HessianNumber(GradientNumber(int_val, int_partials), int_hessvec), int_tensvec)
float_tens = TensorNumber(HessianNumber(GradientNumber(float_val, float_partials), float_hessvec), float_tensvec)
const_tens = TensorNumber{N,T,C}(float_val)

@test convert(typeof(test_tens), test_tens) == test_tens
@test convert(TensorNumber, test_tens) == test_tens
@test convert(TensorNumber{N,T,C}, int_tens) == float_tens
@test convert(TensorNumber{3,T,NTuple{3,T}}, 1) == TensorNumber{3,T,NTuple{3,T}}(1.0)
@test convert(T, TensorNumber(HessianNumber(GradientNumber(1, tuple(0, 0))))) == 1.0

@test float(int_tens) == float_tens

IntTens = TensorNumber{N,Int,NTuple{N,Int}}
FloatTens = TensorNumber{N,Float64,NTuple{N,Float64}}

@test promote_type(IntTens, IntTens) == IntTens
@test promote_type(FloatTens, IntTens) == FloatTens
@test promote_type(IntTens, Float64) == FloatTens
@test promote_type(FloatTens, Int) == FloatTens

@test hash(int_tens) == hash(float_tens)
@test hash(const_tens) == hash(float_val)

@test int_tens == float_tens
@test float_val == const_tens
@test const_tens == float_val

@test isequal(int_tens, float_tens)
@test isequal(float_val, const_tens)
@test isequal(const_tens, float_val)

@test copy(test_tens) == test_tens

####################
# is____ Functions #
####################
@test isnan(test_tens) == isnan(test_val)
@test isnan(TensorNumber{N,T,C}(NaN))

not_const_tens = TensorNumber(HessianNumber(GradientNumber(one(T), map(one, test_partials))))
@test !(isconstant(not_const_tens))
@test !(isreal(not_const_tens))
@test isconstant(const_tens) && isreal(const_tens)
@test isconstant(zero(not_const_tens)) && isreal(zero(not_const_tens))

inf_tens = TensorNumber{N,T,C}(Inf)
@test isfinite(test_tens) == isfinite(test_val)
@test !isfinite(inf_tens)

@test isinf(inf_tens)
@test !(isinf(test_tens))

@test isless(test_tens-1, test_tens)
@test test_tens-1 < test_tens
@test test_tens > test_tens-1

@test isless(test_val-1, test_tens)
@test test_val-1 < test_tens
@test test_tens > test_val-1

@test isless(test_tens, test_val+1)
@test test_tens < test_val+1
@test test_val+1 > test_tens

#######
# I/O #
#######
io = IOBuffer()
write(io, test_tens)
seekstart(io)

@test read(io, typeof(test_tens)) == test_tens

close(io)

##############
# Math tests #
##############
rand_val = rand(floatrange)
rand_partials = map(x -> rand(floatrange), test_partials)
rand_hessvec = map(x -> rand(floatrange), test_hessvec)
rand_tensvec = map(x -> rand(floatrange), test_tensvec)

rand_grad = GradientNumber(rand_val, rand_partials)
rand_hess = HessianNumber(rand_grad, rand_hessvec)
rand_tens = TensorNumber(rand_hess, rand_tensvec)

# Addition/Subtraction #
#----------------------#
@test rand_tens + test_tens == TensorNumber(rand_hess + test_hess, rand_tensvec + test_tensvec)
@test rand_tens + test_tens == test_tens + rand_tens
@test rand_tens - test_tens == TensorNumber(rand_hess - test_hess, rand_tensvec - test_tensvec)

@test rand_val + test_tens == TensorNumber(rand_val + test_hess, test_tensvec)
@test rand_val + test_tens == test_tens + rand_val
@test rand_val - test_tens == TensorNumber(rand_val - test_hess, -test_tensvec)
@test test_tens - rand_val == TensorNumber(test_hess - rand_val, test_tensvec)

@test -test_tens == TensorNumber(-test_hess, -test_tensvec)

# Multiplication/Division #
#-------------------------#
function tens_approx_eq(a::TensorNumber, b::TensorNumber)
    eps = 1e-9
    try
        @test_approx_eq_eps value(a) value(b) eps
        @test_approx_eq_eps collect(grad(a)) collect(grad(b)) eps
        @test_approx_eq_eps hess(a) hess(b) eps
        @test_approx_eq_eps tens(a) tens(b) eps
    catch err
        error("Failure: TensorNumber a and TensorNumber b should be equal.\n Error: $err\n rand_tens = $rand_tens\n test_tens = $test_tens")
    end
end

@test hessnum(rand_tens * test_tens) == rand_hess * test_hess

@test rand_val * test_tens == TensorNumber(rand_val * test_hess, rand_val * test_tensvec)
@test test_tens * rand_val == rand_val * test_tens

@test test_tens * true == test_tens
@test true * test_tens == test_tens * true
@test test_tens * false == zero(test_tens)
@test false * test_tens == test_tens * false

tens_approx_eq(rand_tens, rand_tens * (test_tens/test_tens))
tens_approx_eq(rand_tens, test_tens * (rand_tens/test_tens))
tens_approx_eq(rand_tens, (rand_tens * test_tens)/test_tens)
tens_approx_eq(test_tens, (rand_tens * test_tens)/rand_tens)

tens_approx_eq(2 * test_tens, test_tens + test_tens)
tens_approx_eq(test_tens * inv(test_tens), one(test_tens))

tens_approx_eq(rand_val / test_tens, rand_val * inv(test_tens))
tens_approx_eq(rand_val / test_tens, rand_val * 1/test_tens)
tens_approx_eq(rand_tens / test_tens, rand_tens * inv(test_tens))
tens_approx_eq(rand_tens / test_tens, rand_tens * 1/test_tens)
tens_approx_eq(test_tens / test_tens, one(test_tens))

@test test_tens / rand_val == TensorNumber(test_hess / rand_val, test_tensvec / rand_val)

# Exponentiation #
#----------------#
tens_approx_eq(test_tens^rand_tens, exp(rand_tens * log(test_tens)))
tens_approx_eq(test_tens^rand_val, exp(rand_val * log(test_tens)))
tens_approx_eq(rand_val^test_tens, exp(test_tens * log(rand_val)))

# Special Cases #
#---------------#
@test abs(test_tens) == test_tens
@test abs(-test_tens) == test_tens
tens_approx_eq(abs2(test_tens), test_tens*test_tens)

atan2_tens = atan2(test_tens, rand_tens)
atanyx_tens = atan(test_tens/rand_tens)

@test value(atan2_tens) == atan2(test_val, rand_val)
@test_approx_eq collect(grad(atan2_tens)) collect(grad(atanyx_tens))
@test_approx_eq hess(atan2_tens) hess(atanyx_tens)
@test_approx_eq tens(atan2_tens) tens(atanyx_tens)

# Unary functions/API usage testing #
#-----------------------------------#
testout = Array(Float64, N, N, N)

function tens_deriv_ijk(f_expr, x::Vector, i, j, k)
    var_syms = [:a, :b, :c, :d]
    diff_expr = differentiate(f_expr, var_syms[k])
    diff_expr = differentiate(diff_expr, var_syms[j])
    diff_expr = differentiate(diff_expr, var_syms[i])
    @eval begin
        a,b,c,d = $x
        return $diff_expr
    end
end

function tens_test_result(f_expr, x::Vector)
    return [tens_deriv_ijk(f_expr, x, i, j, k) for i in 1:N, j in 1:N, k in 1:N]
end

function tens_test_x(fsym, N)
    randrange = .01:.01:.99

    needs_modification = tuple(:acosh, :acoth)
    if fsym in needs_modification
        randrange += 1
    end

    return rand(randrange, N)
end

for fsym in ForwardDiff.auto_defined_unary_tens_funcs
    testexpr = :($(fsym)(a) + $(fsym)(b) - $(fsym)(c) * $(fsym)(d))

    @eval function testf(x::Vector)
        a,b,c,d = x
        return $testexpr
    end

    try
        testx = tens_test_x(fsym, N)
        val_result = testf(testx)
        grad_result = ForwardDiff.gradient(testf, testx)
        hess_result = ForwardDiff.hessian(testf, testx)
        tens_result = tens_test_result(testexpr, testx)

        # Non-AllResults
        test_tens = (testout) -> @test_approx_eq testout tens_result

        ForwardDiff.tensor!(testout, testf, testx)
        test_tens(testout)

        test_tens(ForwardDiff.tensor(testf, testx))

        tensf! = ForwardDiff.tensor(testf; mutates=true)
        testout = similar(testout)
        tensf!(testout, testx)
        test_tens(testout)

        tensf = ForwardDiff.tensor(testf; mutates=false)
        test_tens(tensf(testx))

        # AllResults
        test_all_results = (testout, results) -> begin
            @test_approx_eq ForwardDiff.value(results) val_result
            @test_approx_eq ForwardDiff.gradient(results) grad_result
            @test_approx_eq ForwardDiff.hessian(results) hess_result
            test_tens(ForwardDiff.tensor(results))
            test_tens(testout)
        end

        testout = similar(testout)
        results = ForwardDiff.tensor!(testout, testf, testx, AllResults)
        test_all_results(testout, results[2])

        testout = similar(testout)
        testout, results2 = ForwardDiff.tensor(testf, testx, AllResults)
        test_all_results(testout, results2)

        tensf! = ForwardDiff.tensor(testf, AllResults; mutates=true)
        testout = similar(testout)
        results3 = tensf!(testout, testx)
        test_all_results(testout, results3[2])

        tensf = ForwardDiff.tensor(testf, AllResults; mutates=false)
        testout = similar(testout)
        testout, results4 = tensf(testx)
        test_all_results(testout, results4)
    catch err
        warn("Failure when testing Tensors involving $fsym:")
        throw(err)
    end
end

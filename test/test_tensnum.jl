using Base.Test
using Calculus
using ForwardDiff
using ForwardDiff: 
        GradientNum,
        HessianNum,
        TensorNum,
        value,
        grad,
        hess,
        tens,
        npartials,
        isconstant

floatrange = 0.0:.01:.99
intrange = 0:10
N = 4
T = Float64
C = NTuple{N,T}
hessveclen = ForwardDiff.halfhesslen(N)
tensveclen = ForwardDiff.halftenslen(N)

test_val = rand(floatrange)
test_partials = tuple(rand(floatrange, N)...)
test_hessvec = rand(floatrange, hessveclen)
test_tensvec = rand(floatrange, tensveclen)

test_grad = GradientNum(test_val, test_partials)
test_hess = HessianNum(test_grad, test_hessvec)
test_tens = TensorNum(test_hess, test_tensvec)

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

tens_zero = TensorNum(zero(test_hess), map(zero, test_tensvec))
tens_one = TensorNum(one(test_hess), map(zero, test_tensvec))

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

int_tens = TensorNum(HessianNum(GradientNum(int_val, int_partials), int_hessvec), int_tensvec)
float_tens = TensorNum(HessianNum(GradientNum(float_val, float_partials), float_hessvec), float_tensvec)
const_tens = TensorNum(float_val)

@test convert(typeof(test_tens), test_tens) == test_tens
@test convert(TensorNum, test_tens) == test_tens
@test convert(TensorNum{N,T,C}, int_tens) == float_tens
@test convert(TensorNum{0,T,Tuple{}}, 1) == TensorNum(1.0)
@test convert(TensorNum{3,T,NTuple{3,T}}, 1) == TensorNum{3,T,NTuple{3,T}}(1.0)
@test convert(T, TensorNum(HessianNum(GradientNum(1, tuple(0, 0))))) == 1.0

IntTens = TensorNum{N,Int,NTuple{N,Int}}
FloatTens = TensorNum{N,Float64,NTuple{N,Float64}}

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
@test isnan(TensorNum(NaN))

not_const_tens = TensorNum(HessianNum(GradientNum(one(T), map(one, test_partials))))
@test !(isconstant(not_const_tens) || isreal(not_const_tens))
@test isconstant(const_tens) && isreal(const_tens)
@test isconstant(zero(not_const_tens)) && isreal(zero(not_const_tens))

@test isfinite(test_tens) == isfinite(test_val)
@test !isfinite(TensorNum(Inf))

@test isless(test_tens-1, test_tens)
@test isless(test_val-1, test_tens)
@test isless(test_tens, test_val+1)

#######
# I/O #
#######
io = IOBuffer()
write(io, test_tens)
seekstart(io)

@test read(io, typeof(test_tens)) == test_tens

close(io)

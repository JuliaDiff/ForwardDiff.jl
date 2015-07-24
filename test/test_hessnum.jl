using Base.Test
using Calculus
using ForwardDiff
using ForwardDiff: 
        GradientNum,
        HessianNum,
        value,
        grad,
        hess,
        npartials,
        isconstant

floatrange = 0.0:.01:.99
intrange = 0:10
N = 4
T = Float64
C = NTuple{N,T}
hessveclen = ForwardDiff.halfhesslen(N)

test_val = rand(floatrange)
test_partials = tuple(rand(floatrange, N)...)
test_hessvec = rand(floatrange, hessveclen)
test_grad = GradientNum(test_val, test_partials)
test_hess = HessianNum(test_grad, test_hessvec)

######################
# Accessor Functions #
######################
@test value(test_hess) == test_val
@test grad(test_hess) == test_partials
@test hess(test_hess) == test_hessvec

for i in 1:N
    @test grad(test_hess, i) == test_partials[i]
end

for i in 1:hessveclen
    @test hess(test_hess, i) == test_hessvec[i]
end

@test npartials(test_hess) == npartials(typeof(test_hess)) == N

##################################
# Value Representation Functions #
##################################
@test eps(test_hess) == eps(test_val)
@test eps(typeof(test_hess)) == eps(T)

hess_zero = HessianNum(zero(test_grad), map(zero, test_hessvec))
hess_one = HessianNum(one(test_grad), map(zero, test_hessvec))

@test zero(test_hess) == hess_zero
@test zero(typeof(test_hess)) == hess_zero

@test one(test_hess) == hess_one
@test one(typeof(test_hess)) == hess_one

#########################################
# Conversion/Promotion/Hashing/Equality #
#########################################
int_val = round(Int, test_val)
int_partials = map(x -> round(Int, x), test_partials)
int_hessvec = map(x -> round(Int, x), test_hessvec)

float_val = float(int_val)
float_partials = map(float, int_partials)
float_hessvec= map(float, int_hessvec)

int_hess = HessianNum(GradientNum(int_val, int_partials), int_hessvec)
float_hess = HessianNum(GradientNum(float_val, float_partials), float_hessvec)
const_hess = HessianNum(GradientNum(float_val))

@test convert(typeof(test_hess), test_hess) == test_hess
@test convert(HessianNum, test_hess) == test_hess
@test convert(HessianNum{N,T,C}, int_hess) == float_hess
@test convert(HessianNum{0,T,Tuple{}}, 1) == HessianNum(GradientNum(1.0))
@test convert(HessianNum{3,T,NTuple{3,T}}, 1) == HessianNum(GradientNum{3,T,NTuple{3,T}}(1.0))
@test convert(T, HessianNum(GradientNum(1, tuple(0, 0)))) == 1.0

IntHess = HessianNum{N,Int,NTuple{N,Int}}
FloatHess = HessianNum{N,Float64,NTuple{N,Float64}}

@test promote_type(IntHess, IntHess) == IntHess
@test promote_type(FloatHess, IntHess) == FloatHess
@test promote_type(IntHess, Float64) == FloatHess
@test promote_type(FloatHess, Int) == FloatHess

@test hash(int_hess) == hash(float_hess)
@test hash(const_hess) == hash(float_val)

@test int_hess == float_hess
@test float_val == const_hess
@test const_hess == float_val

@test isequal(int_hess, float_hess)
@test isequal(float_val, const_hess)
@test isequal(const_hess, float_val)

@test copy(test_hess) == test_hess

####################
# is____ Functions #
####################
@test isnan(test_hess) == isnan(test_val)
@test isnan(HessianNum(GradientNum(NaN)))

not_const_hess = HessianNum(GradientNum(one(T), map(one, test_partials)))
@test !(isconstant(not_const_hess) || isreal(not_const_hess))
@test isconstant(const_hess) && isreal(const_hess)
@test isconstant(zero(not_const_hess)) && isreal(zero(not_const_hess))

@test isfinite(test_hess) == isfinite(test_val)
@test !isfinite(HessianNum(GradientNum(Inf)))

@test isless(test_hess-1, test_hess)
@test isless(test_val-1, test_hess)
@test isless(test_hess, test_val+1)

#######
# I/O #
#######
io = IOBuffer()
write(io, test_hess)
seekstart(io)

@test read(io, typeof(test_hess)) == test_hess

close(io)

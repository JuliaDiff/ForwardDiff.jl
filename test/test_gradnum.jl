#############################################################
# Test basic, non-math functions on ForwardDiff.GradientNum #
#############################################################
floatrange = -10.0:.01:10.0
intrange = -10:10
N = 3
T = Float64

val = rand(floatrange)
partialtup = tuple(rand(floatrange, N)...)
partialvec = collect(partialtup)

for (testpartials, TestGrad) in ((:partialtup, :GradNumTup), (:partialvec, :GradNumVec))
    @eval begin
        derivs = $testpartials       
        Grad = $TestGrad
        testgrad = Grad{N,T}(val, derivs)

        ######################
        # Accessor Functions #
        ######################
        @test ForwardDiff.value(testgrad) == val
        @test ForwardDiff.partials(testgrad) == derivs

        for i in 1:N
            @test ForwardDiff.partials(testgrad, i) == derivs[i]
        end

        @test ForwardDiff.npartials(testgrad) == ForwardDiff.npartials(typeof(testgrad)) == N
        
        ##################################
        # Value Representation Functions #
        ##################################
        @test eps(testgrad) == eps(ForwardDiff.value(testgrad))
        @test eps(typeof(testgrad)) == eps(T)

        grad_zero = Grad{N,T}(zero(val), map(zero, derivs))
        grad_one = Grad{N,T}(one(val), map(zero, derivs))

        @test zero(testgrad) == grad_zero
        @test zero(typeof(testgrad)) == grad_zero

        @test one(testgrad) == grad_one
        @test one(typeof(testgrad)) == grad_one

        #########################################
        # Conversion/Promotion/Hashing/Equality #
        #########################################
        int_val = round(Int, val)
        int_derivs = map(x -> round(Int, x), derivs)
        float_val = float(int_val)
        float_derivs = map(float, int_derivs)

        int_grad = Grad{N,Int}(int_val, int_derivs)
        float_grad = Grad{N,Float64}(float_val, float_derivs)
        const_grad = Grad{N,Float64}(float_val)

        @test convert(typeof(testgrad), testgrad) == testgrad
        @test convert(ForwardDiff.GradientNum, testgrad) == testgrad
        @test convert(Grad{N,Float64}, int_grad) == float_grad
        @test convert(Grad{0,Float64}, 1) == Grad{0,Float64}(1.0, ForwardDiff.zero_partials(Grad{0,Float64}))
        @test convert(Grad{3,Float64}, 1) == Grad{3,Float64}(1.0, ForwardDiff.zero_partials(Grad{3,Float64}))
        @test convert(Float64, Grad{2,Float64}(1)) == 1.0

        @test promote_type(Grad{N,Int}, Grad{N,Int}) == Grad{N,Int}
        @test promote_type(Grad{N,Float64}, Grad{N,Int}) == Grad{N,Float64}
        @test promote_type(Grad{N,Int}, Float64) == Grad{N,Float64}
        @test promote_type(Grad{N,Float64}, Int) == Grad{N,Float64}

        @test hash(int_grad) == hash(float_grad)
        @test hash(const_grad) == hash(float_val)

        @test int_grad == float_grad
        @test float_val == const_grad
        @test const_grad == float_val

        @test isequal(int_grad, float_grad)
        @test isequal(float_val, const_grad)
        @test isequal(const_grad, float_val)

        @test copy(testgrad) == testgrad

        ####################
        # is____ Functions #
        ####################
        @test isnan(testgrad) == isnan(ForwardDiff.value(testgrad))
        @test isnan(Grad{0,Float64}(NaN))

        not_const_grad = Grad{N,T}(1, map(one, derivs))
        @test !(ForwardDiff.isconstant(not_const_grad) || isreal(not_const_grad) || any(x -> x == 0, ForwardDiff.partials(not_const_grad)))
        @test ForwardDiff.isconstant(const_grad) && isreal(const_grad)
        @test ForwardDiff.isconstant(zero(not_const_grad)) && isreal(zero(not_const_grad))

        @test isfinite(testgrad) && isfinite(ForwardDiff.value(testgrad))
        @test !isfinite(Grad{N,T}(Inf))

        @test isless(testgrad-1, testgrad)
        @test isless(ForwardDiff.value(testgrad)-1, testgrad)
        @test isless(testgrad, ForwardDiff.value(testgrad)+1)

        #######
        # I/O #
        #######
        io = IOBuffer()
        write(io, testgrad)
        seekstart(io)

        @test read(io, typeof(testgrad)) == testgrad
        
        close(io)

    end
end

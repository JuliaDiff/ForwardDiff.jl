module ComplexTest
using ForwardDiff: Dual
using Test, ForwardDiff

function numeric_jacobian_complex(f, args::T...; δ=1e-5, kwargs...) where T<:Complex
    n = length(args)
    J = zeros(2, 2n)
    largs = [args...]
    for i=1:n
        # perturb real
        largs[i] += δ/2
        pos = f(largs...; kwargs...)
        largs[i] -= δ
        neg = f(largs...; kwargs...)
        largs[i] += δ/2
        J[1,2i-1] = (real(pos) - real(neg))/δ
        J[2,2i-1] = (imag(pos) - imag(neg))/δ
        # perturb real
        largs[i] += δ/2*im
        pos = f(largs...; kwargs...)
        largs[i] -= δ*im
        neg = f(largs...; kwargs...)
        largs[i] += δ/2*im
        J[1,2i] = (real(pos) - real(neg))/δ
        J[2,2i] = (imag(pos) - imag(neg))/δ
    end
    return J
end

function complex_jacobian_wrapper(f)
    function newf(params)
        newargs = [Complex(params[2i-1], params[2i]) for i=1:length(params)÷2]
        res = f(newargs...)
        [real(res), imag(res)]
    end
end

function check_complex_jacobian(f, args...; kwargs...)
    nj = numeric_jacobian_complex(f, args...; δ=1e-5, kwargs...)
    params = vcat([[x.re, x.im] for x in args]...)
    fj = ForwardDiff.jacobian(complex_jacobian_wrapper(f), params)
    @test isapprox(nj, fj, atol=1e-5)
end

@testset "complex instructions" begin
    for OP in [+, *, /, -, ^]
        println("  ...testing Complex Valued $OP")
        check_complex_jacobian(OP, 4.0+2im, 2.0+1im)
    end
    for OP in [abs, abs2, real, imag, conj, adjoint, sin, cos, tan,
            sinh, cosh, tanh, exp, log, angle, x->x^3, x->x^0.5, sqrt]
        println("  ...testing Complex Valued $OP")
        check_complex_jacobian(OP, 4.0+2im)
    end
end
end

#include "benchmarks.h"

Dual2::Dual2(double real, double eps1, double eps2) : real(real), eps1(eps1), eps2(eps2) {}
Dual2::Dual2(double real) : real(real), eps1(0.0), eps2(0.0) {}
Dual2::Dual2() : real(0.0), eps1(0.0), eps2(0.0) {}

std::ostream& operator<<(std::ostream& os, const Dual2& x) {
    os << "Dual2(" << x.real << "," << x.eps1 << "," << x.eps2 << ")";
    return os;
}

Dual2 operator +(const Dual2& x, const Dual2& y) {
    return Dual2(x.real+y.real, x.eps1+y.eps1, x.eps2+y.eps2);
}

Dual2 operator -(const Dual2& x, const Dual2& y) {
    return Dual2(x.real-y.real, x.eps1-y.eps1, x.eps2-y.eps2);
}

Dual2 operator *(const Dual2& x, const Dual2& y) {
    return Dual2(x.real*y.real,
                 x.real*y.eps1 + x.eps1*y.real,
                 x.real*y.eps2 + x.eps2*y.real);
}

Dual2 operator *(double x, const Dual2 &y) {
    return Dual2(x*y.real, x*y.eps1, x*y.eps2);
}

Dual2 operator -(double x, const Dual2 &y) {
    return Dual2(x-y.real, -y.eps1, -y.eps2);
}

Dual2 sin(const Dual2& x) {
    double deriv = cos(x.real);
    return Dual2(sin(x.real), x.eps1*deriv, x.eps2*deriv);
}

Dual2 cos(const Dual2& x) {
    double deriv = -sin(x.real);
    return Dual2(cos(x.real), x.eps1*deriv, x.eps2*deriv);
}

Dual2 sqrt(const Dual2& x) {
    double sqrt_real = sqrt(x.real);
    double deriv = 1/(2*sqrt_real);
    return Dual2(sqrt_real, x.eps1*deriv, x.eps2*deriv);
}

Dual2 exp(const Dual2& x) {
    double deriv = exp(x.real);
    return Dual2(deriv, x.eps1*deriv, x.eps2*deriv);
}

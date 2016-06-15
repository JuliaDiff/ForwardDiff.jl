#include "benchmarks.h"

Dual1::Dual1(double real, double eps1) : real(real), eps1(eps1) {}
Dual1::Dual1(double real) : real(real), eps1(0.0) {}
Dual1::Dual1() : real(0.0), eps1(0.0) {}

std::ostream& operator<<(std::ostream& os, const Dual1& x) {
    os << "Dual1(" << x.real << "," << x.eps1 << ")";
    return os;
}

Dual1 operator +(const Dual1& x, const Dual1& y) {
    return Dual1(x.real+y.real, x.eps1+y.eps1);
}

Dual1 operator -(const Dual1& x, const Dual1& y) {
    return Dual1(x.real-y.real, x.eps1-y.eps1);
}

Dual1 operator *(const Dual1& x, const Dual1& y) {
    return Dual1(x.real*y.real, x.real*y.eps1 + x.eps1*y.real);
}

Dual1 operator *(double x, const Dual1 &y) {
    return Dual1(x*y.real, x*y.eps1);
}

Dual1 operator -(double x, const Dual1 &y) {
    return Dual1(x-y.real, -y.eps1);
}

Dual1 sin(const Dual1& x) {
    double deriv = cos(x.real);
    return Dual1(sin(x.real), x.eps1*deriv);
}

Dual1 cos(const Dual1& x) {
    double deriv = -sin(x.real);
    return Dual1(cos(x.real), x.eps1*deriv);
}

Dual1 sqrt(const Dual1& x) {
    double sqrt_real = sqrt(x.real);
    double deriv = 1/(2*sqrt_real);
    return Dual1(sqrt_real, x.eps1*deriv);
}

Dual1 exp(const Dual1& x) {
    double deriv = exp(x.real);
    return Dual1(deriv, x.eps1*deriv);
}

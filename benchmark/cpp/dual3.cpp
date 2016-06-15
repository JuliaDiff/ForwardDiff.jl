#include "benchmarks.h"

Dual3::Dual3(double real, double eps1, double eps2, double eps3) : real(real), eps1(eps1), eps2(eps2), eps3(eps3) {}
Dual3::Dual3(double real) : real(real), eps1(0.0), eps2(0.0), eps3(0.0) {}
Dual3::Dual3() : real(0.0), eps1(0.0), eps2(0.0), eps3(0.0) {}

std::ostream& operator<<(std::ostream& os, const Dual3& x) {
    os << "Dual3(" << x.real << "," << x.eps1 << "," << x.eps2 << "," << x.eps3 << ")";
    return os;
}

Dual3 operator +(const Dual3& x, const Dual3& y) {
    return Dual3(x.real+y.real, x.eps1+y.eps1, x.eps2+y.eps2, x.eps3+y.eps3);
}

Dual3 operator -(const Dual3& x, const Dual3& y) {
    return Dual3(x.real-y.real, x.eps1-y.eps1, x.eps2-y.eps2, x.eps3-y.eps3);
}

Dual3 operator *(const Dual3& x, const Dual3& y) {
    return Dual3(x.real*y.real,
                 x.real*y.eps1 + x.eps1*y.real,
                 x.real*y.eps2 + x.eps2*y.real,
                 x.real*y.eps3 + x.eps3*y.real);
}

Dual3 operator *(double x, const Dual3 &y) {
    return Dual3(x*y.real, x*y.eps1, x*y.eps2, x*y.eps3);
}

Dual3 operator -(double x, const Dual3 &y) {
    return Dual3(x-y.real, -y.eps1, -y.eps2, -y.eps3);
}

Dual3 sin(const Dual3& x) {
    double deriv = cos(x.real);
    return Dual3(sin(x.real), x.eps1*deriv, x.eps2*deriv, x.eps3*deriv);
}

Dual3 cos(const Dual3& x) {
    double deriv = -sin(x.real);
    return Dual3(cos(x.real), x.eps1*deriv, x.eps2*deriv, x.eps3*deriv);
}

Dual3 sqrt(const Dual3& x) {
    double sqrt_real = sqrt(x.real);
    double deriv = 1/(2*sqrt_real);
    return Dual3(sqrt_real, x.eps1*deriv, x.eps2*deriv, x.eps3*deriv);
}

Dual3 exp(const Dual3& x) {
    double deriv = exp(x.real);
    return Dual3(deriv, x.eps1*deriv, x.eps2*deriv, x.eps3*deriv);
}

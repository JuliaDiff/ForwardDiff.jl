#include "benchmarks.h"

Dual4::Dual4(double real, double eps1, double eps2, double eps3, double eps4) : real(real), eps1(eps1), eps2(eps2), eps3(eps3), eps4(eps4) {}
Dual4::Dual4(double real) : real(real), eps1(0.0), eps2(0.0), eps3(0.0), eps4(0.0) {}
Dual4::Dual4() : real(0.0), eps1(0.0), eps2(0.0), eps3(0.0), eps4(0.0) {}

std::ostream& operator<<(std::ostream& os, const Dual4& x) {
    os << "Dual4(" << x.real << "," << x.eps1 << "," << x.eps2 << "," << x.eps3 << "," << x.eps4 << ")";
    return os;
}

Dual4 operator +(const Dual4& x, const Dual4& y) {
    return Dual4(x.real+y.real, x.eps1+y.eps1, x.eps2+y.eps2, x.eps3+y.eps3, x.eps4+y.eps4);
}

Dual4 operator -(const Dual4& x, const Dual4& y) {
    return Dual4(x.real-y.real, x.eps1-y.eps1, x.eps2-y.eps2, x.eps3-y.eps3, x.eps4-y.eps4);
}

Dual4 operator *(const Dual4& x, const Dual4& y) {
    return Dual4(x.real*y.real,
                 x.real*y.eps1 + x.eps1*y.real,
                 x.real*y.eps2 + x.eps2*y.real,
                 x.real*y.eps3 + x.eps3*y.real,
                 x.real*y.eps4 + x.eps4*y.real);
}

Dual4 operator *(double x, const Dual4 &y) {
    return Dual4(x*y.real, x*y.eps1, x*y.eps2, x*y.eps3, x*y.eps4);
}

Dual4 operator -(double x, const Dual4 &y) {
    return Dual4(x-y.real, -y.eps1, -y.eps2, -y.eps3, -y.eps4);
}

Dual4 sin(const Dual4& x) {
    double deriv = cos(x.real);
    return Dual4(sin(x.real), x.eps1*deriv, x.eps2*deriv, x.eps3*deriv, x.eps4*deriv);
}

Dual4 cos(const Dual4& x) {
    double deriv = -sin(x.real);
    return Dual4(cos(x.real), x.eps1*deriv, x.eps2*deriv, x.eps3*deriv, x.eps4*deriv);
}

Dual4 sqrt(const Dual4& x) {
    double sqrt_real = sqrt(x.real);
    double deriv = 1/(2*sqrt_real);
    return Dual4(sqrt_real, x.eps1*deriv, x.eps2*deriv, x.eps3*deriv, x.eps4*deriv);
}

Dual4 exp(const Dual4& x) {
    double deriv = exp(x.real);
    return Dual4(deriv, x.eps1*deriv, x.eps2*deriv, x.eps3*deriv, x.eps4*deriv);
}

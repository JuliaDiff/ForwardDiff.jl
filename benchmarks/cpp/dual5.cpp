#include "benchmarks.h"

Dual5::Dual5(double real, double eps1, double eps2, double eps3, double eps4, double eps5) : real(real), eps1(eps1), eps2(eps2), eps3(eps3), eps4(eps4), eps5(eps5) {}
Dual5::Dual5(double real) : real(real), eps1(0.0), eps2(0.0), eps3(0.0), eps4(0.0), eps5(0.0) {}
Dual5::Dual5() : real(0.0), eps1(0.0), eps2(0.0), eps3(0.0), eps4(0.0), eps5(0.0) {}

std::ostream& operator<<(std::ostream& os, const Dual5& x) {
    os << "Dual5(" << x.real << "," << x.eps1 << "," << x.eps2 << "," << x.eps3 << "," << x.eps4 << "," << x.eps5 << ")";
    return os;
}

Dual5 operator +(const Dual5& x, const Dual5& y) {
    return Dual5(x.real+y.real, x.eps1+y.eps1, x.eps2+y.eps2,
                 x.eps3+y.eps3, x.eps4+y.eps4, x.eps5+y.eps5);
}

Dual5 operator -(const Dual5& x, const Dual5& y) {
    return Dual5(x.real-y.real, x.eps1-y.eps1, x.eps2-y.eps2,
                 x.eps3-y.eps3, x.eps4-y.eps4, x.eps5-y.eps5);
}

Dual5 operator *(const Dual5& x, const Dual5& y) {
    return Dual5(x.real*y.real,
                 x.real*y.eps1 + x.eps1*y.real,
                 x.real*y.eps2 + x.eps2*y.real,
                 x.real*y.eps3 + x.eps3*y.real,
                 x.real*y.eps4 + x.eps4*y.real,
                 x.real*y.eps5 + x.eps5*y.real);
}

Dual5 operator *(double x, const Dual5 &y) {
    return Dual5(x*y.real, x*y.eps1, x*y.eps2, x*y.eps3, x*y.eps4, x*y.eps5);
}

Dual5 operator -(double x, const Dual5 &y) {
    return Dual5(x-y.real, -y.eps1, -y.eps2, -y.eps3, -y.eps4, -y.eps5);
}

Dual5 sin(const Dual5& x) {
    double deriv = cos(x.real);
    return Dual5(sin(x.real), x.eps1*deriv, x.eps2*deriv,
             x.eps3*deriv, x.eps4*deriv, x.eps5*deriv);
}

Dual5 cos(const Dual5& x) {
    double deriv = -sin(x.real);
    return Dual5(cos(x.real), x.eps1*deriv, x.eps2*deriv,
             x.eps3*deriv, x.eps4*deriv, x.eps5*deriv);
}

Dual5 sqrt(const Dual5& x) {
    double sqrt_real = sqrt(x.real);
    double deriv = 1/(2*sqrt_real);
    return Dual5(sqrt_real, x.eps1*deriv, x.eps2*deriv,
                 x.eps3*deriv, x.eps4*deriv, x.eps5*deriv);
}

Dual5 exp(const Dual5& x) {
    double deriv = exp(x.real);
    return Dual5(deriv, x.eps1*deriv, x.eps2*deriv,
                 x.eps3*deriv, x.eps4*deriv, x.eps5*deriv);
}

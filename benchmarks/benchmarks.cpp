#include <iostream>
#include <cmath>
#include <vector>
#include <sys/time.h>

using namespace std;

// g++ -std=c++11 -O2 -o benchmarks benchmarks.cpp /path/to/julia/usr/lib/libopenlibm.a


double clock_now()
{
    struct timeval now;
    gettimeofday(&now, NULL);
    return (double)now.tv_sec + (double)now.tv_usec/1.0e6;
}

class Dual {
    double real;
    double eps;
    public:
    Dual(double real, double eps) : real(real), eps(eps) {}
    Dual(double real) : real(real), eps(0.0) {}
    Dual() : real(0.0), eps(0.0) {}

    friend ostream& operator<<(ostream& os, const Dual& dt) {
        os << "Dual(" << dt.real << "," << dt.eps << ")";
        return os;
    }

    Dual operator+(const Dual& rhs) const {
        return Dual(real+rhs.real,eps+rhs.eps);
    }

    Dual operator-(const Dual& rhs) const {
        return Dual(real-rhs.real,eps-rhs.eps);
    }

    Dual operator*(const Dual& rhs) const {
        return Dual(real*rhs.real, real*rhs.eps + eps*rhs.real);
    }

    double real_part() const { return real; }
    double eps_part() const { return eps; }

};

Dual operator*(double x, const Dual &rhs) {
    return Dual(x*rhs.real_part(), x*rhs.eps_part());
}

Dual operator-(double x, const Dual &rhs) {
    return Dual(x - rhs.real_part(), -rhs.eps_part());
}

Dual sin(const Dual& x) {
    return Dual(sin(x.real_part()), x.eps_part()*cos(x.real_part()));
}

Dual cos(const Dual& x) {
    return Dual(cos(x.real_part()), -x.eps_part()*sin(x.real_part()));
}

Dual sqrt(const Dual& x) {
    return Dual(sqrt(x.real_part()), x.eps_part()/(2*sqrt(x.real_part())));
}

Dual exp(const Dual& x) {
    double expval = exp(x.real_part());
    return Dual(expval, x.eps_part()*expval);
}

/*
function ackley(x)
    a, b, c = 20.0, -0.2, 2.0*π
    len_recip = inv(length(x))
    sum_sqrs = zero(eltype(x))
    sum_cos = sum_sqrs
    for i in x
        sum_cos += cos(c*i)
        sum_sqrs += sqr(i)
    end
    return (-a * exp(b * sqrt(len_recip*sum_sqrs)) -
            exp(len_recip*sum_cos) + a + e)
end
*/

template<typename T> T ackley(const vector<T> &x) {
    double a = 20.0, b = -0.2, c = 2*M_PI;
    double len_recip = 1.0/x.size();
    T sum_sqrs(0);
    T sum_cos(0);

    for (auto i : x) {
        sum_cos = sum_cos + cos(c*i);
        sum_sqrs = sum_sqrs + i*i;
    }
    return (-a * exp(b * sqrt(len_recip*sum_sqrs)) -
            exp(len_recip*sum_cos) + a + M_E);
}

/*
function rosenbrock(x)
    a, b = 100.0, 1.0
    result = zero(eltype(x))
    for i in 1:length(x)-1
        result += sqr(b - x[i]) + a*sqr(x[i+1] - sqr(x[i]))
    end
    return result
end
*/

template <typename T> T rosenbrock(const vector<T> &x) {
    double a = 100.0;
    double b = 1.0;
    T result(0);
    for (size_t i = 0; i < x.size()-1; i++) {
        T t1 = b-x[i];
        T t2 = x[i+1]-x[i]*x[i];
        result = result + t1*t1 + a*t2*t2;
    }
    return result;
}

int main() {

    int len = 16000;
    vector<double> vec(len);
    for (int i = 0; i < len; i++) {
        vec[i] = i+1;
    }
    double start = clock_now();
    double val = ackley(vec);
    double end = clock_now();

    vector<double> gradient(len);
    vector<Dual> dualvec(len);
    for (int i = 0; i < len; i++) {
        dualvec[i] = Dual(vec[i]);
    }

    // compute gradient in a loop
    double startgrad = clock_now();
    for (int i = 0; i < len; i++) {
        dualvec[i] = Dual(vec[i],1.0);
        Dual val = ackley(dualvec);
        gradient[i] = val.eps_part();
        dualvec[i] = Dual(vec[i],0.0);
    }
    double endgrad = clock_now();

    cout << "Ackley" << endl;
    cout << val << endl;
    cout << end-start << " sec for function value" << endl;
    cout << endgrad-startgrad << " sec for gradient" << endl;


    start = clock_now();
    val = rosenbrock(vec);
    end = clock_now();

    for (int i = 0; i < len; i++) {
        dualvec[i] = Dual(vec[i]);
    }

    // compute gradient in a loop
    startgrad = clock_now();
    for (int i = 0; i < len; i++) {
        dualvec[i] = Dual(vec[i],1.0);
        Dual val = rosenbrock(dualvec);
        gradient[i] = val.eps_part();
        dualvec[i] = Dual(vec[i],0.0);
    }
    endgrad = clock_now();

    cout << "Rosenbrock" << endl;
    cout << val << endl;
    cout << end-start << " sec for function value" << endl;
    cout << endgrad-startgrad << " sec for gradient" << endl;
}

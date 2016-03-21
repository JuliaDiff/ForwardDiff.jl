#include "benchmarks.h"

double clock_now()
{
    struct timeval now;
    gettimeofday(&now, NULL);
    return (double)now.tv_sec + (double)now.tv_usec/1.0e6;
}

template <typename T> T ackley(const std::vector<T> &x) {
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

template <typename T> T rosenbrock(const std::vector<T> &x) {
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
    int len = 1*2*3*4*5*100; // must be divisible by all chunk sizes

    std::vector<double> input(len);
    std::vector<double> result(len);
    std::vector<Dual1> dualvec1(len);
    std::vector<Dual2> dualvec2(len);
    std::vector<Dual3> dualvec3(len);
    std::vector<Dual4> dualvec4(len);
    std::vector<Dual5> dualvec5(len);

    for (int i = 0; i < len; i++) {
        input[i] = i+1;
    }

    // ackley
    std::cout << "benchmarking ackley with vector of length " << len << "..." << std::endl;

    double start = clock_now();
    ackley(input);
    double end = clock_now();
    std::cout << "  took " << end-start << " seconds for value" << std::endl;

    start = clock_now();
    gradient<ackley<Dual1>>(result, dualvec1, input);
    end = clock_now();
    std::cout << "  took " << end-start << " seconds for gradient using Dual1" << std::endl;

    start = clock_now();
    gradient<ackley<Dual2>>(result, dualvec2, input);
    end = clock_now();
    std::cout << "  took " << end-start << " seconds for gradient using Dual2" << std::endl;

    start = clock_now();
    gradient<ackley<Dual3>>(result, dualvec3, input);
    end = clock_now();
    std::cout << "  took " << end-start << " seconds for gradient using Dual3" << std::endl;

    start = clock_now();
    gradient<ackley<Dual4>>(result, dualvec4, input);
    end = clock_now();
    std::cout << "  took " << end-start << " seconds for gradient using Dual4" << std::endl;

    start = clock_now();
    gradient<ackley<Dual5>>(result, dualvec5, input);
    end = clock_now();
    std::cout << "  took " << end-start << " seconds for gradient using Dual5" << std::endl;

    // rosenbrock
    std::cout << "benchmarking rosenbrock with vector of length " << len << "..." << std::endl;

    start = clock_now();
    rosenbrock(input);
    end = clock_now();
    std::cout << "  took " << end-start << " seconds for value" << std::endl;

    start = clock_now();
    gradient<rosenbrock<Dual1>>(result, dualvec1, input);
    end = clock_now();
    std::cout << "  took " << end-start << " seconds for gradient using Dual1" << std::endl;

    start = clock_now();
    gradient<rosenbrock<Dual2>>(result, dualvec2, input);
    end = clock_now();
    std::cout << "  took " << end-start << " seconds for gradient using Dual2" << std::endl;

    start = clock_now();
    gradient<rosenbrock<Dual3>>(result, dualvec3, input);
    end = clock_now();
    std::cout << "  took " << end-start << " seconds for gradient using Dual3" << std::endl;

    start = clock_now();
    gradient<rosenbrock<Dual4>>(result, dualvec4, input);
    end = clock_now();
    std::cout << "  took " << end-start << " seconds for gradient using Dual4" << std::endl;

    start = clock_now();
    gradient<rosenbrock<Dual5>>(result, dualvec5, input);
    end = clock_now();
    std::cout << "  took " << end-start << " seconds for gradient using Dual5" << std::endl;

    return 0;
}

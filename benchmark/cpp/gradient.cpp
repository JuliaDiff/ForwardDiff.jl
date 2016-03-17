#include "benchmarks.h"

void gradient(std::function<Dual1(std::vector<Dual1>)> f, std::vector<double> &result, std::vector<Dual1> &dualvec, const std::vector<double> &input) {
    int len = input.size();
    for (int i = 0; i < len; i++) {
        dualvec[i] = Dual1(input[i]);
    }
    for (int i = 0; i < len; i++) {
        dualvec[i] = Dual1(input[i], 1.0);
        Dual1 val = f(dualvec);
        result[i] = val.eps1;
        dualvec[i] = Dual1(input[i]);
    }
}

void gradient(std::function<Dual2(std::vector<Dual2>)> f, std::vector<double> &result, std::vector<Dual2> &dualvec, const std::vector<double> &input) {
    int len = input.size();
    for (int i = 0; i < len; i++) {
        dualvec[i] = Dual2(input[i]);
    }
    for (int i = 0; i < (len / 2); i++) {
        int j1 = i*2;
        int j2 = j1+1;
        dualvec[j1] = Dual2(input[j1], 1.0, 0.0);
        dualvec[j2] = Dual2(input[j2], 0.0, 1.0);
        Dual2 val = f(dualvec);
        result[j1] = val.eps1;
        result[j2] = val.eps2;
        dualvec[j1] = Dual2(input[j1]);
        dualvec[j2] = Dual2(input[j2]);
    }
}

void gradient(std::function<Dual3(std::vector<Dual3>)> f, std::vector<double> &result, std::vector<Dual3> &dualvec, const std::vector<double> &input) {
    int len = input.size();
    for (int i = 0; i < len; i++) {
        dualvec[i] = Dual3(input[i]);
    }
    for (int i = 0; i < (len / 3); i++) {
        int j1 = i*3;
        int j2 = j1+1;
        int j3 = j1+2;
        dualvec[j1] = Dual3(input[j1], 1.0, 0.0, 0.0);
        dualvec[j2] = Dual3(input[j2], 0.0, 1.0, 0.0);
        dualvec[j3] = Dual3(input[j3], 0.0, 0.0, 1.0);
        Dual3 val = f(dualvec);
        result[j1] = val.eps1;
        result[j2] = val.eps2;
        result[j3] = val.eps3;
        dualvec[j1] = Dual3(input[j1]);
        dualvec[j2] = Dual3(input[j2]);
        dualvec[j3] = Dual3(input[j3]);
    }
}

void gradient(std::function<Dual4(std::vector<Dual4>)> f, std::vector<double> &result, std::vector<Dual4> &dualvec, const std::vector<double> &input) {
    int len = input.size();
    for (int i = 0; i < len; i++) {
        dualvec[i] = Dual4(input[i]);
    }
    for (int i = 0; i < (len / 4); i++) {
        int j1 = i*4;
        int j2 = j1+1;
        int j3 = j1+2;
        int j4 = j1+3;
        dualvec[j1] = Dual4(input[j1], 1.0, 0.0, 0.0, 0.0);
        dualvec[j2] = Dual4(input[j2], 0.0, 1.0, 0.0, 0.0);
        dualvec[j3] = Dual4(input[j3], 0.0, 0.0, 1.0, 0.0);
        dualvec[j4] = Dual4(input[j4], 0.0, 0.0, 0.0, 1.0);
        Dual4 val = f(dualvec);
        result[j1] = val.eps1;
        result[j2] = val.eps2;
        result[j3] = val.eps3;
        result[j4] = val.eps4;
        dualvec[j1] = Dual4(input[j1]);
        dualvec[j2] = Dual4(input[j2]);
        dualvec[j3] = Dual4(input[j3]);
        dualvec[j4] = Dual4(input[j4]);
    }
}

void gradient(std::function<Dual5(std::vector<Dual5>)> f, std::vector<double> &result, std::vector<Dual5> &dualvec, const std::vector<double> &input) {
    int len = input.size();
    for (int i = 0; i < len; i++) {
        dualvec[i] = Dual5(input[i]);
    }
    for (int i = 0; i < (len / 5); i++) {
        int j1 = i*5;
        int j2 = j1+1;
        int j3 = j1+2;
        int j4 = j1+3;
        int j5 = j1+4;
        dualvec[j1] = Dual5(input[j1], 1.0, 0.0, 0.0, 0.0, 0.0);
        dualvec[j2] = Dual5(input[j2], 0.0, 1.0, 0.0, 0.0, 0.0);
        dualvec[j3] = Dual5(input[j3], 0.0, 0.0, 1.0, 0.0, 0.0);
        dualvec[j4] = Dual5(input[j4], 0.0, 0.0, 0.0, 1.0, 0.0);
        dualvec[j5] = Dual5(input[j5], 0.0, 0.0, 0.0, 0.0, 1.0);
        Dual5 val = f(dualvec);
        result[j1] = val.eps1;
        result[j2] = val.eps2;
        result[j3] = val.eps3;
        result[j4] = val.eps4;
        result[j5] = val.eps5;
        dualvec[j1] = Dual5(input[j1]);
        dualvec[j2] = Dual5(input[j2]);
        dualvec[j3] = Dual5(input[j3]);
        dualvec[j4] = Dual5(input[j4]);
        dualvec[j5] = Dual5(input[j5]);
    }
}

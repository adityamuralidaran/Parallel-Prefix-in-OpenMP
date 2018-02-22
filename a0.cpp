#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

#include "a0.hpp"


int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "usage: " << argv[0] << " n" << std::endl;
        return -1;
    }

    int n = std::atoi(argv[1]);

    if (n < 8) {
        std::cout << "hey, n is too small even for debugging!" << std::endl;
        return -1;
    }

    // in and out (in is set to 1s for fun)
    std::vector<int> in(n, 1);

    std::vector<int> out_s(n, 0);
    std::vector<int> out_p(n, 0);

    // first run sequential to get T1
    std::cout << "running..." << std::endl;
    auto t0 = std::chrono::system_clock::now();

    std::partial_sum(in.begin(), in.end(), out_s.begin(), std::plus<int>());

    // no running awesome parallel from a0.hpp
    auto t1 = std::chrono::system_clock::now();

    omp_scan(n, in.data(), out_p.data(), std::plus<int>());

    auto t2 = std::chrono::system_clock::now();

    // test for correctness
    std::cout << "testing..." << std::endl;
    std::cout << ((out_s == out_p) ? "pass" : "fail") << std::endl;

    // and produce report
    auto elapsed_seq = std::chrono::duration<double>(t1 - t0);
    auto elapsed_par = std::chrono::duration<double>(t2 - t1);

    std::cout << "T1: " << elapsed_seq.count() << "s" << std::endl;
    std::cout << "Tp: " << elapsed_par.count() << "s" << std::endl;

    std::cout << "Sp: " << static_cast<double>(elapsed_seq.count()) / elapsed_par.count() << std::endl;

    return 0;
} // main

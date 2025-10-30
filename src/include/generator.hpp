#include <array>
#include <cstdint>
#include <limits>
#include <random>
#include <cmath>
#include <vector>
#include <numeric>
#include <stdexcept>
#include <iostream>

namespace generator {

class NttHelper {
public:
    //전체 다항식의 크기  N = 2^n
    uint32_t n_ = 14;
    uint32_t N_ = std::pow(2,n_);
    // q_ 는 10~20-bit 크기이며, q mod 2N = 1 을 만족하는 prime number.
    uint32_t q_ = 65537;
    std::vector<uint32_t> arr_;

    NttHelper() {
        // 초기 다항식의 계수들은 [0, q-1] 사이의 랜덤 값들을 갖는다.
        arr_.resize(N_);
        std::random_device rd;
        std::mt19937 engine(rd());
        std::uniform_int_distribution<std::uint32_t> dist(0, q_ - 1);

        for (auto& value : arr_) {
            value = dist(engine);
        }
    }

    std::vector<uint32_t> makeTwiddleFactor();

};

/*
    twiddle factor 행렬 W를 만든다.
    W_NxN은 (psi_(2N,q))^(2ij+j)을 원소로 갖는다.
*/
std::vector<uint32_t> NttHelper::makeTwiddleFactor() {
    // base^exp mod m의 값을 구한다.
    auto mod_pow = [](uint64_t base, uint64_t exp, uint64_t m) {
        uint64_t result = 1;
        while (exp > 0) {
            if (exp & 1) {
                result = (result * base) % m;
            }
            base = (base * base) % m;
            exp >>= 1;
        }
        return static_cast<uint32_t>(result);
    };

    // value의 소인수들을 return 한다.
    auto factorize = [](uint32_t value) {
        std::vector<uint32_t> factors;
        for (uint32_t p = 2; p * p <= value; ++p) {
            if (value % p == 0) {
                factors.push_back(p);
                while (value % p == 0) {
                    value /= p;
                }
            }
        }
        if (value > 1) {
            factors.push_back(value);
        }
        return factors;
    };

    const uint32_t target_order = 2 * N_;

    //q % 2N = 1 을 만족해야 한다.
    if (q_ - 1 % target_order == 0) {
        throw std::runtime_error("q_-1 not divisible by 2N_, cannot find primitive 2N-th root.");
    }

    const auto order_factors = factorize(target_order);


    // twiddle factor(2N,q)를 찾는다.
    uint32_t twiddle = 0;

    // g=2 부터 시작하여 후보를 테스트
    for (uint32_t g = 2; g < q_; ++g) {
        // 조건 1 : g^2N % q = 1 인지 확인
        if(mod_pow(g, target_order, q_) != 1) {
            continue;
        }

        // 조건 2 : 2N의 모든 소인수 p에 대해 g^(2N/p) % q = 1 인지 확인
        bool is_primitive = true;
        for(auto& pf : order_factors) {
            if(mod_pow(g, target_order / pf, q_) == 1) {
                is_primitive = false;
                break;
            }
        }

        if(is_primitive) {
            twiddle = g;
            break;
        }
    }

    // twiddle factor를 찾지 못할 경우 runtime error를 낸다.
    if(twiddle == 0) {
        throw std::runtime_error("Failed to find primitive 2N-th root of unity.");
    }

    // 구한 twiddle factor로 행렬 W를 만든다.
    std::vector<uint32_t> W(N_*N_);
    for(int i = 0; i < N_; i++) {
        for(int j = 0; j < N_; j++) {
            W.at(i * N_ + j) = mod_pow(twiddle, 2*i*j + j, q_);
        }
    }

    return W;
}


}  // namespace generator

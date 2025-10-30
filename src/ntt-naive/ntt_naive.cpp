#include "generator.hpp"
#include <fmt/core.h>
#include <vector>

/*
    golden_matmul 함수는 tenstorrent에서 연산을 제대로 했는지 검증하기 위한 정답지를 계산하는 함수이다.
    (W_NxN * a^T) 를 계산한다.
*/
void golden_matmul(std::vector<uint32_t>& a, std::vector<uint32_t>& W, std::vector<uint32_t>& output, std::uint32_t N, std::uint32_t q) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            output.at(i) += (W.at(i * N + j) * a.at(j));
            output.at(i) %= q;
        }
    }
}

int main() {

    generator::NttHelper nttHelper = generator::NttHelper();
    std::vector<uint32_t> W = nttHelper.makeTwiddleFactor();

    fmt::print("N = 2^14, q = 65537 일 때의 primitive 2N-th root of unity - {}\n", W.at(1));

    std::vector<uint32_t> golden_output(nttHelper.N_);
    golden_matmul(nttHelper.arr_, W, golden_output, nttHelper.N_, nttHelper.q_);

    fmt::print("golden output b_0 = {}\n", golden_output.at(0));


    return 0;
}
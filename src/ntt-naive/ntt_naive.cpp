#include "generator.hpp"
#include <fmt/core.h>
#include <vector>
#include <random>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tilize_utils.hpp>
#include <tt-metalium/distributed.hpp>
#include <bmm_op.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt-metalium/core_coord.hpp"

using namespace tt::constants;
using namespace tt;
using namespace std;
using namespace tt::tt_metal;

/*
    golden_matmul 함수는 tenstorrent에서 연산을 제대로 했는지 검증하기 위한 정답지를 계산하는 함수이다.
    (W_NxN * a^T) 를 계산한다.
*/
void golden_matmul(std::vector<uint32_t>& a, std::vector<uint32_t>& W, std::vector<uint32_t>& output, std::uint32_t N, std::uint32_t q) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            output.at(i) += (W.at(i * N + j) * a.at(j * TILE_WIDTH));
            output.at(i) %= q;
        }
    }
}

// 1단계 32-bit matrix를 8-bit matrix 4개로 분할한다.
void matrix_seg(std::vector<uint32_t>& a, std::vector<uint8_t>& b, std::vector<uint8_t>& c, std::vector<uint8_t>& d, std::vector<uint8_t>& e) {
    const std::size_t n = a.size();
    for (std::size_t i = 0; i < n; ++i) {
        const uint32_t value = a[i];
        e[i] = static_cast<uint8_t>(value & 0xFF);
        d[i] = static_cast<uint8_t>((value >> 8) & 0xFF);
        c[i] = static_cast<uint8_t>((value >> 16) & 0xFF);
        b[i] = static_cast<uint8_t>((value >> 24) & 0xFF);
    }
}


int main() {

    generator::NttHelper nttHelper = generator::NttHelper();
    std::vector<uint32_t> W = nttHelper.makeTwiddleFactor();

    // 2N-th root of unity = 858584
    fmt::print("N = 2^16, q = 8650753 일 때의 primitive 2N-th root of unity - {}\n", W.at(1));

    std::vector<uint32_t> golden_output(nttHelper.N_);
    golden_matmul(nttHelper.arr_, W, golden_output, nttHelper.N_, nttHelper.q_);

    fmt::print("golden output b_0 = {}\n", golden_output.at(0));

    // 1단계
    // 원본 행렬 X = a b c d 위치대로 분해된다. (32-bit = 8 8 8 8)
    std::vector<uint8_t> arr_a(nttHelper.arr_.size());
    std::vector<uint8_t> arr_b(nttHelper.arr_.size());
    std::vector<uint8_t> arr_c(nttHelper.arr_.size());
    std::vector<uint8_t> arr_d(nttHelper.arr_.size());

    matrix_seg(nttHelper.arr_, arr_a, arr_b, arr_c, arr_d);

    std::vector<uint8_t> W_a(W.size());
    std::vector<uint8_t> W_b(W.size());
    std::vector<uint8_t> W_c(W.size());
    std::vector<uint8_t> W_d(W.size());

    matrix_seg(W, W_a, W_b, W_c, W_d);

    fmt::print("matrix segmantation\n");
    fmt::print("W = {}, W_a = {}, W_b = {}, W_c = {}, W_d = {} , index at = {}\n", 
        W.at(2), W_a.at(2), W_b.at(2), W_c.at(2), W_d.at(2), 2);
    fmt::print("a = {}, a_a = {}, a_b = {}, a_c = {}, a_d = {} , index at = {}\n", 
        nttHelper.arr_.at(0), arr_a.at(0), arr_b.at(0), arr_c.at(0), arr_d.at(0), 0);
    

    // 2단계
    // tenstorrent FPU에서 W * a 의 행렬 곱을 수행한다.

    

    return 0;
}

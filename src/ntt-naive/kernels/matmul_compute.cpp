#include <cstdint>
#include <vector>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "hostdevcommon/kernel_structs.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api.h"

using std::uint32_t;

#ifdef TRISC_MATH

inline void fusion_tile_face(uint32_t m, uint32_t q) {
    constexpr size_t vecotrs_per_face = 8;
    for (size_t i = 0; i < vectors_per_face; i++) {
        vInt x = dst_reg[i];
        x *= m;
        x %= q;
        dst_reg[i];
    }
}

inline void my_add_tile_face(const uint32_t dst_index_in0, const uint32_t dst_index_in1, const uint32_t dst_index_out, uint32_t q) {
    constexpr uint32_t n_vector_in_tile = 32;

    const uint32_t in0_base_idx = dst_index_in0 * n_vector_in_tile;
    const uint32_t in1_base_idx = dst_index_in1 * n_vector_in_tile;
    const uint32_t out_base_idx = dst_index_out * n_vector_in_tile;

    for (size_t i = 0; i < 8; i++) {
        vInt a = dst_reg[in0_base_idx + i];
        vInt b = dst_reg[in1_base_idx + i];

        dst_reg[out_base_idx + i] = (a + b) % q;
    }
}

#endif



inline void fusion_tile(uint32_t idx_dst0, uint32_t m, uint32_t q) {
    MATH(_llk_math_eltwise_unary_sfpu_params_<false>(
        fusion_tile_face, idx_dst0, VectorMode::RC, m, q
    ));
}

inline void my_add_tile(uint32_t idx_dst0, uint32_t idx_dst1, uint32_t idx_out0, uint32_t q) {
    MATH(_llk_math_eltwise_binary_sfpu_params_<false>(my_add_tile_face, idx_dst0, idx_dst1, idx_out0, q));
}

namespace NAMESPACE {

void MAIN {
    const uint32_t Nt = get_compile_time_arg_val(0);
    std::vector<tt::CBIndex> w_cb_in_vec(4);
    w_cb_in_vec.at(0) = tt::CBIndex::c_0;
    w_cb_in_vec.at(1) = tt::CBIndex::c_1;
    w_cb_in_vec.at(2) = tt::CBIndex::c_2;
    w_cb_in_vec.at(3) = tt::CBIndex::c_3;
    std::vector<tt::CBIndex> a_cb_in_vec(4);
    a_cb_in_vec.at(0) = tt::CBIndex::c_4;
    a_cb_in_vec.at(1) = tt::CBIndex::c_5;
    a_cb_in_vec.at(2) = tt::CBIndex::c_6;
    a_cb_in_vec.at(3) = tt::CBIndex::c_7;

    uint32_t pow256_mod[7];
    for(int i = 0; i < 7; i++) {
        pow256_mod[i] = get_arg_val<uint32_t>(i);
    }

    uint32_t q = get_arg_val<uint32_t>(7);

    constexpr tt::CBIndex cb_out = tt::CBIndex::c_16;   //output

    // Setup the FPU (matrix engine) for the matmul operation. And specify the input
    // and output circular buffers.
    for(uint32_t i = 0; i < Nt; ++i) {
        tile_regs_acquire();
        for(int w = 0; w < 4; w++) {
            for(int a = 0; a < 4; a++) {
                for (uint32_t j = 0; j < Nt; ++j) {
                    mm_init(w_cb_in_vec.at(w), a_cb_in_vec.at(a), cb_out);

                    cb_wait_front(w_cb_in_vec.at(w), 1);
                    cb_wait_front(a_cb_in_vec.at(a), 1);

                    // 1번 dst register에는 seg tile의 fusion 값들이 누적으로 더해진다.
                    if(w == 0 && a == 0) matmul_tiles(w_cb_in_vec.at(w), a_cb_in_vec.at(a), 0, 0, 1, false);
                    // 0번 dst register에는 현재 seg tile의 mat mul 값이 누적된다.
                    matmul_tiles(w_cb_in_vec.at(w), a_cb_in_vec.at(a), 0, 0, 0, false);

                    cb_pop_front(w_cb_in_vec.at(w), 1);
                    cb_pop_front(a_cb_in_vec.at(a), 1);
                }
                if(w == 0 && a == 0) continue;
                init_sfpu(w_cb_invec.at(w), a_cb_in_vec.at(a));
                fusion_tile(0, pow256_mod[w + a], q);
                my_add_tile(0, 1, 1);
            }
        }
        tile_regs_commit();
        tile_regs_wait();

        // Ensure the output circular buffer has space for the result tile.
        cb_reserve_back(cb_out, 1);
        // Pack the result tile into the output circular buffer.
        pack_tile(1, cb_out);
        // Mark the output tile as ready so the writer can read it.
        cb_push_back(cb_out, 1);

        // We don't need the registers anymore, so we can release them and prepare for the next output tile.
        tile_regs_release();
    }

        
}
}  // namespace NAMESPACE
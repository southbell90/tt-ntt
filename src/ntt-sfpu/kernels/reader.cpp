#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    // same arg indices as in reader_binary_diff_lengths for compat
    uint32_t W_seg_0_addr = get_arg_val<uint32_t>(0);
    uint32_t W_seg_1_addr = get_arg_val<uint32_t>(1);
    uint32_t W_seg_2_addr = get_arg_val<uint32_t>(2);
    uint32_t W_seg_3_addr = get_arg_val<uint32_t>(3);
    uint32_t a_seg_0_addr = get_arg_val<uint32_t>(4);
    uint32_t a_seg_1_addr = get_arg_val<uint32_t>(5);
    uint32_t a_seg_2_addr = get_arg_val<uint32_t>(6);
    uint32_t a_seg_3_addr = get_arg_val<uint32_t>(7);
    uint32_t Nt = get_arg_val<uint32_t>(8);

    uint32_t cb_id_in_arr[8];
    for(int i = 0; i < 8; i++) {
        cb_id_in_arr[i] = i;
    }

    // Declare address in which we stored the source matrices. We have set the exact same format between CBs and DRAM
    // buffers in the host code, so we can use the same address for both DRAM and CBs.
    constexpr auto s0_args = TensorAccessorArgs<0>();
    auto s0 = TensorAccessor(s0_args, W_seg_0_addr, get_tile_size(cb_id_in_arr[0]));
    constexpr auto s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    auto s1 = TensorAccessor(s1_args, W_seg_1_addr, get_tile_size(cb_id_in_arr[1]));
    constexpr auto s2_args = TensorAccessorArgs<s1_args.next_compile_time_args_offset()>();
    auto s2 = TensorAccessor(s2_args, W_seg_2_addr, get_tile_size(cb_id_in_arr[2]));
    constexpr auto s3_args = TensorAccessorArgs<s2_args.next_compile_time_args_offset()>();
    auto s3 = TensorAccessor(s3_args, W_seg_3_addr, get_tile_size(cb_id_in_arr[3]));
    constexpr auto s4_args = TensorAccessorArgs<s3_args.next_compile_time_args_offset()>();
    auto s4 = TensorAccessor(s4_args, a_seg_0_addr, get_tile_size(cb_id_in_arr[4]));
    constexpr auto s5_args = TensorAccessorArgs<s4_args.next_compile_time_args_offset()>();
    auto s5 = TensorAccessor(s5_args, a_seg_1_addr, get_tile_size(cb_id_in_arr[5]));
    constexpr auto s6_args = TensorAccessorArgs<s5_args.next_compile_time_args_offset()>();
    auto s6 = TensorAccessor(s6_args, a_seg_2_addr, get_tile_size(cb_id_in_arr[6]));
    constexpr auto s7_args = TensorAccessorArgs<s6_args.next_compile_time_args_offset()>();
    auto s7 = TensorAccessor(s7_args, a_seg_3_addr, get_tile_size(cb_id_in_arr[7]));

    auto read_tile_from_accessor = [&](int accessor_idx, uint32_t tile_index, uint32_t l1_addr) {
        switch (accessor_idx) {
            case 0: noc_async_read_tile(tile_index, s0, l1_addr); break;
            case 1: noc_async_read_tile(tile_index, s1, l1_addr); break;
            case 2: noc_async_read_tile(tile_index, s2, l1_addr); break;
            case 3: noc_async_read_tile(tile_index, s3, l1_addr); break;
            case 4: noc_async_read_tile(tile_index, s4, l1_addr); break;
            case 5: noc_async_read_tile(tile_index, s5, l1_addr); break;
            case 6: noc_async_read_tile(tile_index, s6, l1_addr); break;
            case 7: noc_async_read_tile(tile_index, s7, l1_addr); break;
            default: ASSERT(false && "Invalid tensor accessor index");
        }
    };

    for (uint32_t i = 0; i < Nt; i++) {
        for(int w = 0; w < 4; w++) {
            for(int a = 0; a < 4; a++) {
                for (uint32_t j = 0; j < Nt; j++) {
                    uint32_t w_tile_index = i * Nt + j;  
                    uint32_t a_tile_index = j; 
                    {                                          
                        cb_reserve_back(cb_id_in_arr[w], 1);
                        uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in_arr[w]);
                        read_tile_from_accessor(w, w_tile_index, l1_write_addr_in0);
                        noc_async_read_barrier();
                        cb_push_back(cb_id_in_arr[w], 1);
                    }

                    {                                          
                        cb_reserve_back(cb_id_in_arr[a+4], 1);
                        uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in_arr[a+4]);
                        read_tile_from_accessor(a + 4, a_tile_index, l1_write_addr_in1);
                        noc_async_read_barrier();
                        cb_push_back(cb_id_in_arr[a+4], 1);
                    }
                }    
            }
        }
    }
}

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
void golden_matmul(std::vector<uint32_t>& a, std::vector<uint32_t>& W, std::vector<uint64_t>& output, std::uint32_t N, std::uint32_t q) {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            // q의 값이 2^16 보다 크면 2개의 곱이 2^32 uint 에서는 overflow가 날 수 있다.
            output.at(i) += ((uint64_t)W.at(i * N + j) * (uint64_t)a.at(j * TILE_WIDTH));
            output.at(i) %= q;
        }
    }
}


void matmul_single_core(
    const std::vector<std::vector<uint8_t>>& W_seg,
    const std::vector<std::vector<uint8_t>>& a_seg,
    std::vector<uint32_t>& output,
    uint32_t N,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t q
) {
    // Set up mesh command queue, workload, device range, and program. This is a single-core example using core {0,0}.
    distributed::MeshCommandQueue& cq = mesh_device->mesh_command_queue();
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    Program program{};
    // Core range from x: [0, 0] to y: [0, 0] (single core at {0, 0})
    CoreCoord core({0, 0});

    // Calcaulate the number of tiles for each dimension.
    uint32_t Nt = N / TILE_WIDTH;

    // Create DRAM buffers for the input and output data.
    uint32_t single_tile_size = sizeof(uint8_t) * TILE_HEIGHT * TILE_WIDTH;

    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = single_tile_size, .buffer_type = tt_metal::BufferType::DRAM};

    distributed::DeviceLocalBufferConfig out_dram_config{
        .page_size = single_tile_size * 4, .buffer_type = tt_metal::BufferType::DRAM};

    distributed::ReplicatedBufferConfig buffer_config_A{.size = sizeof(uint8_t) * W_seg.at(0).size()};
    distributed::ReplicatedBufferConfig buffer_config_B{.size = sizeof(uint8_t) * a_seg.at(0).size()};
    distributed::ReplicatedBufferConfig buffer_config_C{.size = sizeof(uint32_t) * output.size()};

    // W_seg_i_dram_buffer
    auto W_seg_0_dram_buffer = distributed::MeshBuffer::create(buffer_config_A, dram_config, mesh_device.get());
    auto W_seg_1_dram_buffer = distributed::MeshBuffer::create(buffer_config_A, dram_config, mesh_device.get());
    auto W_seg_2_dram_buffer = distributed::MeshBuffer::create(buffer_config_A, dram_config, mesh_device.get());
    auto W_seg_3_dram_buffer = distributed::MeshBuffer::create(buffer_config_A, dram_config, mesh_device.get());

    // a_seg_i_dram_buffer
    auto a_seg_0_dram_buffer = distributed::MeshBuffer::create(buffer_config_B, dram_config, mesh_device.get());
    auto a_seg_1_dram_buffer = distributed::MeshBuffer::create(buffer_config_B, dram_config, mesh_device.get());
    auto a_seg_2_dram_buffer = distributed::MeshBuffer::create(buffer_config_B, dram_config, mesh_device.get());
    auto a_seg_3_dram_buffer = distributed::MeshBuffer::create(buffer_config_B, dram_config, mesh_device.get());

    // dst_dram_buffer
    auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config_C, out_dram_config, mesh_device.get());

    tt::DataFormat cb_data_format = tt::DataFormat::UInt8;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t num_input_tiles = 2;

    // Circular buffer for matrix W_seg_i tiles
     uint32_t W_seg_0_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_W_seg_0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{W_seg_0_cb_index, cb_data_format}})
            .set_page_size(W_seg_0_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_W_seg_0_config);

    uint32_t W_seg_1_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_W_seg_1_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{W_seg_1_cb_index, cb_data_format}})
            .set_page_size(W_seg_1_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_W_seg_1_config);

    uint32_t W_seg_2_cb_index = CBIndex::c_2;
    CircularBufferConfig cb_W_seg_2_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{W_seg_2_cb_index, cb_data_format}})
            .set_page_size(W_seg_2_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_W_seg_2_config);

    uint32_t W_seg_3_cb_index = CBIndex::c_3;
    CircularBufferConfig cb_W_seg_3_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{W_seg_3_cb_index, cb_data_format}})
            .set_page_size(W_seg_3_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_W_seg_3_config);


    // Circular buffer for matrix a_seg_i tiles
    uint32_t a_seg_0_cb_index = CBIndex::c_4;
    CircularBufferConfig cb_a_seg_0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{a_seg_0_cb_index, cb_data_format}})
            .set_page_size(a_seg_0_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_a_seg_0_config);

    uint32_t a_seg_1_cb_index = CBIndex::c_5;
    CircularBufferConfig cb_a_seg_1_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{a_seg_1_cb_index, cb_data_format}})
            .set_page_size(a_seg_1_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_a_seg_1_config);

    uint32_t a_seg_2_cb_index = CBIndex::c_6;
    CircularBufferConfig cb_a_seg_2_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{a_seg_2_cb_index, cb_data_format}})
            .set_page_size(a_seg_2_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_a_seg_2_config);

    uint32_t a_seg_3_cb_index = CBIndex::c_7;
    CircularBufferConfig cb_a_seg_3_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{a_seg_3_cb_index, cb_data_format}})
            .set_page_size(a_seg_3_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_a_seg_3_config);


    // Circular buffer for output tiles
    uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size * 4, {{output_cb_index, tt::DataFormat::UInt32}})
            .set_page_size(output_cb_index, single_tile_size * 4);
    tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    // Create the data movement kernels and the compute kernel
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*W_seg_0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*W_seg_1_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*W_seg_2_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*W_seg_3_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*a_seg_0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*a_seg_1_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*a_seg_2_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*a_seg_3_dram_buffer).append_to(reader_compile_time_args);
    auto reader_id = tt_metal::CreateKernel(
        program,
        "/home/southbell/tt-ntt/src/ntt-sfpu/kernels/reader.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
    auto writer_id = tt_metal::CreateKernel(
        program,
        "/home/southbell/tt-ntt/src/ntt-sfpu/kernels/writer.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = writer_compile_time_args});

    // Compile time arguments for the kernels
    // Note that these take effect at the kernel's compile time. Chaning these values will require recompilation of the
    // kernel. Having arguments at compile time allows the compiler to optimize the kernel for the specific use case.
    // Like applying loop unrolling, constant folding, etc.. resulting in a more efficient kernel.
    std::vector<uint32_t> compute_compile_time_args = {
        Nt   // Nt
    };
    auto compute_id = tt_metal::CreateKernel(
        program,
        "/home/southbell/tt-ntt/src/ntt-sfpu/kernels/matmul_compute.cpp",
        core,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .fp32_dest_acc_en = true, .compile_args = compute_compile_time_args, }   // use 32-bit dst registers
        );

    // Set kernel arguments
    uint32_t W_seg_0_addr = W_seg_0_dram_buffer->address();
    uint32_t W_seg_1_addr = W_seg_1_dram_buffer->address();
    uint32_t W_seg_2_addr = W_seg_2_dram_buffer->address();
    uint32_t W_seg_3_addr = W_seg_3_dram_buffer->address();
    uint32_t a_seg_0_addr = a_seg_0_dram_buffer->address();
    uint32_t a_seg_1_addr = a_seg_1_dram_buffer->address();
    uint32_t a_seg_2_addr = a_seg_2_dram_buffer->address();
    uint32_t a_seg_3_addr = a_seg_3_dram_buffer->address();
    uint32_t dst_addr = dst_dram_buffer->address();
    tt_metal::SetRuntimeArgs(program, reader_id, core, {W_seg_0_addr, W_seg_1_addr, W_seg_2_addr, W_seg_3_addr,
        a_seg_0_addr, a_seg_1_addr, a_seg_2_addr, a_seg_3_addr, Nt});
    tt_metal::SetRuntimeArgs(program, writer_id, core, {dst_addr, Nt});

    uint32_t pow256_mod[7];
    pow256_mod[0] = 1;
    for(int i = 1; i < 7; i++) {
        pow256_mod[i] = static_cast<uint32_t>(((uint64_t)pow256_mod[i-1] * 256ULL) % q);
    }

    tt_metal::SetRuntimeArgs(program, compute_id, core, 
        {pow256_mod[0], pow256_mod[1], pow256_mod[2], pow256_mod[3], pow256_mod[4], pow256_mod[5], pow256_mod[6], q});

    // Upload the input data to the DRAM buffers, execute the kernels, wait for the result to be read into the output
    // buffer
    distributed::EnqueueWriteMeshBuffer(cq, W_seg_0_dram_buffer, W_seg.at(0), false);
    distributed::EnqueueWriteMeshBuffer(cq, W_seg_1_dram_buffer, W_seg.at(1), false);
    distributed::EnqueueWriteMeshBuffer(cq, W_seg_2_dram_buffer, W_seg.at(2), false);
    distributed::EnqueueWriteMeshBuffer(cq, W_seg_3_dram_buffer, W_seg.at(3), false);
    distributed::EnqueueWriteMeshBuffer(cq, a_seg_0_dram_buffer, a_seg.at(0), false);
    distributed::EnqueueWriteMeshBuffer(cq, a_seg_1_dram_buffer, a_seg.at(1), false);
    distributed::EnqueueWriteMeshBuffer(cq, a_seg_2_dram_buffer, a_seg.at(2), false);
    distributed::EnqueueWriteMeshBuffer(cq, a_seg_3_dram_buffer, a_seg.at(3), false);
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::EnqueueReadMeshBuffer(cq, output, dst_dram_buffer, true);
}


// 1단계 32-bit matrix를 8-bit matrix 4개로 분할한다. a = b c d e
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
    // fmt::print("N = 2^16, q = 8650753 일 때의 primitive 2N-th root of unity - {}\n", W.at(1));

    std::vector<uint64_t> golden_output(nttHelper.N_);
    golden_matmul(nttHelper.arr_, W, golden_output, nttHelper.N_, nttHelper.q_);

    fmt::print("golden output b_0 = {}\n", golden_output.at(0));

    fmt::print("===== stage 1 start =====\n");

    // 1단계
    // 원본 행렬 X = y[3] y[2] y[1] y[0] 위치대로 분해된다. (32-bit = 8 8 8 8)
    std::vector<std::vector<uint8_t>> arr_seg(4, std::vector<uint8_t>(nttHelper.arr_.size()));

    matrix_seg(nttHelper.arr_, arr_seg.at(3), arr_seg.at(2), arr_seg.at(1), arr_seg.at(0));

    std::vector<std::vector<uint8_t>> W_seg(4, std::vector<uint8_t>(W.size()));

    matrix_seg(W, W_seg.at(3), W_seg.at(2), W_seg.at(1), W_seg.at(0));

    fmt::print("===== stage 1 complete =====\n");
    
    fmt::print("===== stage 2 start =====\n");

    // 2단계
    // tenstorrent FPU 에서 W * a 의 행렬 곱을 수행한다.
    bool pass = true;
    
    constexpr int device_id = 0;
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

    uint32_t N = nttHelper.N_;   // N = 2^16
    const uint32_t q = nttHelper.q_;

    for(int i = 0; i < 4; i++) {
        W_seg.at(i) = tilize_nfaces(W_seg.at(i), N, N);
        arr_seg.at(i) = tilize_nfaces(arr_seg.at(i), N, TILE_WIDTH);
    }

    std::vector<uint32_t> result_vec(N * TILE_WIDTH, 0);

    matmul_single_core(W_seg, arr_seg, result_vec, N, mesh_device, nttHelper.q_);
    result_vec = untilize_nfaces(result_vec, N, TILE_WIDTH);


    //golden_matmul의 실행결과와 FPU에서 실행한 결과를 비교한다.
    for(size_t i = 0; i < golden_output.size(); i++) {
        if(golden_output.at(i) != result_vec.at(i * TILE_WIDTH)) {
            fmt::print("test result invalid -- index : {}, golden_output = {}, result = {}\n", i, golden_output.at(i), result_vec.at(i * 32));
            pass = false;
            break;
        }
    }

    pass &= mesh_device->close();

    if (pass) {
        fmt::print("Test Passed!! ---- ntt_sfpu\n");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}

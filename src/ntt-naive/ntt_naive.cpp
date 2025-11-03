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
            output.at(i) += ((uint64_t)W.at(i * N + j) * (uint64_t)a.at(j * TILE_WIDTH));
            output.at(i) %= q;
        }
    }
}


void matmul_single_core(
    const std::vector<uint8_t>& W,
    const std::vector<uint8_t>& a,
    std::vector<uint32_t>& output,
    uint32_t N,
    const std::shared_ptr<distributed::MeshDevice>& mesh_device) {
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

    distributed::ReplicatedBufferConfig buffer_config_A{.size = sizeof(uint8_t) * W.size()};
    distributed::ReplicatedBufferConfig buffer_config_B{.size = sizeof(uint8_t) * a.size()};
    distributed::ReplicatedBufferConfig buffer_config_C{.size = sizeof(uint32_t) * output.size()};

    auto src0_dram_buffer = distributed::MeshBuffer::create(buffer_config_A, dram_config, mesh_device.get());
    auto src1_dram_buffer = distributed::MeshBuffer::create(buffer_config_B, dram_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config_C, out_dram_config, mesh_device.get());

    tt::DataFormat cb_data_format = tt::DataFormat::UInt8;
    MathFidelity math_fidelity = MathFidelity::HiFi4;
    uint32_t src0_cb_index = CBIndex::c_0;
    uint32_t num_input_tiles = 2;

    // Circular buffer for matrix A tiles
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src0_config);


    // Circular buffer for matrix B tiles
    uint32_t src1_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, cb_data_format}})
            .set_page_size(src1_cb_index, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_src1_config);


    // Circular buffer for output tiles
    uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = 2;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size * 4, {{output_cb_index, tt::DataFormat::UInt32}})
            .set_page_size(output_cb_index, single_tile_size * 4);
    tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    // Create the data movement kernels and the compute kernel
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_dram_buffer).append_to(reader_compile_time_args);
    auto reader_id = tt_metal::CreateKernel(
        program,
        "/home/southbell/tt-ntt/src/ntt-naive/kernels/reader.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
    auto writer_id = tt_metal::CreateKernel(
        program,
        "/home/southbell/tt-ntt/src/ntt-naive/kernels/writer.cpp",
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
    tt_metal::CreateKernel(
        program,
        "/home/southbell/tt-ntt/src/ntt-naive/kernels/matmul_compute.cpp",
        core,
        tt_metal::ComputeConfig{.math_fidelity = math_fidelity, .compile_args = compute_compile_time_args});

    // Set kernel arguments
    uint32_t src0_addr = src0_dram_buffer->address();
    uint32_t src1_addr = src1_dram_buffer->address();
    uint32_t dst_addr = dst_dram_buffer->address();
    tt_metal::SetRuntimeArgs(program, reader_id, core, {src0_addr, src1_addr, Nt});

    tt_metal::SetRuntimeArgs(program, writer_id, core, {dst_addr, Nt});

    // Upload the input data to the DRAM buffers, execute the kernels, wait for the result to be read into the output
    // buffer
    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, W, false);
    distributed::EnqueueWriteMeshBuffer(cq, src1_dram_buffer, a, false);
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
    // tenstorrent FPU에서 W * a 의 행렬 곱을 수행한다.
    bool pass = true;
    
    constexpr int device_id = 0;
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);

    uint32_t N = nttHelper.N_;   // N = 2^16
    const uint32_t q = nttHelper.q_;

    // static_assert(N % TILE_WIDTH == 0, "N must be divisible by TILE_WIDTH");

    for(int i = 0; i < 4; i++) {
        W_seg.at(i) = tilize_nfaces(W_seg.at(i), N, N);
        arr_seg.at(i) = tilize_nfaces(arr_seg.at(i), N, TILE_WIDTH);
    }


    // result_vec[k] = W_i * arr_j , where k = 4 * i + j
    std::vector<std::vector<uint32_t>> result_vec(16, std::vector<uint32_t>(N * TILE_WIDTH));

    for(int i = 0; i < W_seg.size(); i++) {
        for(int j = 0; j < arr_seg.size(); j++) {
            matmul_single_core(W_seg.at(i), arr_seg.at(j), result_vec.at(i * 4 + j), N, mesh_device);
            result_vec.at(i * 4 + j) = untilize_nfaces(result_vec.at(i * 4 + j), N, TILE_WIDTH);
        }
    }

    fmt::print("===== stage 2 complete =====\n");

    fmt::print("===== stage 3 start =====\n");

    // 3단계 8-bit integer 행렬들을 다시 32-bit integer 행렬 1개로 합친다.
    uint64_t pow256_mod[7];
    pow256_mod[0] = 1;
    for(int i = 1; i < 7; i++) {
        pow256_mod[i] = (pow256_mod[i-1] * 256ULL) % q;
    }

    std::vector<uint64_t> result(N * TILE_WIDTH, 0);
    for(size_t i = 0; i < result.size(); i++) {
        uint64_t temp_sum = 0;
        for(int j = 0; j < result_vec.size(); j++) {
            // uint32_t shift_amount = (j / 4 + j % 4) * 8;
            uint64_t accum = ((uint64_t)result_vec.at(j).at(i)* pow256_mod[j / 4 + j % 4]) % q;
            temp_sum += accum;
            temp_sum %= q;
        }
        result.at(i) = temp_sum;
    }

    fmt::print("===== stage 3 complete =====\n");

    //golden_matmul의 실행결과와 FPU에서 실행한 결과를 비교한다.
    for(size_t i = 0; i < golden_output.size(); i++) {
        if(golden_output.at(i) != result.at(i * TILE_WIDTH)) {
            fmt::print("test result invalid -- index : {}, golden_output = {}, result = {}\n", i, golden_output.at(i), result.at(i * 32));
            break;
        }
    }

    pass &= mesh_device->close();

    if (pass) {
        fmt::print("Test Passed!! ---- ntt_naive\n");
    } else {
        TT_THROW("Test Failed");
    }

    return 0;
}

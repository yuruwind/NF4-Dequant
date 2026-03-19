#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <stdint.h>
#include <algorithm>
#include <cmath>

// 错误检查宏
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    } \
}

// NF4 查找表：放入 __constant__ 内存
__constant__ float d_NF4_TABLE[16] = {
    -1.0f, -0.69487101f, -0.51209301f, -0.37391701f,
    -0.25611401f, -0.14725500f, -0.04162400f, 0.06282201f,
    0.16859101f, 0.28551400f, 0.40619302f, 0.53675699f,
    0.68502200f, 0.87091398f, 1.0f, 0.0f
};

// V1 Kernel: 每个线程处理一个字节 (2个权重)
__global__ void dequantize_nf4_kernel_v1(
    const uint8_t* packed_w,
    const uint8_t* absmax_q,
    const half* code2,
    const half* absmax2,
    half* output,
    int64_t total_elements,
    int block_size,
    int group_size
) {
    int64_t byte_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (byte_idx * 2 >= total_elements) return;

    uint8_t byte = packed_w[byte_idx];
    uint8_t idx0 = byte & 0x0F;
    uint8_t idx1 = (byte >> 4) & 0x0F;

    uint8_t indices[2] = {idx0, idx1};

    for (int j = 0; j < 2; ++j) {
        int64_t curr_idx = byte_idx * 2 + j;
        if (curr_idx < total_elements) {
            int32_t b_idx = curr_idx / block_size;
            int32_t g_idx = b_idx / group_size;

            float s1 = __half2float(code2[absmax_q[b_idx]]);
            float s2 = __half2float(absmax2[g_idx]);
            
            float val = d_NF4_TABLE[indices[j]] * s1 * s2;
            output[curr_idx] = __float2half(val);
        }
    }
}

int main() {
    // 1. 读取数据 (使用你修复后的手动读取逻辑)
    std::ifstream ifs("input.bin", std::ios::binary);
    if (!ifs) { std::cerr << "Cannot open input.bin\n"; return 1; }

    int64_t num_rows, num_cols;
    int32_t blocksize;
    ifs.read((char*)&num_rows, 8);
    ifs.read((char*)&num_cols, 8);
    ifs.read((char*)&blocksize, 4);

    int64_t total_elements = num_rows * num_cols;
    int32_t num_blocks = (total_elements + blocksize - 1) / blocksize;
    int32_t group_size = 256;
    int32_t num_groups = (num_blocks + group_size - 1) / group_size;

    std::vector<uint8_t> h_packed_w((total_elements + 1) / 2);
    std::vector<uint8_t> h_absmax_q(num_blocks);
    std::vector<half> h_code2(256);
    std::vector<half> h_absmax2(num_groups);
    float offset;

    ifs.read((char*)h_packed_w.data(), h_packed_w.size());
    ifs.read((char*)h_absmax_q.data(), h_absmax_q.size());
    ifs.read((char*)h_code2.data(), 256 * 2);
    ifs.read((char*)h_absmax2.data(), num_groups * 2);
    ifs.read((char*)&offset, 4);

    // 2. GPU 内存分配
    uint8_t *d_packed_w, *d_absmax_q;
    half *d_code2, *d_absmax2, *d_output;
    CHECK_CUDA(cudaMalloc(&d_packed_w, h_packed_w.size()));
    CHECK_CUDA(cudaMalloc(&d_absmax_q, h_absmax_q.size()));
    CHECK_CUDA(cudaMalloc(&d_code2, 256 * 2));
    CHECK_CUDA(cudaMalloc(&d_absmax2, num_groups * 2));
    CHECK_CUDA(cudaMalloc(&d_output, total_elements * 2));

    // 3. H2D 拷贝
    CHECK_CUDA(cudaMemcpy(d_packed_w, h_packed_w.data(), h_packed_w.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_absmax_q, h_absmax_q.data(), h_absmax_q.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_code2, h_code2.data(), 256 * 2, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_absmax2, h_absmax2.data(), num_groups * 2, cudaMemcpyHostToDevice));

    // 4. Kernel 启动与计时
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);

    int threads = 256;
    int64_t num_bytes = (total_elements + 1) / 2;
    int blocks = (num_bytes + threads - 1) / threads;

    dequantize_nf4_kernel_v1<<<blocks, threads>>>(
        d_packed_w, d_absmax_q, d_code2, d_absmax2, d_output,
        total_elements, blocksize, group_size
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // 5. 验证结果
    std::vector<half> h_output(total_elements);
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, total_elements * 2, cudaMemcpyDeviceToHost));

    // 读取 Ground Truth
    std::vector<half> h_gt(total_elements);
    std::ifstream gfs("gt_output.bin", std::ios::binary);
    gfs.read((char*)h_gt.data(), total_elements * 2);

    float max_err = 0;
    for(int i=0; i<total_elements; ++i) {
        float out_f = __half2float(h_output[i]);
        float gt_f = __half2float(h_gt[i]);
        max_err = std::max(max_err, std::abs(out_f - gt_f));
    }

    std::cout << "CUDA V1 Max Error: " << max_err << std::endl;
    std::cout << "Time: " << ms << " ms" << std::endl;
    std::cout << "Bandwidth: " << (total_elements * 0.5 + num_blocks + (num_blocks+num_groups)*2) / (ms * 1e6) << " GB/s" << std::endl;

    // 清理
    cudaFree(d_packed_w); cudaFree(d_absmax_q); cudaFree(d_code2); cudaFree(d_absmax2); cudaFree(d_output);
    return 0;
}
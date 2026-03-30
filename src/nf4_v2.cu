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
__constant__ float NF4_TABLE[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f, 0.33791524171829224f,
    0.44070982933044434f, 0.5626170039176941f, 0.7229568362236023f, 1.0f
};

__global__ void dequantize_nf4_kernel_v2(
    const uint8_t* packed_w,
    const uint8_t* absmax_q,
    const half* code2,
    const half* absmax2,
    half2* output,          // 注意：这里改为 half2 指针
    int64_t total_elements,
    int block_size,
    int group_size
) {
    // 每个线程处理 1 字节数据，对应 2 个权重
    int64_t byte_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    
    // total_elements / 2 是总的 byte 数，也是 half2 的个数
    if (byte_idx >= total_elements / 2) return;

    // 1. 读取 1 字节并解包
    uint8_t byte = packed_w[byte_idx];
    uint8_t idx0 = byte & 0x0F;
    uint8_t idx1 = (byte >> 4) & 0x0F;

    // 2. 获取缩放因子
    // 技巧：由于两个相邻元素通常属于同一个 block，我们可以减少计算
    int64_t curr_idx0 = byte_idx * 2;
    int32_t b_idx = curr_idx0 / block_size;
    int32_t g_idx = b_idx / group_size;

    float s1 = __half2float(code2[absmax_q[b_idx]]);
    float s2 = __half2float(absmax2[g_idx]);
    float final_scale = s1 * s2;

    // 3. 计算两个 half 结果
    half res0 = __float2half(NF4_TABLE[idx0] * final_scale);
    half res1 = __float2half(NF4_TABLE[idx1] * final_scale);

    // 4. 使用 make_half2 进行打包并向量化写入
    // 这一行在汇编层面会生成一条 STG.32 指令（一次性写 32 位）
    output[byte_idx] = make_half2(res0, res1);
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

    // 在 main 函数调用时：
    int threads = 256;
    int64_t num_elements = num_rows * num_cols;
    int64_t num_vectors = num_elements / 2; // half2 的总数
    int blocks = (num_vectors + threads - 1) / threads;

    dequantize_nf4_kernel_v2<<<blocks, threads>>>(
        d_packed_w, 
        d_absmax_q, 
        d_code2, 
        d_absmax2, 
        (half2*)d_output, // 强制转换为 half2 指针
        total_elements, 
        blocksize, 
        group_size
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
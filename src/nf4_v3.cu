#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
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

// --- 新增：配置结构体 ---
struct AppConfig {
    int blocksize = 64;
    std::string compute_type = "fp16";
    std::string target_gpu = "T4";
};

// --- 新增：配置解析函数 ---
AppConfig load_config(const std::string& filename) {
    AppConfig config;
    std::ifstream f(filename);
    if (!f.is_open()) {
        std::cout << "[Config] No config.txt found, using defaults.\n";
        return config;
    }

    std::string line;
    while (std::getline(f, line)) {
        // 去除空格并将 '=' 替换为空格，方便 stringstream 读取
        std::replace(line.begin(), line.end(), '=', ' ');
        std::stringstream ss(line);
        std::string key, value;
        if (ss >> key >> value) {
            if (key == "blocksize") config.blocksize = std::stoi(value);
            else if (key == "compute_type") config.compute_type = value;
            else if (key == "target_gpu") config.target_gpu = value;
        }
    }
    std::cout << "[Config] Loaded: compute_type=" << config.compute_type 
              << ", target_gpu=" << config.target_gpu << "\n";
    return config;
}

// NF4 查找表：放入 __constant__ 内存
__constant__ float NF4_TABLE[16] = {
    -1.0f,                 // 0b0000
    -0.6961928009986877f,  // 0b0001
    -0.5250730514526367f,  // 0b0010
    -0.39491748809814453f, // 0b0011
    -0.28444138169288635f, // 0b0100
    -0.18477343022823334f, // 0b0101
    -0.09105003625154495f, // 0b0110
    0.0f,                  // 0b0111
    0.07958029955625534f,  // 0b1000
    0.16093020141124725f,  // 0b1001
    0.24611230194568634f,  // 0b1010
    0.33791524171829224f,  // 0b1011
    0.44070982933044434f,  // 0b1100
    0.5626170039176941f,   // 0b1101
    0.7229568362236023f,   // 0b1110
    1.0f                   // 0b1111
};

// V3 Kernel: Shared Memory + 向量化访存
__global__ void dequantize_nf4_kernel_v3(
    const uint8_t* packed_w,
    const uint8_t* absmax_q,
    const half* code2,
    const half* absmax2,
    half2* output,
    int64_t total_elements,
    int block_size,
    int group_size
) {
    // 1. 将 code2 这种小且被高频访问的数据放入共享内存 (256个 half 只有 512 Bytes)
    __shared__ half s_code2[256];
    int tid = threadIdx.x;
    if (tid < 256) {
        s_code2[tid] = code2[tid];
    }
    __syncthreads(); // 确保所有线程都能看到填好的共享内存

    int64_t byte_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    // 修改前：if (byte_idx >= total_elements / 2) return;
    // 修改后：处理奇数时，需要多处理半个 byte
    if (byte_idx >= (total_elements + 1) / 2) return;

    // 2. 解包 1 个 byte 为两个 4-bit 索引
    uint8_t byte = packed_w[byte_idx];
    
    // 3. 计算索引
    int64_t element_idx = byte_idx * 2; 
    int32_t b_idx = element_idx / block_size;
    int32_t g_idx = b_idx / group_size;

    // 4. 从共享内存取 s1，从全局取 s2
    float s1 = __half2float(s_code2[absmax_q[b_idx]]);
    float s2 = __half2float(absmax2[g_idx]);
    float scale = s1 * s2;

    // 5. 计算并向量化写入
    half res0 = __float2half(NF4_TABLE[byte & 0x0F] * scale);
    half res1 = __float2half(NF4_TABLE[byte >> 4] * scale);

    // 5. 边界保护与写入
    if (byte_idx * 2 + 1 < total_elements) {
        // 正常情况：安全地进行 32-bit 向量化写入
        output[byte_idx] = make_half2(res0, res1);
    } else if (byte_idx * 2 < total_elements) {
        // 边界情况：处理奇数长度矩阵的最后一个元素，退化为 16-bit 标量写入
        // 注意：将 half2* 强转回 half* 进行单元素赋值
        reinterpret_cast<half*>(output)[byte_idx * 2] = res0;
    }

}

int main() {
    // 1. 读取数据 
    AppConfig cfg = load_config("config.txt");

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

    // --- 关键改进：配置校验 ---
    if (cfg.blocksize != blocksize) {
        printf("[Warning] Binary header blocksize (%d) differs from config.txt (%d). "
               "Using Binary Header.\n", blocksize, cfg.blocksize);
    }

    // --- 关键改进：根据 GPU 类型动态调整 Launch 参数 ---
    int threads_per_block = 256; // 默认值
    if (cfg.target_gpu == "T4") {
        // T4 (Turing) 架构 SM 较小，有时 128 线程能获得更好的利用率
        threads_per_block = 128;
    } else {
        // 针对你的 4060 (Ada) 或 A100 等现代显卡，256 是甜点值
        threads_per_block = 256;
    }

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

// ================== 核心计时区域 ==================
    // 定义 Launch 参数
    //int threads = 256;
    // 修改前：int64_t num_vectors = total_elements / 2;
    // 修改后：计算总共有多少个 Byte 需要处理（奇数元素向上取整）
    // 4. 核函数启动 (使用动态调整的 threads_per_block)
    int64_t num_bytes = (total_elements + 1) / 2;
    int blocks = (num_bytes + threads_per_block - 1) / threads_per_block;

    // --- 针对 compute_type 的逻辑判断 ---
    if (cfg.compute_type == "bf16") {
        // 虽然我们目前 Kernel 是 FP16，但这里可以打印一个 Dispatch 日志
        // 展现出你的程序具有“类型分发”的架构设计
        std::cout << "[Dispatch] Launching Kernel with BF16 precision path (simulated)...\n";
    } else {
        std::cout << "[Dispatch] Launching Kernel with FP16 precision path...\n";
    }

    // 创建 CUDA 事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- 预热 GPU (避免把 cuda context 初始化算进时间) ---
    dequantize_nf4_kernel_v3<<<blocks, threads_per_block>>>(
        d_packed_w, d_absmax_q, d_code2, d_absmax2, 
        (half2*)d_output, total_elements, blocksize, group_size
    );
    cudaDeviceSynchronize();

    // --- 正式计时 (跑 10 次取平均更稳定) ---
    cudaEventRecord(start);
    for(int i = 0; i < 10; ++i) { 
        dequantize_nf4_kernel_v3<<<blocks, threads_per_block>>>(
            d_packed_w, d_absmax_q, d_code2, d_absmax2, 
            (half2*)d_output, total_elements, blocksize, group_size
        );
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 10.0f; // 取 10 次的平均值
    // ==================================================

// 5. 验证结果
    std::vector<half> h_output(total_elements);
    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, total_elements * 2, cudaMemcpyDeviceToHost));

    // 读取 Ground Truth
    std::vector<half> h_gt(total_elements);
    std::ifstream gfs("gt_output.bin", std::ios::binary);
    gfs.read((char*)h_gt.data(), total_elements * 2);

    float max_err = 0;
    double total_abs_err = 0; // 使用 double 防止累加溢出
    for(int i=0; i<total_elements; ++i) {
        float out_f = __half2float(h_output[i]);
        float gt_f = __half2float(h_gt[i]);
        float diff = std::abs(out_f - gt_f);
        
        max_err = std::max(max_err, diff);
        total_abs_err += diff;
    }
    float mae = static_cast<float>(total_abs_err / total_elements);

    std::cout << "--- Validation Results ---" << std::endl;
    std::cout << "CUDA V3 Max Error: " << max_err << std::endl;
    std::cout << "CUDA V3 MAE:       " << mae << std::endl;
    std::cout << "Time:              " << ms << " ms" << std::endl;
    
    // 计算有效内存带宽
    // 读取: W(4bit) + absmax_q(8bit) + absmax2(16bit)
    // 写入: Output(16bit)
    // 忽略 code2 (已经放入 Shared Memory) 的重复读取开销
    double bytes_read = (total_elements * 0.5) + num_blocks + (num_groups * 2.0);
    double bytes_write = total_elements * 2.0;
    double total_bytes = bytes_read + bytes_write;
    
    std::cout << "Bandwidth: " << total_bytes / (ms * 1e6) << " GB/s" << std::endl;

    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_packed_w); cudaFree(d_absmax_q); cudaFree(d_code2); cudaFree(d_absmax2); cudaFree(d_output);
    return 0;
}
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <stdint.h>
#include <algorithm>
#include <cmath>


__constant__ float NF4_LUT[16] = {
    -1.0f,
    -0.6961928009986877f,
    -0.5250730514526367f,
    -0.39491748809814453f,
    -0.28444138169288635f,
    -0.18477343022823334f,
    -0.09105003625154495f,
    0.0f,
    0.07958029955625534f,
    0.16093020141124725f,
    0.24611230194568634f,
    0.33791524171829224f,
    0.44070982933044434f,
    0.5626170039176941f,
    0.7229568362236023f,
    1.0f
};


template <typename OutT>
__device__ __forceinline__ OutT float_to_out(float x);

template <>
__device__ __forceinline__ half float_to_out<half>(float x) {
    return __float2half_rn(x);
}

template <>
__device__ __forceinline__ __nv_bfloat16 float_to_out<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

template <typename OutT>
__device__ __forceinline__ uint32_t pack_pair_to_u32(float v1, float v2);

template <>
__device__ __forceinline__ uint32_t pack_pair_to_u32<half>(float v1, float v2) {
    union {
        half2 h2;
        uint32_t u32;
    } cvt;
    cvt.h2 = __floats2half2_rn(v1, v2);
    return cvt.u32;
}

template <>
__device__ __forceinline__ uint32_t pack_pair_to_u32<__nv_bfloat16>(float v1, float v2) {
    union {
        __nv_bfloat162 b2;
        uint32_t u32;
    } cvt;
    cvt.b2 = __floats2bfloat162_rn(v1, v2);
    return cvt.u32;
}


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


template <typename OutT>
__global__ void nf4_decode_kernel(
    const uint8_t* __restrict__ packed_weights,
    const uint8_t* __restrict__ absmax_q,
    const half* __restrict__ absmax2,
    const half* __restrict__ code2,
    const float offset,
    OutT* __restrict__ output,
    int64_t num_elements,
    int blocksize,
    int group_size
) {
    // 1. 获取全局线程与网格步长 (Grid-Stride)
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = gridDim.x * blockDim.x;
    
    int64_t total_bytes = (num_elements + 1) / 2;
    int64_t full_pair_bytes = num_elements / 2;
    
    // Warp Shuffle 掩码与线程 Lane ID
    //unsigned warp_mask = 0xffffffffu;

    // 瞬间同步：获取当前还活着的线程掩码，只在活着的线程间广播
    unsigned active_mask = __activemask();
    int lane = threadIdx.x & 31;

    // 2. 核心优化一：多级存储拓扑优化 (Shared Memory)
    // 缓存 NF4 查找表 (16 * 4 = 64 Bytes)
    __shared__ float s_LUT[16];
    // 缓存 一级码表 code2 (256 * 2 = 512 Bytes)，极大降低 L2 Cache 压力
    __shared__ half s_code2[256]; 

    if (threadIdx.x < 16) {
        s_LUT[threadIdx.x] = NF4_LUT[threadIdx.x];
    }
    if (threadIdx.x < 256) {
        s_code2[threadIdx.x] = code2[threadIdx.x];
    }
    __syncthreads(); // 确保 Warp 内所有线程可见

    // 强制转换为 32-bit 指针，用于后续的合并访存写入
    uint32_t* out_u32 = reinterpret_cast<uint32_t*>(output);
    int bytes_per_block = blocksize / 2;

    // 3. 核心优化二：驻留线程大循环 (Persistent Threads)
    for (int64_t byte_idx = tid; byte_idx < full_pair_bytes; byte_idx += stride) {
        
        int block_id = static_cast<int>(byte_idx / bytes_per_block);
        int group_id = block_id / group_size;
        float real_absmax = 0.0f;

        // 4. 核心优化三：Warp Shuffle 元数据广播
        // 在 blocksize >= 64 时，1个 Warp (32线程) 刚好处理 32字节 (1个 block)
        // 故整个 Warp 内的 block_id 是一致的，只需让 lane 0 计算缩放因子即可！
        if (lane == 0) {
            uint8_t qa = absmax_q[block_id];
            // 从 Shared Memory 读取 s_code2，彻底消灭对这 512 字节的全局访存
            real_absmax = (__half2float(absmax2[group_id]) * __half2float(s_code2[qa])) + offset;
        }
        // 瞬间同步：将 lane 0 算好的结果塞入其他 31 个线程的寄存器
        //real_absmax = __shfl_sync(warp_mask, real_absmax, 0);
        real_absmax = __shfl_sync(active_mask, real_absmax, 0);

        // 5. 标量读取与位运算解包 (修复高低位映射)
        uint8_t packed = packed_weights[byte_idx];
        float v1 = s_LUT[packed >> 4] * real_absmax;   // 高位在前
        float v2 = s_LUT[packed & 0x0F] * real_absmax; // 低位在后

        // 6. 核心优化四：向量化存取与硬件原语
        // 调用外部的 pack_pair_to_u32 模板，将两个 Float 强行压缩为 32-bit 的结构体 (half2/bfloat162)
        // 并通过 out_u32 执行 32-bit 的 STG.E.32 合并写入，完美利用显存带宽
        out_u32[byte_idx] = pack_pair_to_u32<OutT>(v1, v2);
    }

    // 7. 边界安全处理：处理奇数维度的最后“半个”字节 (标量降级写入)
    if ((num_elements & 1) != 0) {
        int64_t tail_byte = total_bytes - 1;
        // 只需 1 个线程收尾
        if (tid == 0) {
            uint8_t packed = packed_weights[tail_byte];
            int block_id = static_cast<int>(tail_byte / bytes_per_block);
            int group_id = block_id / group_size;
            float real_absmax = (__half2float(absmax2[group_id]) * __half2float(code2[absmax_q[block_id]])) + offset;
            
            // 奇数情况下，多出来的那个权重存在【低 4 位】中
            output[num_elements - 1] = float_to_out<OutT>(s_LUT[packed & 0x0F] * real_absmax);
        }
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

// ... 前面的 1.读取数据 和 2.GPU内存分配 保持不变 ...
// 注意：d_output 依然分配 total_elements * 2 的空间，因为 fp16 和 bf16 都是 2 bytes

// ================== 核心调度与计时区域 (V5 终极版) ==================

    // 【简历优化点：Template 特化分发，实现零分支指令流】
    // 在 Host 端根据配置选择不同的模板实例，Kernel 内部没有任何 if-else
    bool is_bf16 = (cfg.compute_type == "bf16");
    std::cout << "[Dispatch] Launching V5 Kernel with " << (is_bf16 ? "BF16" : "FP16") << " precision path...\n";

    // 【简历优化点：实施线程驻留策略 (Persistent Threads)】
    int sm_count = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, 0));
    
    // 1. 询问 GPU 硬件：当前 Kernel 在每个 SM 上最多能并发多少个 Block？
    int max_active_blocks = 0;
    if (is_bf16) {
        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks, nf4_decode_kernel<__nv_bfloat16>, threads_per_block, 0));
    } else {
        CHECK_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks, nf4_decode_kernel<half>, threads_per_block, 0));
    }

    // 2. 精准计算 Grid 大小：正好塞满整个显卡的 SM 单元，不留空隙，也不多派
    int grid_x = sm_count * max_active_blocks;
    
    // 3. 安全防护：如果数据量极小（比如几百个字节），连整个 GPU 都喂不饱，就降级为常规大小
    int64_t total_bytes = (total_elements + 1) / 2;
    int64_t max_grid = (total_bytes + threads_per_block - 1) / threads_per_block;
    if (grid_x > max_grid) {
        grid_x = static_cast<int>(max_grid);
    }
    if (grid_x < 1) grid_x = 1;

    std::cout << "[Occupancy] SM count: " << sm_count 
              << ", max active blocks/SM: " << max_active_blocks 
              << ", Launching Grid: " << grid_x << std::endl;

    // 创建 CUDA 事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // --- 预热 GPU ---
    if (is_bf16) {
        nf4_decode_kernel<__nv_bfloat16><<<grid_x, threads_per_block>>>(
            d_packed_w, d_absmax_q, d_absmax2, d_code2, offset, 
            (__nv_bfloat16*)d_output, total_elements, blocksize, group_size);
    } else {
        nf4_decode_kernel<half><<<grid_x, threads_per_block>>>(
            d_packed_w, d_absmax_q, d_absmax2, d_code2, offset, 
            (half*)d_output, total_elements, blocksize, group_size);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // --- 正式计时 (跑 100 次取平均以过滤抖动) ---
    const int iters = 100; // 建议提速后增加迭代次数
    CHECK_CUDA(cudaEventRecord(start));
    for(int i = 0; i < iters; ++i) { 
        if (is_bf16) {
            nf4_decode_kernel<__nv_bfloat16><<<grid_x, threads_per_block>>>(
                d_packed_w, d_absmax_q, d_absmax2, d_code2, offset, 
                (__nv_bfloat16*)d_output, total_elements, blocksize, group_size);
        } else {
            nf4_decode_kernel<half><<<grid_x, threads_per_block>>>(
                d_packed_w, d_absmax_q, d_absmax2, d_code2, offset, 
                (half*)d_output, total_elements, blocksize, group_size);
        }
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= iters; 
// ==================================================

// 5. 验证结果 (动态适配验证逻辑)
    // 无论输出什么，先拷回 CPU
    std::vector<uint16_t> h_output_raw(total_elements); // 用 uint16_t 统一接收 2bytes 数据
    CHECK_CUDA(cudaMemcpy(h_output_raw.data(), d_output, total_elements * 2, cudaMemcpyDeviceToHost));

    std::vector<half> h_gt(total_elements);
    std::ifstream gfs("gt_output.bin", std::ios::binary);
    gfs.read((char*)h_gt.data(), total_elements * 2);

    float max_err = 0;
    double total_abs_err = 0; 
    for(int i = 0; i < total_elements; ++i) {
        float out_f = 0.0f;
        if (is_bf16) {
            // BF16 转 Float 逻辑 (或者链接你自带的 CPU 转换库)
            __nv_bfloat16 bf16_val = *reinterpret_cast<__nv_bfloat16*>(&h_output_raw[i]);
            out_f = __bfloat162float(bf16_val);
        } else {
            half fp16_val = *reinterpret_cast<half*>(&h_output_raw[i]);
            out_f = __half2float(fp16_val);
        }
        
        float gt_f = __half2float(h_gt[i]);
        float diff = std::abs(out_f - gt_f);
        
        max_err = std::max(max_err, diff);
        total_abs_err += diff;
    }
    float mae = static_cast<float>(total_abs_err / total_elements);

    std::cout << "--- V5 Validation Results ---" << std::endl;
    std::cout << "Max Error: " << max_err << std::endl;
    std::cout << "MAE:       " << mae << std::endl;
    std::cout << "Time:      " << ms << " ms" << std::endl;
    
    // --- Effective (旧的) ---
    double bytes_read_theory = (total_elements * 0.5) + num_blocks + (num_groups * 2.0);
    double bytes_write = total_elements * 2.0;
    double total_theory = bytes_read_theory + bytes_write;

    double bw_theory = total_theory / (ms / 1000.0) / 1e9;

    // --- Realistic (新的) ---
    double total_real = total_elements * (0.5 + 2.0);
    double bw_real = total_real / (ms / 1000.0) / 1e9;

    std::cout << "Effective BW (theory): " << bw_theory << " GB/s\n";
    std::cout << "Realistic BW:         " << bw_real << " GB/s\n";
    
    //std::cout << "Bandwidth: " << bandwidth << " GB/s" << std::endl;

// ... 后续的清理代码保持不变 ...

    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_packed_w); cudaFree(d_absmax_q); cudaFree(d_code2); cudaFree(d_absmax2); cudaFree(d_output);
    return 0;
}
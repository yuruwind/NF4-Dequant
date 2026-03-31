import torch
import bitsandbytes as bnb
import argparse

def benchmark_bitsandbytes(rows, cols, blocksize=64, iters=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] 准备数据: {rows}x{cols}, BlockSize: {blocksize}")

    # ======================
    # 1. 生成数据
    # ======================
    W = torch.randn(rows, cols, dtype=torch.float16, device=device)

    W_q, quant_state = bnb.functional.quantize_4bit(
        W,
        quant_type='nf4',
        blocksize=blocksize,
        compress_statistics=True
    )

    total_elements = rows * cols
    num_blocks = (total_elements + blocksize - 1) // blocksize
    num_groups = (num_blocks + 255) // 256

    # ======================
    # 2. 预热
    # ======================
    print("[*] 正在预热 GPU...")
    for _ in range(20):
        bnb.functional.dequantize_4bit(W_q, quant_state)
    torch.cuda.synchronize()

    # ======================
    # 3. 计时
    # ======================
    print(f"[*] 开始测试 (迭代 {iters} 次)...")

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        bnb.functional.dequantize_4bit(W_q, quant_state)
    end_event.record()

    torch.cuda.synchronize()

    time_ms = start_event.elapsed_time(end_event) / iters

    # ======================
    # 4. 带宽计算（两种）
    # ======================

    # 原始（理论 IO，容易虚高）
    bytes_read_theory = (total_elements * 0.5) + num_blocks + (num_groups * 2.0)
    bytes_write = total_elements * 2.0
    total_theory = bytes_read_theory + bytes_write

    bw_theory = total_theory / (time_ms / 1000.0) / 1e9

    # 更真实（只统计必走 DRAM 的）
    # packed weights + output
    total_real = total_elements * (0.5 + 2.0)

    bw_real = total_real / (time_ms / 1000.0) / 1e9

    # ======================
    # 5. 输出
    # ======================
    print("\n" + "="*50)
    print("🚀 bitsandbytes Benchmark")
    print("="*50)
    print(f"矩阵规模: {rows}x{cols}")
    print(f"Kernel 耗时: {time_ms:.5f} ms")

    print("\n--- 带宽分析 ---")
    print(f"Effective Bandwidth (理论IO): {bw_theory:.2f} GB/s")
    print(f"Realistic Bandwidth (更真实): {bw_real:.2f} GB/s")

    print("\n--- IO 组成 ---")
    print(f"Packed Weights: {total_elements * 0.5 / 1e6:.2f} MB")
    print(f"Output:         {total_elements * 2.0 / 1e6:.2f} MB")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=16384)
    parser.add_argument("--cols", type=int, default=16384)
    parser.add_argument("--blocksize", type=int, default=64)
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    benchmark_bitsandbytes(
        args.rows,
        args.cols,
        args.blocksize,
        args.iters
    )
import torch
import bitsandbytes as bnb
import struct
import numpy as np
import os
import argparse

def generate_golden_data(rows, cols, blocksize=64, out_dir="./"):
    # 1. 参数设置
    # out_dir 默认与 C++ 可执行文件同目录
    
    print(f"[*] 正在生成 {rows}x{cols} 的随机权重矩阵...")
    # 使用正态分布生成权重，最符合 NF4 的设计初衷
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    W = torch.randn(rows, cols, dtype=torch.float16, device=device)
    
    # 2. 调用 bitsandbytes 官方量化接口
    print("[*] 调用 bitsandbytes 进行 NF4 双重量化...")
    # compress_statistics=True 开启双重量化 (Double Quantization)
    W_q, quant_state = bnb.functional.quantize_4bit(
        W, 
        quant_type='nf4', 
        blocksize=blocksize, 
        compress_statistics=True 
    )
    
    # 3. 生成官方真值 (Ground Truth)
    print("[*] 生成官方反量化真值 (Ground Truth)...")
    W_dq = bnb.functional.dequantize_4bit(W_q, quant_state)
    W_dq = W_dq.to(torch.float16) # 确保对齐 FP16
    
    # 4. 提取双重量化数据结构 (适配 C++ 端)
    print("[*] 提取底层量化状态 (QuantState)...")
    
    # 获取打包后的 4-bit 权重 (1 个 byte 存 2 个权重)
    packed_weights = W_q.cpu().numpy().astype(np.uint8)
    
    # bitsandbytes > 0.40 版本，双重量化状态保存在 state2 属性中
    qs2 = quant_state.state2 
    
    # 提取一级缩放因子索引 (8-bit)
    absmax_q = quant_state.absmax.cpu().numpy().astype(np.uint8)
    
    # 提取一级缩放因子码表 (code2, 256个 FP16)
    code2 = qs2.code.cpu().to(torch.float16).numpy()
    
    # 提取二级缩放因子 (absmax2, 每 256 个 block 共享一个 FP16)
    absmax2 = qs2.absmax.cpu().to(torch.float16).numpy()
    
    # 偏移量 (默认为 0)
    offset = 0.0
    if hasattr(quant_state, 'offset') and quant_state.offset is not None:
        offset = quant_state.offset.item()
        
    # 5. 写入二进制文件供 C++ 读取
    input_path = os.path.join(out_dir, "input.bin")
    print(f"[*] 正在写入 {input_path} ...")
    with open(input_path, 'wb') as f:
        # 写入 Header
        f.write(struct.pack('q', rows))
        f.write(struct.pack('q', cols))
        f.write(struct.pack('i', blocksize))
        
        # 写入 Data
        f.write(packed_weights.tobytes())
        f.write(absmax_q.tobytes())
        f.write(code2.tobytes())
        f.write(absmax2.tobytes())
        f.write(struct.pack('f', offset))
        
    gt_path = os.path.join(out_dir, "gt_output.bin")
    print(f"[*] 正在写入 {gt_path} ...")
    with open(gt_path, 'wb') as f:
        f.write(W_dq.cpu().numpy().tobytes())
        
    print(f"\n[+] 成功！数据准备完毕。")
    print(f"    - 参数规模: {rows * cols} 个元素")
    print(f"    - 一级 Blocks 数量: {len(absmax_q)}")
    print(f"    - 二级 Groups 数量: {len(absmax2)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 NF4 双重量化输入与真值数据")
    parser.add_argument("--rows", type=int, required=True, help="矩阵行数")
    parser.add_argument("--cols", type=int, required=True, help="矩阵列数")
    parser.add_argument("--blocksize", type=int, default=64, help="quantize blocksize，默认 64")
    parser.add_argument("--out_dir", type=str, default="./", help="输出目录，默认当前目录")
    args = parser.parse_args()

    generate_golden_data(
        rows=args.rows,
        cols=args.cols,
        blocksize=args.blocksize,
        out_dir=args.out_dir,
    )
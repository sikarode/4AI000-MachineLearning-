"""
diagnose_gpu.py — GPU 和数据诊断
==================================
运行: python diagnose_gpu.py --data_dir ./data
"""

import sys
import os
import glob
import time
import argparse

print("=" * 60)
print("1. PyTorch & CUDA 诊断")
print("=" * 60)

try:
    import torch
    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"  CUDA 版本: {torch.version.cuda}")
        print(f"  GPU 数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            mem = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
            print(f"    显存: {mem / 1024**3:.1f} GB")
            print(f"    计算能力: {props.major}.{props.minor}")
        
        print(f"  当前默认设备: {torch.cuda.current_device()}")
        print(f"  默认 GPU 名: {torch.cuda.get_device_name()}")
        
        # 实际计算测试
        print("\n  计算测试:")
        x = torch.randn(256, 256, device="cuda")
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            y = x @ x
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"    100次矩阵乘法: {(t1-t0)*1000:.1f} ms")
        print(f"    GPU 正在工作 ✓" if (t1-t0) < 1.0 else "    GPU 可能有问题 ✗")
        
        # 显存使用
        print(f"\n  显存使用:")
        print(f"    已分配: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        print(f"    已缓存: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
    else:
        print("  ⚠️ CUDA 不可用！PyTorch 可能是 CPU 版本")
        print("  检查: pip show torch | findstr Version")
        print("  如果版本不含 'cu'，需重装: pip install torch --index-url https://download.pytorch.org/whl/cu124")
        
except ImportError:
    print("  PyTorch 未安装！")
    sys.exit(1)

print("\n" + "=" * 60)
print("2. diffusers 版本")
print("=" * 60)
try:
    import diffusers
    print(f"  diffusers 版本: {diffusers.__version__}")
except ImportError:
    print("  diffusers 未安装！")

print("\n" + "=" * 60)
print("3. 数据目录诊断")
print("=" * 60)

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./data")
args, _ = parser.parse_known_args()

all_files = os.listdir(args.data_dir) if os.path.isdir(args.data_dir) else []
data_files = [f for f in all_files if f.endswith(".data")]
other_files = [f for f in all_files if not f.endswith(".data") and not f.startswith(".")]

print(f"  目录: {os.path.abspath(args.data_dir)}")
print(f"  总文件数: {len(all_files)}")
print(f"  .data 文件数: {len(data_files)}")

if other_files:
    exts = {}
    for f in other_files:
        ext = os.path.splitext(f)[1] or "(无扩展名)"
        exts[ext] = exts.get(ext, 0) + 1
    print(f"  其他扩展名: {exts}")
    print(f"  前5个非.data文件: {other_files[:5]}")

if data_files:
    print(f"  前5个 .data 文件: {sorted(data_files)[:5]}")
    print(f"  后5个 .data 文件: {sorted(data_files)[-5:]}")
    
    # Check glob pattern
    glob_result = glob.glob(os.path.join(args.data_dir, "*.data"))
    print(f"  glob('*.data') 匹配: {len(glob_result)} 个")
else:
    print("  ⚠️ 没有找到 .data 文件！")
    # 查看子目录
    subdirs = [f for f in all_files if os.path.isdir(os.path.join(args.data_dir, f))]
    if subdirs:
        print(f"  子目录: {subdirs[:10]}")
        for sd in subdirs[:3]:
            sub_data = glob.glob(os.path.join(args.data_dir, sd, "*.data"))
            if sub_data:
                print(f"    {sd}/ 中有 {len(sub_data)} 个 .data 文件")

print("\n  预期的增强后数据量:")
print(f"    {len(data_files)} 文件 × 4 旋转 = {len(data_files)*4} 样本")
print(f"    每 epoch {len(data_files)*4//4} 步 (batch_size=4)")

print("\n" + "=" * 60)
print("4. U-Net 模型大小测试")
print("=" * 60)

if torch.cuda.is_available():
    try:
        from inverse_design_diffusion import build_stage1_unet, StressStrainEncoder
        
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated()
        
        unet = build_stage1_unet(256).cuda()
        enc = StressStrainEncoder().cuda()
        
        mem_model = torch.cuda.memory_allocated() - mem_before
        n_params_unet = sum(p.numel() for p in unet.parameters())
        n_params_enc = sum(p.numel() for p in enc.parameters())
        
        print(f"  U-Net 参数量: {n_params_unet/1e6:.1f}M")
        print(f"  Encoder 参数量: {n_params_enc/1e6:.1f}M")
        print(f"  模型显存: {mem_model/1024**3:.2f} GB")
        
        # 测试一次前向传播
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated()
        
        with torch.amp.autocast("cuda"):
            fake_cond = torch.randn(4, 8, 64, device="cuda")
            fake_noisy = torch.randn(4, 2, 128, 128, device="cuda")
            fake_t = torch.randint(0, 500, (4,), device="cuda")
            
            cond_emb = enc(fake_cond)
            pred = unet(fake_noisy, fake_t, encoder_hidden_states=cond_emb).sample
        
        mem_peak = torch.cuda.max_memory_allocated()
        print(f"  前向传播峰值显存: {mem_peak/1024**3:.2f} GB")
        print(f"  剩余可用: {(8.0 - mem_peak/1024**3):.2f} GB")
        
        if mem_peak / 1024**3 > 7.5:
            print("  ⚠️ 显存几乎用满，建议 batch_size=2")
        elif mem_peak / 1024**3 > 6.0:
            print("  ✓ batch_size=4 可行但紧张")
        else:
            print("  ✓ 显存充足")
            
        # 计时
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            with torch.amp.autocast("cuda"):
                cond_emb = enc(fake_cond)
                pred = unet(fake_noisy, fake_t, encoder_hidden_states=cond_emb).sample
                loss = pred.mean()
            loss.backward()
        torch.cuda.synchronize()
        t1 = time.time()
        print(f"\n  5步训练耗时: {(t1-t0):.2f}s → {(t1-t0)/5:.2f}s/步")
        print(f"  预计 Stage 1 (500 epochs): {(t1-t0)/5 * len(data_files)*4//4 * 500 / 3600:.1f} 小时")
        
        del unet, enc
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"  模型测试失败: {e}")

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)

"""
Benchmark optimal batch size for RTX 4090 training.
Run on RunPod: python3 scripts/benchmark_runpod.py
"""
import os, sys, torch, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_float32_matmul_precision('high')

from configs.base_config import ProjectConfig, FusionConfig
from ferretnet.dual_branch import DualBranchFerretNet

proj = ProjectConfig()
device = torch.device('cuda')
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
print()

model = DualBranchFerretNet(str(proj.ferretnet_weights), FusionConfig()).to(device)
criterion = torch.nn.BCEWithLogitsLoss()

best_ips, best_bs = 0, 128
for bs in [64, 128, 256, 512, 1024]:
    try:
        torch.cuda.empty_cache()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        # warmup
        for _ in range(3):
            f = torch.randn(bs, 3, 256, 256, device=device)
            l = torch.randint(0, 2, (bs, 1), dtype=torch.float32, device=device)
            fl, fll = model(f, f)
            loss = criterion(fl, l) + criterion(fll, l)
            loss.backward(); opt.step(); opt.zero_grad()
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(20):
            f = torch.randn(bs, 3, 256, 256, device=device)
            l = torch.randint(0, 2, (bs, 1), dtype=torch.float32, device=device)
            fl, fll = model(f, f)
            loss = criterion(fl, l) + criterion(fll, l)
            loss.backward(); opt.step(); opt.zero_grad()
        torch.cuda.synchronize()

        sps = (time.time() - t0) / 20
        ips = bs / sps
        steps = 160000 // bs
        emin = steps * sps / 60
        vram = torch.cuda.max_memory_allocated() // 1024**2
        if ips > best_ips:
            best_ips, best_bs = ips, bs
        print(f"bs={bs:5d} | {ips:6.0f} img/s | {emin:4.1f} min/epoch | VRAM: {vram} MB | OK")
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"bs={bs:5d} | OOM")
            torch.cuda.empty_cache()
            break
        raise

print(f"\nBEST: bs={best_bs} -> {best_ips:.0f} img/s")
print(f"Estimated 15 epochs: {160000//best_bs * (best_ips and 1/best_ips or 0) * 15 / 60:.1f} min")

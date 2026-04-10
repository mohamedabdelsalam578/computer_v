import os, sys, torch, time
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
torch.set_float32_matmul_precision('high')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.base_config import ProjectConfig, FusionConfig
from ferretnet.dual_branch import DualBranchFerretNet

proj = ProjectConfig()
device = torch.device('mps')
print('Loading model...', flush=True)
model = DualBranchFerretNet(str(proj.ferretnet_weights), FusionConfig()).to(device)
criterion = torch.nn.BCEWithLogitsLoss()
print('Model loaded. Benchmarking...', flush=True)

best_ips, best_bs = 0, 64
for bs in [64, 96, 128]:
    try:
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        for _ in range(2):
            f = torch.randn(bs, 3, 256, 256, device=device)
            l = torch.randint(0, 2, (bs, 1), dtype=torch.float32, device=device)
            fl, fll = model(f, f)
            loss = criterion(fl, l) + criterion(fll, l)
            loss.backward(); opt.step(); opt.zero_grad()
        torch.mps.synchronize()
        t0 = time.time()
        for _ in range(8):
            f = torch.randn(bs, 3, 256, 256, device=device)
            l = torch.randint(0, 2, (bs, 1), dtype=torch.float32, device=device)
            fl, fll = model(f, f)
            loss = criterion(fl, l) + criterion(fll, l)
            loss.backward(); opt.step(); opt.zero_grad()
        torch.mps.synchronize()
        sps = (time.time() - t0) / 8
        ips = bs / sps
        emin = (160000 // bs) * sps / 60
        if ips > best_ips: best_ips, best_bs = ips, bs
        print(f'bs={bs:3d} | {ips:5.0f} img/s | {emin:4.0f} min/epoch | OK', flush=True)
    except Exception as e:
        print(f'bs={bs:3d} | OOM', flush=True)
        break

print(f'\nBEST: bs={best_bs} -> {best_ips:.0f} img/s', flush=True)

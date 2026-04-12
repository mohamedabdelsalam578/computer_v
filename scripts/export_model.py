"""
Export a trained checkpoint for transfer to Mac.

Extracts only model weights (no optimizer state) — much smaller download:
  clip     → ~350 MB  (vs ~1.5 GB full ckpt)
  vgg16    →  ~60 MB  (vs ~250 MB full ckpt)
  ferretnet → ~20 MB  (vs ~80 MB full ckpt)

The exported file works with the Streamlit app on any device.

Usage (on RunPod after training):
    python scripts/export_model.py                 # export CLIP (default)
    python scripts/export_model.py --model vgg16   # export VGG16
    python scripts/export_model.py --model ferretnet
    python scripts/export_model.py --out weights/my_model.pt
"""
import sys
import glob
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.base_config import ProjectConfig

MODELS = {
    'clip': {
        'ckpt_subdir': 'lightning_logs_clip/checkpoints',
        'prefix':      'clip',
        'out_name':    'clip_finetuned.pt',
    },
    'vgg16': {
        'ckpt_subdir': 'lightning_logs_vgg16/checkpoints',
        'prefix':      'vgg16',
        'out_name':    'vgg16_finetuned.pt',
    },
    'ferretnet': {
        'ckpt_subdir': 'lightning_logs/checkpoints',
        'prefix':      'ferretnet',
        'out_name':    'ferretnet_finetuned.pt',
    },
}


def find_best_ckpt(ckpt_dir: Path, prefix: str) -> Path:
    candidates = [p for p in glob.glob(str(ckpt_dir / f"{prefix}-*.ckpt"))
                  if 'last' not in p]
    if candidates:
        best = sorted(candidates, key=lambda p: p.split('=')[-1], reverse=True)[0]
        return Path(best)
    last = ckpt_dir / "last.ckpt"
    if last.exists():
        return last
    raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=list(MODELS.keys()), default='clip')
    parser.add_argument('--out',   type=str, default=None)
    args = parser.parse_args()

    proj = ProjectConfig()
    cfg  = MODELS[args.model]

    ckpt_dir  = proj.project_root / cfg['ckpt_subdir']
    ckpt_path = find_best_ckpt(ckpt_dir, cfg['prefix'])
    out_path  = Path(args.out) if args.out else proj.weights_dir / cfg['out_name']

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # Keep only model weights — drop optimizer / scheduler / epoch state
    export = {'state_dict': ckpt['state_dict']}
    if 'hyper_parameters' in ckpt:
        export['hyper_parameters'] = ckpt['hyper_parameters']

    proj.weights_dir.mkdir(parents=True, exist_ok=True)
    torch.save(export, out_path)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Exported → {out_path}  ({size_mb:.0f} MB)")
    print()
    print("Next steps:")
    print(f"  1. Download {out_path} to your Mac")
    print(f"  2. Place it at: AI_COMPUTER_VISION/weights/{cfg['out_name']}")
    print(f"  3. Run: streamlit run app/streamlit_app.py")


if __name__ == '__main__':
    main()

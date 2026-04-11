"""
Zero-shot evaluation of pretrained CLIP models on GRAVEX-200K val set.
Tests multiple text prompt variations and reports accuracy.

Run BEFORE training to see baseline pretrained accuracy.

Usage:
    python scripts/eval_pretrained_clip.py
    python scripts/eval_pretrained_clip.py --samples 2000   # quick 2k subset
    python scripts/eval_pretrained_clip.py --model ViT-B/32 # smaller/faster
"""
import sys
import csv
import json
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from configs.base_config import ProjectConfig

# CLIP normalization constants
CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

MODEL_MAP = {
    "ViT-L/14": "openai/clip-vit-large-patch14",
    "ViT-B/32": "openai/clip-vit-base-patch32",
}

# Text prompt pairs: (real_text, ai_text)
# Label 0 = real, label 1 = AI  →  high similarity to ai_text → predict 1
PROMPT_PAIRS = [
    ("a real photograph",
     "an AI generated image"),
    ("a genuine photo taken with a camera",
     "a synthetic image created by artificial intelligence"),
    ("a photo of a real person",
     "a photo of an AI generated face"),
    ("real",
     "fake"),
    ("a photograph of a real scene",
     "a digitally synthesized image"),
]


# ── Dataset ────────────────────────────────────────────────────────────────────

class EvalDataset(Dataset):
    def __init__(self, manifest_csv: str, transform, max_samples: int = None):
        self.transform = transform
        self.samples = []

        with open(manifest_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row['image_path'], int(row['label'])))
                if max_samples and len(self.samples) >= max_samples:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img), label
        except Exception:
            return torch.zeros(3, 224, 224), label


# ── Evaluation ──────────────────────────────────────────────────────────────────

def evaluate_zero_shot(model, processor_transform, data_loader, text_features, device):
    """Compute zero-shot accuracy: cosine similarity to text prompts."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="  evaluating", leave=False):
            images = images.to(device)

            image_features = model.get_image_features(pixel_values=images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Similarity to [real_text, ai_text] — shape (B, 2)
            logits = (100.0 * image_features @ text_features.T)
            probs  = logits.softmax(dim=-1)        # (B, 2)
            preds  = probs[:, 1].cpu().numpy() > 0.5  # index 1 = AI text

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    all_preds  = np.array(all_preds,  dtype=int)
    all_labels = np.array(all_labels, dtype=int)

    acc      = (all_preds == all_labels).mean()
    real_acc = (all_preds[all_labels == 0] == 0).mean()  # correctly called real
    ai_acc   = (all_preds[all_labels == 1] == 1).mean()  # correctly called AI
    bal_acc  = (real_acc + ai_acc) / 2.0

    return acc, real_acc, ai_acc, bal_acc


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=None,
                        help='Max images to evaluate (default: all val images)')
    parser.add_argument('--model', choices=list(MODEL_MAP.keys()), default='ViT-L/14')
    parser.add_argument('--batch_size', type=int, default=128)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    proj_cfg = ProjectConfig()
    val_csv  = str(proj_cfg.data_raw / "manifests" / "val_with_faces.csv")

    # ── Load CLIP model ──────────────────────────────────────────────────────
    model_name = MODEL_MAP[args.model]
    print(f"Loading {args.model} ({model_name}) ...")
    try:
        from transformers import CLIPModel, CLIPTokenizer
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install transformers")
        sys.exit(1)

    clip_model = CLIPModel.from_pretrained(model_name).to(device)
    tokenizer  = CLIPTokenizer.from_pretrained(model_name)
    clip_model.eval()

    # ── Build transform ──────────────────────────────────────────────────────
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

    # ── Load val dataset ─────────────────────────────────────────────────────
    dataset    = EvalDataset(val_csv, transform, max_samples=args.samples)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=(device == 'cuda'),
    )
    n_real = sum(1 for _, lbl in dataset.samples if lbl == 0)
    n_ai   = sum(1 for _, lbl in dataset.samples if lbl == 1)
    print(f"Evaluating on {len(dataset):,} images  ({n_real:,} real / {n_ai:,} AI)\n")

    # ── Evaluate each prompt pair ─────────────────────────────────────────────
    results = []
    print(f"{'Prompt (real vs AI)':<55}  {'Acc':>6}  {'RealAcc':>7}  {'AIAcc':>6}  {'BalAcc':>7}")
    print("─" * 88)

    best_acc = 0.0
    best_prompt = None

    for real_txt, ai_txt in PROMPT_PAIRS:
        tokens       = tokenizer([real_txt, ai_txt],
                                 return_tensors='pt', padding=True).to(device)
        with torch.no_grad():
            text_feats = clip_model.get_text_features(**tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        acc, real_acc, ai_acc, bal_acc = evaluate_zero_shot(
            clip_model, transform, dataloader, text_feats, device
        )

        label = f'"{real_txt[:22]}" vs "{ai_txt[:22]}"'
        print(f"  {label:<53}  {acc:6.2%}  {real_acc:7.2%}  {ai_acc:6.2%}  {bal_acc:7.2%}")
        results.append((real_txt, ai_txt, acc, real_acc, ai_acc, bal_acc))

        if bal_acc > best_acc:
            best_acc    = bal_acc
            best_prompt = (real_txt, ai_txt)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "═" * 88)
    print(f"  Best balanced accuracy: {best_acc:.2%}")
    print(f"  Best prompt:  real = \"{best_prompt[0]}\"")
    print(f"                AI   = \"{best_prompt[1]}\"")
    print("═" * 88)
    print()
    print("Interpretation:")
    print("  >80% → CLIP already understands AI vs real well; fine-tuning will push to 90%+")
    print("  60-80% → Some signal; fine-tuning essential")
    print("  ~50% → Random (prompt engineering failed); fine-tuning is the only path")
    print()


if __name__ == '__main__':
    main()

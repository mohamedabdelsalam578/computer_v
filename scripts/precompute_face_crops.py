"""
Pre-compute face crops for all GRAVEX-200K images using RetinaFace.

Run AFTER download_data.py. This is required before training.

Usage:
    python scripts/precompute_face_crops.py [--split train] [--limit N]

Output:
    data_raw/face_crops/{split}/{class}/{hash}_{idx}.png
    data_raw/manifests/{split}_with_faces.csv
"""
import os
import sys
import csv
import json
import hashlib
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.base_config import ProjectConfig, DataConfig, RetinaFaceConfig
from retinaface.detector import RetinaFaceDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default=None,
                        help='Process only this split (train/val/test). Default: all.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Max images per split (for testing)')
    args = parser.parse_args()

    cfg = ProjectConfig()
    data_cfg = DataConfig()
    rf_cfg = RetinaFaceConfig()

    # Load detector
    print("Loading RetinaFace detector...")
    device = "mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu"
    detector = RetinaFaceDetector(
        weights_path=str(cfg.retinaface_weights),
        device=device,
        confidence_threshold=rf_cfg.confidence_threshold,
        nms_threshold=rf_cfg.nms_threshold,
        vis_threshold=rf_cfg.vis_threshold,
        min_face_size=rf_cfg.min_face_size,
    )
    print("Detector loaded.")

    manifests_dir = cfg.data_raw / "manifests"
    crops_dir = cfg.data_raw / "face_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    splits = [args.split] if args.split else ['train', 'val', 'test']

    for split_name in splits:
        manifest_path = manifests_dir / f"{split_name}.csv"
        if not manifest_path.exists():
            print(f"Skipping {split_name}: {manifest_path} not found")
            continue

        # Read manifest
        rows = []
        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if args.limit:
            rows = rows[:args.limit]

        print(f"Processing {split_name}: {len(rows)} images...")

        # Build hash -> [crop_paths] map in ONE directory scan upfront.
        # No per-image filesystem calls during the main loop.
        done_map = {}  # hash12 -> [str(path), ...]
        for cn in ("real", "ai"):
            class_dir = crops_dir / split_name / cn
            if class_dir.exists():
                for p in class_dir.iterdir():
                    h = p.stem.rsplit('_', 1)[0]
                    done_map.setdefault(h, []).append(str(p))
        # Sort paths so face indices are in order
        for h in done_map:
            done_map[h].sort()
        print(f"  {len(done_map)} images already have crops on disk — skipping RetinaFace for those")

        output_rows = []
        for row in tqdm(rows, desc=f"Face crops [{split_name}]"):
            image_path = row['image_path']
            label = row['label']
            img_hash = hashlib.md5(image_path.encode()).hexdigest()[:12]
            class_name = "real" if int(label) == 0 else "ai"
            crop_save_dir = crops_dir / split_name / class_name
            crop_save_dir.mkdir(parents=True, exist_ok=True)

            # Pure in-memory lookup — zero filesystem calls
            if img_hash in done_map:
                crop_paths = done_map[img_hash]
                output_rows.append({
                    'image_path': image_path,
                    'label': label,
                    'num_faces': len(crop_paths),
                    'face_crop_paths': json.dumps(crop_paths),
                })
                continue

            if not os.path.exists(image_path):
                continue

            try:
                image = Image.open(image_path).convert('RGB')
            except Exception:
                continue

            # Detect and crop faces
            face_crops = detector.detect_faces(
                image,
                crop_size=data_cfg.face_crop_size,
                margin=data_cfg.face_margin,
                max_faces=5,
            )

            crop_paths = []
            for idx, crop in enumerate(face_crops):
                crop_filename = f"{img_hash}_{idx}.png"
                crop_path = crop_save_dir / crop_filename
                crop.save(str(crop_path))
                crop_paths.append(str(crop_path))

            output_rows.append({
                'image_path': image_path,
                'label': label,
                'num_faces': len(face_crops),
                'face_crop_paths': json.dumps(crop_paths),
            })

        # Write output manifest
        output_path = manifests_dir / f"{split_name}_with_faces.csv"
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image_path', 'label', 'num_faces', 'face_crop_paths'])
            writer.writeheader()
            writer.writerows(output_rows)

        n_with_faces = sum(1 for r in output_rows if r['num_faces'] > 0)
        print(f"  {split_name}: {n_with_faces}/{len(output_rows)} images have faces")


if __name__ == '__main__':
    main()

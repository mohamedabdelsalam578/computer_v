"""
Pre-compute face crops for all GRAVEX-200K images using RetinaFace.

Run AFTER download_data.py. This is required before training.

Usage:
    python scripts/precompute_face_crops.py [--split train] [--limit N] [--workers 8]
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
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from configs.base_config import ProjectConfig, DataConfig, RetinaFaceConfig


def process_batch(args):
    """Worker function — each process runs its own detector instance."""
    rows, crops_dir, split_name, done_map, weights_path, rf_cfg_dict, data_cfg_dict, device_str = args

    # Import here so each worker process gets its own copy
    from retinaface.detector import RetinaFaceDetector

    detector = RetinaFaceDetector(
        weights_path=weights_path,
        device=device_str,
        confidence_threshold=rf_cfg_dict['confidence_threshold'],
        nms_threshold=rf_cfg_dict['nms_threshold'],
        vis_threshold=rf_cfg_dict['vis_threshold'],
        min_face_size=rf_cfg_dict['min_face_size'],
    )

    output_rows = []
    for row in rows:
        image_path = row['image_path']
        label = row['label']
        img_hash = hashlib.md5(image_path.encode()).hexdigest()[:12]
        class_name = "real" if int(label) == 0 else "ai"
        crop_save_dir = Path(crops_dir) / split_name / class_name
        crop_save_dir.mkdir(parents=True, exist_ok=True)

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

        face_crops = detector.detect_faces(
            image,
            crop_size=data_cfg_dict['face_crop_size'],
            margin=data_cfg_dict['face_margin'],
            max_faces=5,
        )

        crop_paths = []
        for idx, crop in enumerate(face_crops):
            crop_filename = f"{img_hash}_{idx}.jpg"
            crop_path = crop_save_dir / crop_filename
            crop.save(str(crop_path), 'JPEG', quality=95)
            crop_paths.append(str(crop_path))

        output_rows.append({
            'image_path': image_path,
            'label': label,
            'num_faces': len(face_crops),
            'face_crop_paths': json.dumps(crop_paths),
        })

    return output_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split',   type=str, default=None)
    parser.add_argument('--limit',   type=int, default=None)
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: auto)')
    args = parser.parse_args()

    cfg     = ProjectConfig()
    data_cfg = DataConfig()
    rf_cfg  = RetinaFaceConfig()

    # RetinaFace workers always use CPU — CUDA cannot be shared across processes
    # (CUDA context cannot be forked/spawned safely with multiprocessing)
    # The RTX 4090 will be fully used during training instead.
    device_str = "cpu"
    default_workers = min(8, multiprocessing.cpu_count())
    num_workers = args.workers or default_workers
    print(f"Device: {device_str} | Workers: {num_workers}")

    manifests_dir = cfg.data_raw / "manifests"
    crops_dir     = cfg.data_raw / "face_crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    splits = [args.split] if args.split else ['train', 'val', 'test']

    rf_cfg_dict   = {'confidence_threshold': rf_cfg.confidence_threshold,
                     'nms_threshold': rf_cfg.nms_threshold,
                     'vis_threshold': rf_cfg.vis_threshold,
                     'min_face_size': rf_cfg.min_face_size}
    data_cfg_dict = {'face_crop_size': data_cfg.face_crop_size,
                     'face_margin': data_cfg.face_margin}

    for split_name in splits:
        manifest_path = manifests_dir / f"{split_name}.csv"
        if not manifest_path.exists():
            print(f"Skipping {split_name}: {manifest_path} not found")
            continue

        rows = []
        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

        if args.limit:
            rows = rows[:args.limit]

        print(f"Processing {split_name}: {len(rows)} images...")

        # Build done_map in one scan
        done_map = {}
        for cn in ("real", "ai"):
            class_dir = crops_dir / split_name / cn
            if class_dir.exists():
                for p in class_dir.iterdir():
                    if p.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                        h = p.stem.rsplit('_', 1)[0]
                        done_map.setdefault(h, []).append(str(p))
        for h in done_map:
            done_map[h].sort()
        print(f"  {len(done_map)} already cached — skipping those")

        # Split rows into batches for each worker
        chunk_size = max(1, len(rows) // num_workers)
        chunks = [rows[i:i+chunk_size] for i in range(0, len(rows), chunk_size)]

        all_output_rows = []
        weights_path = str(cfg.retinaface_weights)

        # Use spawn context for CUDA compatibility
        ctx = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            futures = [
                executor.submit(process_batch, (
                    chunk, str(crops_dir), split_name, done_map,
                    weights_path, rf_cfg_dict, data_cfg_dict, device_str
                ))
                for chunk in chunks
            ]
            with tqdm(total=len(rows), desc=f"Face crops [{split_name}]") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    all_output_rows.extend(result)
                    pbar.update(len(result))

        # Write manifest
        output_path = manifests_dir / f"{split_name}_with_faces.csv"
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['image_path', 'label', 'num_faces', 'face_crop_paths'])
            writer.writeheader()
            writer.writerows(all_output_rows)

        n_with_faces = sum(1 for r in all_output_rows if int(r['num_faces']) > 0)
        print(f"  {split_name}: {n_with_faces}/{len(all_output_rows)} images have faces → {output_path}")


if __name__ == '__main__':
    main()

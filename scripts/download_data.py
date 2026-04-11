"""
Download GRAVEX-200K dataset from Kaggle and generate CSV manifests.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --kaggle-cli   # large datasets: avoids kagglehub resume/MD5 issues
    python scripts/download_data.py --skip-download  # only rebuild manifests from existing gravex_200k/

Requires: pip install kagglehub (and pip install kaggle for --kaggle-cli)
Credentials: ~/.kaggle/kaggle.json (from https://www.kaggle.com/settings)
"""
import argparse
import csv
import random
import shutil
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import kagglehub
from kagglehub.exceptions import DataCorruptionError
from configs.base_config import ProjectConfig, DataConfig


def _download_dataset_kaggle_cli(dataset_slug: str, dest_dir: Path) -> None:
    kaggle_bin = shutil.which("kaggle")
    cmd = (
        [kaggle_bin, "datasets", "download", "-d", dataset_slug, "-p", str(dest_dir), "--unzip", "-o"]
        if kaggle_bin
        else [sys.executable, "-m", "kaggle", "datasets", "download", "-d", dataset_slug, "-p", str(dest_dir), "--unzip", "-o"]
    )
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        raise RuntimeError(f"kaggle CLI exit {proc.returncode}: {tail}")


def _gravex_image_roots(raw_dir: Path) -> list[Path]:
    """Unzipped Kaggle bundle often nests images under a single folder."""
    roots: list[Path] = []
    for name in ("my_real_vs_ai_dataset", "dataset", "data"):
        p = raw_dir / name
        if p.is_dir():
            roots.append(p)
    roots.append(raw_dir)
    return roots


def _resolve_gravex_image_path(raw_dir: Path, rel: str) -> Path:
    rel = rel.strip().strip('"').replace("\\", "/")
    p = Path(rel)
    if p.is_absolute():
        return p.resolve()
    for root in _gravex_image_roots(raw_dir):
        cand = (root / rel).resolve()
        if cand.is_file():
            return cand
    return (_gravex_image_roots(raw_dir)[0] / rel).resolve()


def _parse_label_cell(val: str, col_lower: str) -> int:
    """Return 0=real, 1=AI to match training BCE."""
    s = str(val).strip().lower()
    if s in ("real", "authentic", "genuine", "0", "neg", "negative"):
        return 0
    if s in ("ai", "fake", "synthetic", "generated", "1", "pos", "positive"):
        return 1
    n = int(float(val))
    if col_lower in ("is_real", "real_label", "is_authentic"):
        return 1 - n
    return n


def _row_path_and_label_keys(fieldnames: list[str]) -> tuple[str | None, str | None]:
    path_key = None
    label_key = None
    lower_map = {k: k.lower() for k in fieldnames}
    inv = {v: k for k, v in lower_map.items()}
    for cand in (
        "image_path",
        "filepath",
        "path",
        "filename",
        "image",
        "file_name",
        "img_path",
        "file",
    ):
        if cand in inv:
            path_key = inv[cand]
            break
    if path_key is None and fieldnames:
        path_key = fieldnames[0]
    for cand in ("label", "labels", "target", "class", "y", "is_ai", "is_real", "ai", "fake"):
        if cand in inv:
            label_key = inv[cand]
            break
    return path_key, label_key


def load_samples_from_official_label_csvs(raw_dir: Path) -> dict[str, list[tuple[str, int]]] | None:
    """
    Kaggle bundle: train_labels.csv, val_labels.csv, test_labels.csv next to image tree.
    If any split file is missing, return None (caller falls back to folder scan).
    """
    samples: dict[str, list[tuple[str, int]]] = {}
    for split in ("train", "val", "test"):
        fp = raw_dir / f"{split}_labels.csv"
        if not fp.is_file():
            return None
        rows: list[tuple[str, int]] = []
        with open(fp, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return None
            pk, lk = _row_path_and_label_keys(list(reader.fieldnames))
            if pk is None or lk is None:
                print(f"  Could not detect path/label columns in {fp.name}: {reader.fieldnames}", file=sys.stderr)
                return None
            lk_low = lk.lower()
            for row in reader:
                if not row.get(pk):
                    continue
                abs_path = _resolve_gravex_image_path(raw_dir, row[pk])
                try:
                    lab = _parse_label_cell(row[lk], lk_low)
                except (TypeError, ValueError):
                    continue
                rows.append((str(abs_path), lab))
        if not rows:
            return None
        samples[split] = rows
    # Log detected schema from train file (same columns expected on val/test)
    _tr = raw_dir / "train_labels.csv"
    with open(_tr, newline="", encoding="utf-8", errors="replace") as f:
        r = csv.DictReader(f)
        if r.fieldnames:
            pk, lk = _row_path_and_label_keys(list(r.fieldnames))
            print(f"  Label CSV columns: path={pk!r}, label={lk!r}")
    return samples


def _delete_kagglehub_archives(dataset_slug: str) -> int:
    """Remove *.archive under kagglehub cache so the next hub download cannot resume from a bad partial file."""
    parts = dataset_slug.split("/", 1)
    if len(parts) != 2:
        return 0
    owner, name = parts
    base = Path.home() / ".cache" / "kagglehub" / "datasets" / owner / name
    if not base.is_dir():
        return 0
    n = 0
    for p in base.glob("*.archive"):
        try:
            p.unlink()
            print(f"Removed partial hub archive: {p}", file=sys.stderr)
            n += 1
        except OSError:
            pass
    return n


def main():
    parser = argparse.ArgumentParser(description="Download GRAVEX-200K and write split CSV manifests.")
    parser.add_argument(
        "--kaggle-cli",
        action="store_true",
        help="Use the official Kaggle CLI instead of kagglehub (recommended for ~2GB+ bundles if hub fails MD5 after resume).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Do not download; only scan data_raw/gravex_200k and regenerate manifests.",
    )
    parser.add_argument(
        "--fresh-hub",
        action="store_true",
        help="Before kagglehub download, delete any *.archive for this dataset in ~/.cache/kagglehub (avoids broken resume).",
    )
    parser.add_argument(
        "--ignore-official-csv",
        action="store_true",
        help="Do not use train_labels.csv/val_labels.csv/test_labels.csv even if present (old folder-scan / random-split behavior).",
    )
    args = parser.parse_args()

    cfg = ProjectConfig()
    data_cfg = DataConfig()
    raw_dir = cfg.data_raw / "gravex_200k"
    raw_dir.mkdir(parents=True, exist_ok=True)

    raw_has_data = any(raw_dir.iterdir())

    if args.skip_download:
        if not raw_has_data:
            print("data_raw/gravex_200k is empty; nothing to scan. Remove --skip-download to fetch data.", file=sys.stderr)
            sys.exit(1)
        print("Skipping download (--skip-download); rebuilding manifests from existing files.")
    elif not raw_has_data:
        if args.kaggle_cli:
            print("Downloading GRAVEX-200K via Kaggle CLI (zip + unzip into data_raw/gravex_200k)...")
            _download_dataset_kaggle_cli(data_cfg.dataset_name, raw_dir)
            print(f"Dataset downloaded and unzipped into {raw_dir}")
        else:
            if args.fresh_hub:
                _delete_kagglehub_archives(data_cfg.dataset_name)
            print("Downloading GRAVEX-200K via kagglehub...")
            cache_path: str | None = None
            try:
                cache_path = kagglehub.dataset_download(data_cfg.dataset_name)
            except Exception as e:
                if isinstance(e, DataCorruptionError):
                    print(
                        "\nkagglehub: MD5 checksum failed (common after a bad resume). "
                        "Corrupt data is discarded — it cannot be reused.\n"
                        "Retrying with force_download=True; if it fails again use:\n"
                        "  python scripts/download_data.py --kaggle-cli\n",
                        file=sys.stderr,
                    )
                else:
                    print("kagglehub download failed; retrying with force_download=True...", file=sys.stderr)
                try:
                    cache_path = kagglehub.dataset_download(data_cfg.dataset_name, force_download=True)
                except Exception as e2:
                    print("kagglehub failed twice; falling back to `kaggle datasets download`...", file=sys.stderr)
                    try:
                        _download_dataset_kaggle_cli(data_cfg.dataset_name, raw_dir)
                        cache_path = None
                    except Exception as e3:
                        raise e2 from e3

            if cache_path is not None:
                print(f"Downloaded to cache: {cache_path}")
                cache_dir = Path(cache_path)
                for item in cache_dir.iterdir():
                    dest = raw_dir / item.name
                    if not dest.exists():
                        if item.is_dir():
                            shutil.copytree(str(item), str(dest))
                        else:
                            shutil.copy2(str(item), str(dest))
                print(f"Dataset copied to {raw_dir}")
            else:
                print(f"Dataset downloaded and unzipped into {raw_dir} (Kaggle CLI)")
    else:
        print(f"Data already exists at {raw_dir}")

    # Discover structure and generate manifests
    splits_dir = cfg.data_raw / "manifests"
    splits_dir.mkdir(parents=True, exist_ok=True)

    samples: dict[str, list[tuple[str, int]]] = defaultdict(list)

    if not args.ignore_official_csv:
        official = load_samples_from_official_label_csvs(raw_dir)
        if official is not None:
            samples = official
            print(
                "Using official train_labels.csv / val_labels.csv / test_labels.csv "
                f"(train={len(samples['train'])}, val={len(samples['val'])}, test={len(samples['test'])})."
            )
        else:
            print("No complete official *labels.csv trio; scanning folders (or random split)...")
    else:
        print("Ignoring official *labels.csv (--ignore-official-csv); scanning folders...")

    if not samples:
        print("Scanning dataset structure...")

    # Try pre-split structure first: train/val/test subdirs
    if not samples:
        for split_name in ["train", "val", "test"]:
            split_path = raw_dir / split_name
            if split_path.exists() and split_path.is_dir():
                for class_dir in sorted(split_path.iterdir()):
                    if not class_dir.is_dir():
                        continue
                    label = 0 if class_dir.name.lower() in ("real", "0") else 1
                    for img_path in sorted(class_dir.rglob("*")):
                        if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                            samples[split_name].append((str(img_path), label))

    # If no pre-split structure, scan for class folders at any depth
    if not samples:
        all_samples = []
        # Find leaf directories that look like class folders (contain images directly)
        class_dirs = set()
        for img_path in raw_dir.rglob("*"):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                class_dirs.add(img_path.parent)
        for class_dir in sorted(class_dirs):
            dir_name = class_dir.name.lower()
            label = 0 if dir_name in ("real", "0") else 1
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    all_samples.append((str(img_path), label))

        if all_samples:
            random.seed(data_cfg.seed)
            random.shuffle(all_samples)
            n = len(all_samples)
            n_train = int(n * 0.8)
            n_val = int(n * 0.1)
            samples["train"] = all_samples[:n_train]
            samples["val"] = all_samples[n_train : n_train + n_val]
            samples["test"] = all_samples[n_train + n_val :]

    # Write manifests
    for split_name, split_samples in samples.items():
        csv_path = splits_dir / f"{split_name}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "label"])
            for path, label in split_samples:
                writer.writerow([path, label])
        print(f"  {split_name}: {len(split_samples)} samples -> {csv_path}")

    total = sum(len(v) for v in samples.values())
    print(f"Total: {total} samples across {len(samples)} splits")


if __name__ == "__main__":
    main()

"""
Shared utilities for pipeline notebooks.
Handles dataset overview, preprocessing demos, augmentation study, and metrics visualization.
"""
import sys, os, random, io, csv, json, time
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image, ImageFilter, ImageDraw
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, roc_auc_score,
)

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_RAW = PROJECT_ROOT / "data_raw"
WEIGHTS  = PROJECT_ROOT / "weights"
RESULTS  = PROJECT_ROOT / "results"

# ── Dataset ──────────────────────────────────────────────────────────────────

def load_manifest(split="test"):
    """Load manifest CSV → list of dicts with image_path, label, face_crop_paths."""
    csv_path = DATA_RAW / "manifests" / f"{split}_with_faces.csv"
    samples = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            samples.append({
                "image_path": row["image_path"],
                "label": int(row["label"]),
                "num_faces": int(row["num_faces"]),
                "face_crops": json.loads(row["face_crop_paths"]) if row["face_crop_paths"] else [],
            })
    return samples


def show_dataset_overview(n_samples=8):
    """Show dataset statistics and sample images."""
    samples = load_manifest("test")
    labels = [s["label"] for s in samples]
    n_real = labels.count(0)
    n_ai   = labels.count(1)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f"GRAVEX-200K Dataset — Test Split ({len(samples):,} images)", fontsize=14, fontweight="bold")

    real_samples = [s for s in samples if s["label"] == 0]
    ai_samples   = [s for s in samples if s["label"] == 1]
    show = random.sample(real_samples, 4) + random.sample(ai_samples, 4)

    for ax, s in zip(axes.flat, show):
        img = Image.open(s["image_path"]).convert("RGB")
        ax.imshow(img)
        label = "Real" if s["label"] == 0 else "AI Generated"
        color = "green" if s["label"] == 0 else "red"
        ax.set_title(f"{label} | {s['num_faces']} face(s)", color=color, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    print(f"  Total test images:  {len(samples):,}")
    print(f"  Real images:        {n_real:,} ({n_real/len(samples):.1%})")
    print(f"  AI images:          {n_ai:,} ({n_ai/len(samples):.1%})")
    print(f"  Images with faces:  {sum(1 for s in samples if s['num_faces'] > 0):,}")
    return samples


# ── Preprocessing ────────────────────────────────────────────────────────────

def show_preprocessing_steps(image_path, target_size=224):
    """Demonstrate 3 preprocessing steps: resize, color space, noise reduction."""
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(3, 4, hspace=0.4, wspace=0.3)

    # ── Step 1: Resize with different interpolations ──
    fig.text(0.02, 0.95, "Step 1: Resize with Interpolation", fontsize=13, fontweight="bold")
    interps = [
        ("Original", img),
        ("Bilinear", img.resize((target_size, target_size), Image.BILINEAR)),
        ("Bicubic", img.resize((target_size, target_size), Image.BICUBIC)),
        ("Lanczos", img.resize((target_size, target_size), Image.LANCZOS)),
    ]
    for i, (name, im) in enumerate(interps):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(im)
        w, h = im.size
        ax.set_title(f"{name} ({w}×{h})", fontsize=10)
        ax.axis("off")

    # ── Step 2: Color Space ──
    fig.text(0.02, 0.63, "Step 2: Color Space Analysis", fontsize=13, fontweight="bold")
    resized = img.resize((target_size, target_size), Image.LANCZOS)
    resized_np = np.array(resized)
    channels = [("RGB (Full)", resized_np), ("Red Channel", resized_np[:,:,0]),
                ("Green Channel", resized_np[:,:,1]), ("Blue Channel", resized_np[:,:,2])]
    cmaps = [None, "Reds", "Greens", "Blues"]
    for i, ((name, ch), cmap) in enumerate(zip(channels, cmaps)):
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(ch, cmap=cmap)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    # ── Step 3: Noise Reduction ──
    fig.text(0.02, 0.31, "Step 3: Noise Reduction (Gaussian vs Median vs Bilateral)", fontsize=13, fontweight="bold")
    blurs = [
        ("No Filter", resized_np),
        ("Gaussian σ=1.5", cv2.GaussianBlur(resized_np, (5, 5), 1.5)),
        ("Median k=5", cv2.medianBlur(resized_np, 5)),
        ("Bilateral d=9", cv2.bilateralFilter(resized_np, 9, 75, 75)),
    ]
    for i, (name, b) in enumerate(blurs):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(b)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    plt.suptitle(f"Preprocessing Pipeline — {Path(image_path).name}", fontsize=14, y=1.01)
    plt.show()


# ── Augmentation Study ───────────────────────────────────────────────────────

def _apply_jpeg(img, quality=70):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def show_augmentation_study(image_path, size=224):
    """Show 8 augmentations with before/after and explanations."""
    img = Image.open(image_path).convert("RGB").resize((size, size), Image.LANCZOS)
    from torchvision import transforms as T

    augmentations = [
        ("1. Rotation (±15°)",
         T.RandomRotation(15)(img),
         "Simulates camera tilt. Keeps ≤15° to preserve face structure."),
        ("2. Horizontal Flip",
         T.RandomHorizontalFlip(p=1.0)(img),
         "Doubles effective dataset. Vertical flip not used — breaks gravity cues."),
        ("3. Random Crop",
         T.RandomResizedCrop(size, scale=(0.7, 1.0))(img),
         "Simulates different framing/zoom. Scale 0.7–1.0 for variety."),
        ("4. Translation (Affine)",
         T.RandomAffine(degrees=0, translate=(0.1, 0.1))(img),
         "Shifts image ±10%. Forces model to be position-invariant."),
        ("5. Blur (Gaussian)",
         img.filter(ImageFilter.GaussianBlur(radius=2)),
         "Simulates defocus/camera shake. Model must not rely on sharpness."),
        ("6. Sharpening",
         img.filter(ImageFilter.UnsharpMask(radius=2, percent=150)),
         "Enhances edges. Over-sharpening creates halos that mimic AI artifacts."),
        ("7. Color Jitter",
         T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.08)(img),
         "Varies lighting/white balance. Prevents learning brightness as proxy."),
        ("8. JPEG Compression (q=70)",
         _apply_jpeg(img, quality=70),
         "Simulates social media re-encoding. Prevents learning compression artifacts."),
    ]

    fig, axes = plt.subplots(4, 4, figsize=(18, 18))
    for i, (name, aug_img, desc) in enumerate(augmentations):
        row = i // 2
        col = (i % 2) * 2
        # Original
        axes[row, col].imshow(img)
        axes[row, col].set_title("Before", fontsize=9)
        axes[row, col].axis("off")
        # Augmented
        axes[row, col+1].imshow(aug_img)
        axes[row, col+1].set_title(name, fontsize=9, fontweight="bold")
        axes[row, col+1].axis("off")

    plt.suptitle("Augmentation Study — Before vs After", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # Print summary table
    print("\n  Augmentation Summary")
    print("  " + "─" * 75)
    print(f"  {'#':<4} {'Technique':<25} {'Category':<12} {'Probability':<12}")
    print("  " + "─" * 75)
    table = [
        ("1", "Rotation ±15°", "Geometric", "p=1.0"),
        ("2", "Horizontal Flip", "Geometric", "p=0.5"),
        ("3", "Random Crop (0.7-1.0)", "Geometric", "p=1.0"),
        ("4", "Translation ±10%", "Geometric", "p=1.0"),
        ("5", "Blur (G/M/B)", "Noise", "p=0.35"),
        ("6", "Sharpening", "Noise", "p=0.15"),
        ("7", "Color Jitter", "Intensity", "p=1.0"),
        ("8", "JPEG Compression", "Compression", "p=0.5"),
    ]
    for num, name, cat, prob in table:
        print(f"  {num:<4} {name:<25} {cat:<12} {prob:<12}")
    print()

    for name, _, desc in augmentations:
        print(f"  {name}: {desc}")


# ── Metrics & Evaluation ─────────────────────────────────────────────────────

def evaluate_pipeline(pipeline, samples, max_images=2000):
    """Run pipeline on test samples, collect predictions."""
    y_true, y_scores = [], []
    subset = random.sample(samples, min(max_images, len(samples)))

    for i, s in enumerate(subset):
        if (i + 1) % 200 == 0:
            print(f"  Processed {i+1}/{len(subset)}...")
        try:
            img = Image.open(s["image_path"]).convert("RGB")
            result = pipeline.classify(img, threshold=0.5)
            if result.error:
                continue
            y_true.append(s["label"])
            y_scores.append(result.fused_score)
        except Exception:
            continue

    return np.array(y_true), np.array(y_scores)


def show_metrics(y_true, y_scores, model_name="Model", threshold=0.5):
    """Compute and display all metrics with plots."""
    y_pred = (y_scores > threshold).astype(int)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_scores) if len(set(y_true)) > 1 else 0

    print(f"\n  {model_name} — Evaluation Metrics (threshold={threshold})")
    print("  " + "─" * 50)
    print(f"  Accuracy:   {acc:.4f}  ({acc:.1%})")
    print(f"  Precision:  {prec:.4f}")
    print(f"  Recall:     {rec:.4f}")
    print(f"  F1 Score:   {f1:.4f}")
    print(f"  AUC-ROC:    {auc:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=["Real", "AI Generated"]))

    # ── Confusion Matrix + ROC Curve side by side ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1,
                xticklabels=["Real", "AI"], yticklabels=["Real", "AI"])
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("True")
    ax1.set_title(f"{model_name} — Confusion Matrix")

    # ROC Curve
    if len(set(y_true)) > 1:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        ax2.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.4f}")
        ax2.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title(f"{model_name} — ROC Curve")
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc_roc": auc}


def show_predictions(pipeline, samples, n=8):
    """Show model predictions on sample images."""
    subset = random.sample(samples, min(n, len(samples)))
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))

    for ax, s in zip(axes.flat, subset):
        img = Image.open(s["image_path"]).convert("RGB")
        result = pipeline.classify(img, threshold=0.5)

        ax.imshow(img)
        true_label = "Real" if s["label"] == 0 else "AI"
        pred_label = result.label if not result.error else "Error"
        conf = result.fused_score if not result.error else 0

        correct = (s["label"] == 1 and conf > 0.5) or (s["label"] == 0 and conf <= 0.5)
        color = "green" if correct else "red"

        ax.set_title(f"True: {true_label} | Pred: {pred_label}\nConf: {conf:.2%}",
                      fontsize=9, color=color)
        ax.axis("off")

    plt.suptitle(f"{pipeline.name} — Sample Predictions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()


def show_inference_speed(pipeline, samples, n_runs=50):
    """Benchmark inference speed."""
    subset = random.sample(samples, min(n_runs, len(samples)))
    times = []
    for s in subset:
        img = Image.open(s["image_path"]).convert("RGB")
        t0 = time.time()
        pipeline.classify(img, threshold=0.5)
        times.append(time.time() - t0)

    avg = np.mean(times)
    std = np.std(times)
    print(f"  {pipeline.name} — Inference Speed")
    print(f"  Average: {avg:.3f}s ± {std:.3f}s per image")
    print(f"  Throughput: {1/avg:.1f} images/sec")
    return avg


def compare_models(results_dict):
    """Side-by-side comparison bar chart for multiple models."""
    models = list(results_dict.keys())
    metrics = ["accuracy", "precision", "recall", "f1", "auc_roc"]
    labels  = ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]

    x = np.arange(len(labels))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, model in enumerate(models):
        values = [results_dict[model].get(m, 0) for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=model)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", fontsize=8)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — All Pipelines", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

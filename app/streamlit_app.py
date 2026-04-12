"""
AI vs Real Image Detector — Streamlit App

Three pages:
  1. Single Image — upload → detect faces → classify with selected pipeline
  2. Model Comparison — same image through all pipelines side-by-side
  3. Metrics Dashboard — per-model evaluation metrics, confusion matrices, ROC curves

Run:
    streamlit run app/streamlit_app.py
"""
import sys
import time
import json
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="wide",
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ── Load pipelines ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading pipelines...")
def load_pipelines():
    from inference.registry import get_all_pipelines
    return get_all_pipelines()


def draw_faces(image: Image.Image, boxes: list) -> Image.Image:
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        draw.rectangle([x1, y1, x2, y2], outline="#FF4B4B", width=3)
        draw.text((x1 + 4, y1 + 4), f"Face {i+1}", fill="#FF4B4B")
    return img


def confidence_color(prob: float) -> str:
    if prob > 0.7:
        return "#FF4B4B"
    if prob > 0.5:
        return "#FFA500"
    return "#21C55D"


def render_result_card(result, threshold: float):
    if result.error:
        st.warning(f"**{result.pipeline_name}** — {result.error}")
        return

    # ── Full image prediction ──
    full_color = confidence_color(result.full_score)
    full_is_ai = result.full_score > threshold
    icon = "🔴" if full_is_ai else "🟢"
    st.markdown(
        f"<h3 style='color:{full_color}'>{icon} Full Image: {result.label}</h3>",
        unsafe_allow_html=True,
    )

    pct = int(result.full_score * 100)
    st.markdown(
        f"""
        <div style='background:#eee;border-radius:6px;height:24px;width:100%'>
          <div style='background:{full_color};width:{pct}%;height:24px;border-radius:6px;
                      display:flex;align-items:center;padding-left:8px'>
            <span style='color:white;font-size:12px;font-weight:bold'>{pct}% AI</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")

    st.metric("Full Image Score", f"{result.full_score:.3f}")

    st.caption(
        f"Faces detected: {result.num_faces}  |  "
        f"Time: {result.processing_time:.2f}s  |  "
        f"Model: {result.pipeline_name}"
    )

    # ── Per-face predictions ──
    if result.face_crops:
        with st.expander(f"Face predictions ({len(result.face_crops)})", expanded=True):
            cols = st.columns(min(len(result.face_crops), 4))
            for i, crop in enumerate(result.face_crops[:4]):
                with cols[i]:
                    st.image(crop, use_container_width=True)
                    score = result.face_scores[i] if i < len(result.face_scores) else 0
                    label = result.face_labels[i] if i < len(result.face_labels) else "N/A"
                    fc = confidence_color(score)
                    fi = "🔴" if score > threshold else "🟢"
                    st.markdown(f"{fi} **Face {i+1}: {label}**")
                    st.caption(f"Score: {score:.3f} ({int(score*100)}% AI)")
    else:
        st.info("No faces detected — full image score only.")


# ── Sidebar ───────────────────────────────────────────────────────────────────
pipelines = load_pipelines()

with st.sidebar:
    st.header("⚙️ Settings")

    page = st.radio(
        "Page",
        ["Single Image", "Model Comparison", "Metrics Dashboard"],
        index=0,
    )

    st.divider()

    threshold = st.slider(
        "Decision threshold",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Probability above this → AI Generated",
    )

    st.divider()
    st.subheader("Pipeline Status")
    for p in pipelines:
        status = "✅ Ready" if p.is_available() else "⏳ Not trained"
        st.markdown(f"**{p.name}** — {status}")
        st.caption(p.description)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1: Single Image
# ═════════════════════════════════════════════════════════════════════════════
if page == "Single Image":
    st.title("🔍 AI vs Real Image Detector")
    st.markdown("Upload an image → detect faces → classify as **AI Generated** or **Real**.")

    available_names = [p.name for p in pipelines if p.is_available()]
    if not available_names:
        st.warning("No trained pipelines available. Train a model first.")
        st.stop()

    selected_name = st.selectbox("Select pipeline", available_names)

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
    )

    if uploaded is None:
        st.info("Upload an image to get started.")
        st.stop()

    image = Image.open(uploaded).convert("RGB")
    pipeline = next(p for p in pipelines if p.name == selected_name)

    with st.spinner("Classifying..."):
        result = pipeline.classify(image, threshold)

    col_img, col_result = st.columns([1, 2])

    with col_img:
        st.subheader("Input Image")
        annotated = draw_faces(image, result.face_boxes) if result.face_boxes else image
        st.image(annotated, use_container_width=True)
        st.caption(f"{image.width}×{image.height}px  |  {result.num_faces} face(s)")

    with col_result:
        st.subheader("Result")
        render_result_card(result, threshold)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2: Model Comparison
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Model Comparison":
    st.title("📊 Multi-Model Comparison")
    st.markdown("Upload one image → compare results from all available pipelines side-by-side.")

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
    )

    if uploaded is None:
        st.info("Upload an image to compare models.")
        st.stop()

    image = Image.open(uploaded).convert("RGB")
    available = [p for p in pipelines if p.is_available()]

    if not available:
        st.warning("No trained pipelines available.")
        st.stop()

    face_boxes = []
    with st.spinner("Running all pipelines..."):
        results = {}
        for p in available:
            results[p.name] = p.classify(image, threshold)
            if not face_boxes and results[p.name].face_boxes:
                face_boxes = results[p.name].face_boxes

    col_img, col_spacer = st.columns([1, 3])
    with col_img:
        annotated = draw_faces(image, face_boxes) if face_boxes else image
        st.image(annotated, use_container_width=True)
        st.caption(f"{image.width}×{image.height}px  |  {len(face_boxes)} face(s)")

    st.divider()

    cols = st.columns(len(available))
    for col, p in zip(cols, available):
        with col:
            st.markdown(f"### {p.name}")
            render_result_card(results[p.name], threshold)

    st.divider()
    st.subheader("Score Comparison")
    chart_cols = st.columns(len(available))
    for col, p in zip(chart_cols, available):
        r = results[p.name]
        if not r.error:
            color = confidence_color(r.full_score)
            icon = "🔴" if r.full_score > threshold else "🟢"
            col.metric(
                p.name,
                f"{icon} {r.label}",
                f"{r.full_score:.1%} AI (full image)",
            )
            if r.face_scores:
                for i, fs in enumerate(r.face_scores):
                    fi = "🔴" if fs > threshold else "🟢"
                    col.caption(f"{fi} Face {i+1}: {fs:.1%} AI")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3: Metrics Dashboard
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Metrics Dashboard":
    st.title("📈 Metrics Dashboard")
    st.markdown("Pre-computed evaluation metrics from the test set (20,000 images).")

    eval_files = {
        "FerretNet":     RESULTS_DIR / "ferretnet" / "results.json",
        "CLIP ViT-B/32": RESULTS_DIR / "clip" / "eval_results.json",
        "VGG16":         RESULTS_DIR / "vgg16" / "eval_results.json",
    }

    loaded_results = {}
    for name, path in eval_files.items():
        if path.exists():
            try:
                with open(path) as f:
                    data = json.loads(f.read().strip())
                    loaded_results[name] = data
            except (json.JSONDecodeError, Exception):
                pass

    if not loaded_results:
        st.info("No evaluation results found. Run evaluation scripts first.")
        st.stop()

    # ── Summary table ──
    st.subheader("Summary")
    summary_data = []
    for name, data in loaded_results.items():
        d = data.get("fused", data)
        summary_data.append({
            "Model": name,
            "Accuracy": f"{d.get('accuracy', 0):.1%}",
            "Precision": f"{d.get('precision', 0):.3f}",
            "Recall": f"{d.get('recall', 0):.3f}",
            "F1": f"{d.get('f1', 0):.3f}",
            "AUC-ROC": f"{d.get('auc_roc', 0):.4f}",
        })
    st.table(summary_data)

    # ── Per-model details ──
    st.divider()
    selected_model = st.selectbox("Select model for details", list(loaded_results.keys()))
    data = loaded_results[selected_model]
    metrics = data.get("fused", data)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")
    col2.metric("Precision", f"{metrics.get('precision', 0):.3f}")
    col3.metric("Recall", f"{metrics.get('recall', 0):.3f}")
    col4.metric("F1 Score", f"{metrics.get('f1', 0):.3f}")

    cm = metrics.get("confusion_matrix")
    if cm is not None:
        st.subheader("Confusion Matrix")
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            np.array(cm), annot=True, fmt="d", cmap="Blues",
            xticklabels=["Real", "AI"], yticklabels=["Real", "AI"], ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{selected_model} — Confusion Matrix")
        st.pyplot(fig)
        plt.close(fig)

    report = metrics.get("classification_report")
    if report:
        st.subheader("Classification Report")
        st.code(report)

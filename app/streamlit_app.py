"""
AI vs Real Image Detector — Streamlit App

Upload an image → detect faces → classify with selected pipeline(s) → compare results.

Run:
    cd /workspace/AI_COMPUTER_VISION
    streamlit run app/streamlit_app.py
"""
import sys
import time
from pathlib import Path
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🔍",
    layout="wide",
)

# ── Load pipelines (cached so models load only once) ──────────────────────────
@st.cache_resource(show_spinner="Loading pipelines...")
def load_pipelines():
    from inference.registry import get_all_pipelines
    return get_all_pipelines()


def draw_faces(image: Image.Image, boxes: list) -> Image.Image:
    """Draw bounding boxes on image."""
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        draw.rectangle([x1, y1, x2, y2], outline="#FF4B4B", width=3)
        draw.text((x1 + 4, y1 + 4), f"Face {i+1}", fill="#FF4B4B")
    return img


def confidence_color(prob: float) -> str:
    """Red for AI, green for real."""
    if prob > 0.7:
        return "#FF4B4B"
    if prob > 0.5:
        return "#FFA500"
    return "#21C55D"


def render_result_card(result, threshold: float):
    """Render one pipeline result as a styled card."""
    if result.error:
        st.warning(f"**{result.pipeline_name}** — {result.error}")
        return

    prob = result.fused_score
    color = confidence_color(prob)
    is_ai = prob > threshold

    # Header
    icon = "🔴" if is_ai else "🟢"
    st.markdown(
        f"<h3 style='color:{color}'>{icon} {result.label}</h3>",
        unsafe_allow_html=True,
    )

    # Confidence bar
    pct = int(prob * 100)
    st.markdown(
        f"""
        <div style='background:#eee;border-radius:6px;height:20px;width:100%'>
          <div style='background:{color};width:{pct}%;height:20px;border-radius:6px;
                      display:flex;align-items:center;padding-left:6px'>
            <span style='color:white;font-size:12px;font-weight:bold'>{pct}% AI</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")

    # Score breakdown
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        face_val = f"{result.face_score:.3f}" if result.face_score >= 0 else "No faces"
        st.metric("Face Score", face_val)
    with col_b:
        st.metric("Full Image", f"{result.full_score:.3f}")
    with col_c:
        st.metric("Fused", f"{result.fused_score:.3f}")

    st.caption(
        f"Faces detected: {result.num_faces}  |  "
        f"Time: {result.processing_time:.2f}s  |  "
        f"Model: {result.pipeline_name}"
    )

    # Face crops
    if result.face_crops:
        with st.expander(f"Face crops ({len(result.face_crops)})", expanded=False):
            cols = st.columns(min(len(result.face_crops), 4))
            for i, crop in enumerate(result.face_crops[:4]):
                cols[i].image(crop, caption=f"Face {i+1}", use_container_width=True)


# ── Main UI ────────────────────────────────────────────────────────────────────
st.title("🔍 AI vs Real Image Detector")
st.markdown("Upload an image to detect faces and classify as **AI Generated** or **Real**.")

pipelines = load_pipelines()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Pipelines")
    selected_names = []
    for p in pipelines:
        available = p.is_available()
        label = p.name if available else f"{p.name} *(not trained)*"
        default = available
        checked = st.checkbox(label, value=default, key=f"pipeline_{p.name}")
        if checked:
            selected_names.append(p.name)

    if not selected_names:
        st.warning("Select at least one pipeline.")

    st.divider()
    threshold = st.slider(
        "Decision threshold",
        min_value=0.1, max_value=0.9, value=0.5, step=0.05,
        help="Probability above this → AI Generated",
    )

    st.divider()
    st.subheader("Pipeline info")
    for p in pipelines:
        status = "✅ Ready" if p.is_available() else "⏳ Not trained"
        st.markdown(f"**{p.name}** — {status}")
        st.caption(p.description)

# ── Image upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    label_visibility="collapsed",
)

if uploaded is None:
    st.info("Upload an image to get started.")
    st.stop()

image = Image.open(uploaded).convert("RGB")

# ── Run selected pipelines ─────────────────────────────────────────────────────
selected_pipelines = [p for p in pipelines if p.name in selected_names]

if not selected_pipelines:
    st.warning("No pipelines selected. Choose at least one in the sidebar.")
    st.stop()

with st.spinner("Detecting faces and classifying..."):
    results = {p.name: p.classify(image, threshold) for p in selected_pipelines}

# ── Display: image + face boxes ────────────────────────────────────────────────
# Use boxes from first result that has them
face_boxes = []
for r in results.values():
    if r.face_boxes:
        face_boxes = r.face_boxes
        break

col_img, col_results = st.columns([1, 2])

with col_img:
    st.subheader("Input Image")
    annotated = draw_faces(image, face_boxes) if face_boxes else image
    st.image(annotated, use_container_width=True)
    st.caption(
        f"{image.width}×{image.height}px  |  "
        f"{len(face_boxes)} face(s) detected"
    )

# ── Results ────────────────────────────────────────────────────────────────────
with col_results:
    st.subheader("Results")

    if len(selected_pipelines) == 1:
        render_result_card(list(results.values())[0], threshold)
    else:
        tabs = st.tabs([p.name for p in selected_pipelines])
        for tab, p in zip(tabs, selected_pipelines):
            with tab:
                render_result_card(results[p.name], threshold)

        # Comparison summary
        st.divider()
        st.subheader("Pipeline Comparison")
        cols = st.columns(len(selected_pipelines))
        for col, p in zip(cols, selected_pipelines):
            r = results[p.name]
            if not r.error:
                color = confidence_color(r.fused_score)
                icon = "🔴" if r.fused_score > threshold else "🟢"
                col.metric(
                    p.name,
                    f"{icon} {r.label}",
                    f"{r.fused_score:.1%} AI confidence",
                )

# streamlit_app.py
# DeepFake Detective — real FerretNet + CLIP pipelines from this repo, futuristic UI + 3D assistant.
#
# Run from repo root:
#   streamlit run app/streamlit_app.py
#
# Optional: export PROJECT_ROOT=/path/to/AI_COMPUTER_VISION

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw
from streamlit.components.v1 import html as st_html

_APP_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _APP_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Default pipelines to have checked when [READY] (sidebar widgets reset when cache key / bump changes)
_DEFAULT_SIDEBAR_PIPELINES = frozenset({"FerretNet", "CLIP ViT-B/32"})


def _pipeline_cache_key() -> str:
    """Bust @st.cache_resource when project root or checkpoint set changes."""
    from configs.base_config import ProjectConfig

    cfg = ProjectConfig()
    root = str(cfg.project_root.resolve())
    fd = cfg.lightning_logs / "checkpoints"
    cd = cfg.project_root / "lightning_logs_clip" / "checkpoints"
    has_f = bool(fd.is_dir() and any(fd.glob("*.ckpt")))
    has_c = bool(cd.is_dir() and any(cd.glob("*.ckpt")))
    return f"{root}|ferret={has_f}|clip={has_c}"


def _sidebar_widget_id() -> str:
    """Short id so checkbox keys reset when ckpts appear or user clicks Reload pipelines."""
    bump = int(st.session_state.get("pipeline_ui_bump", 0))
    raw = f"{_pipeline_cache_key()}|bump={bump}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# ═══════════════════════════════════════════════════════════════════
#  PAGE & PIPELINES
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="DeepFake Detective",
    page_icon=str(_APP_DIR / "app_icon.svg"),
    layout="wide",
    initial_sidebar_state="expanded",
)

ENSEMBLE_META = {
    "Ensemble v3": {
        "trained": False,
        "desc": "Multi-model fusion with calibrated confidence (training in progress).",
        "model": "ensemble-v3-beta",
    },
}

MODEL_IDS = {
    "FerretNet": "ferretnet-dual-branch",
    "CLIP ViT-B/32": "clip-vitb32-dual-head",
    "Ensemble v3": "ensemble-v3-beta",
}


@st.cache_resource(show_spinner="Loading pipeline registry…")
def cached_pipelines(_cache_key: str):
    from inference.registry import get_all_pipelines

    return get_all_pipelines()


def pipeline_catalog():
    """Sidebar metadata: real pipelines + placeholder ensemble."""
    out = {}
    for p in cached_pipelines(_pipeline_cache_key()):
        out[p.name] = {
            "trained": p.is_available(),
            "desc": p.description,
            "model": MODEL_IDS.get(p.name, p.name),
            "instance": p,
        }
    for name, meta in ENSEMBLE_META.items():
        out[name] = {**meta, "instance": None}
    return out


# ═══════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════

for k, v in {
    "anim_trigger": 0,
    "results": None,
    "processed_image": None,
    "processed_file_key": None,
    "threshold": 0.5,
    "active_pipelines": ["FerretNet", "CLIP ViT-B/32"],
    "viz_boxes": None,
    "viz_scores": None,
    "_pipeline_ck_seen": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

_ck = _pipeline_cache_key()
if st.session_state.get("_pipeline_ck_seen") != _ck:
    st.session_state["_pipeline_ck_seen"] = _ck
    st.session_state.results = None
    st.session_state.processed_file_key = None
    st.session_state.viz_boxes = None
    st.session_state.viz_scores = None


# ═══════════════════════════════════════════════════════════════════
#  CSS (futuristic theme)
# ═══════════════════════════════════════════════════════════════════

FUTURISTIC_CSS = """
<style>
:root {
    --bg: #0a0e17;
    --bg2: #111827;
    --cyan: #00f0ff;
    --teal: #0d9488;
    --purple: #7c3aed;
    --red: #ef4444;
    --green: #22c55e;
    --amber: #f59e0b;
    --text1: #e2e8f0;
    --text2: #94a3b8;
    --card: rgba(17,24,39,0.85);
    --border: rgba(0,240,255,0.15);
}
.stApp { background: linear-gradient(160deg,#0a0e17 0%,#0f172a 50%,#0a0e17 100%) !important; }
.stMarkdown, .stText { color: var(--text1) !important; }
[data-testid="stSidebar"] {
    background: rgba(10,14,23,0.97) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }
.section-line {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--cyan), transparent);
    margin: 1.2rem 0;
    opacity: 0.45;
}
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(0,240,255,0.38) !important;
    border-radius: 16px !important;
    background: rgba(0,240,255,0.04) !important;
    padding: 1.75rem !important;
    transition: border-color 0.25s, background 0.25s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(0,240,255,0.65) !important;
    background: rgba(0,240,255,0.07) !important;
}
[data-testid="stFileUploader"] label {
    color: var(--cyan) !important;
    font-weight: 600 !important;
}
.stButton > button {
    background: linear-gradient(135deg, var(--cyan), var(--teal)) !important;
    color: #000 !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    transition: transform 0.12s, box-shadow 0.25s;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 0 22px rgba(0,240,255,0.35);
}
.stSlider [data-baseweb="thumb"] {
    background: var(--cyan) !important;
    box-shadow: 0 0 10px rgba(0,240,255,0.45);
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(0,240,255,0.05) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    color: var(--text2) !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,240,255,0.14) !important;
    color: var(--cyan) !important;
    box-shadow: 0 0 14px rgba(0,240,255,0.12);
}
[data-testid="stMetricValue"] { color: var(--cyan) !important; font-weight: 800 !important; }
.streamlit-expanderHeader {
    background: rgba(0,240,255,0.05) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
.pipe-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
    transition: border-color 0.25s, box-shadow 0.25s;
}
.pipe-card:hover {
    border-color: rgba(0,240,255,0.35);
    box-shadow: 0 0 18px rgba(0,240,255,0.08);
}
.result-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
}
.result-card.scanlines::after {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(0,240,255,0.02) 2px, rgba(0,240,255,0.02) 4px
    );
    pointer-events: none;
    border-radius: inherit;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--cyan), var(--teal), var(--purple));
    z-index: 1;
}
.conf-bar-track {
    width: 100%;
    height: 22px;
    background: rgba(255,255,255,0.06);
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
    position: relative;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 5px;
    transition: width 1.1s cubic-bezier(0.22,1,0.36,1);
    position: relative;
}
.conf-bar-fill::after {
    content: '';
    position: absolute;
    top: 0; right: 0; bottom: 0;
    width: 36px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.22));
    animation: bar-shimmer 2.2s ease-in-out infinite;
}
@keyframes bar-shimmer {
    0%, 100% { opacity: 0.25; }
    50% { opacity: 1; }
}
.cmp-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid var(--border);
}
.cmp-table th {
    background: rgba(0,240,255,0.08);
    color: var(--cyan);
    padding: 10px 14px;
    text-align: left;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.cmp-table td {
    padding: 10px 14px;
    border-top: 1px solid var(--border);
    color: var(--text1);
    font-size: 0.9rem;
}
.cmp-table tr:hover td { background: rgba(0,240,255,0.03); }
.vp-frame-label {
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #64748b;
    margin: 6px 0 10px 2px;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: rgba(0,240,255,0.22); border-radius: 3px; }
.title-grad {
    background: linear-gradient(135deg, var(--cyan), var(--teal));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 900;
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
</style>
"""
st.markdown(FUTURISTIC_CSS, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  3D ASSISTANT
# ═══════════════════════════════════════════════════════════════════

def render_3d_assistant(should_wave: bool, iframe_key: str) -> None:
    """3D robot: mouse tracking + wave on analyze. Unique HTML prefix forces iframe refresh when re-run."""
    tpl = (_APP_DIR / "robot_scene.html").read_text(encoding="utf-8")
    inner = tpl.replace("__SHOULD_WAVE__", "true" if should_wave else "false")
    html_str = f"<!--streamlit-run:{iframe_key}-->\n{inner}"
    st.markdown(
        '<p class="vp-frame-label">Interactive viewport — move the pointer; on each run the unit advances and waves</p>',
        unsafe_allow_html=True,
    )
    st_html(html_str, height=440, scrolling=False)


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════

def draw_face_boxes(
    image: Image.Image,
    boxes: list,
    scores: list[float] | None = None,
) -> Image.Image:
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    if scores is None:
        scores = [0.0] * len(boxes)
    for face_i, (bbox, conf) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=(0, 240, 255), width=2)
        bl = min(x2 - x1, y2 - y1) // 4
        for cx, cy, dx, dy in [
            (x1, y1, 1, 1),
            (x2, y1, -1, 1),
            (x1, y2, 1, -1),
            (x2, y2, -1, -1),
        ]:
            draw.line([cx, cy, cx + bl * dx, cy], fill=(0, 255, 136), width=3)
            draw.line([cx, cy, cx, cy + bl * dy], fill=(0, 255, 136), width=3)
        label = f"{conf:.0%}" if conf > 0 else f"Face {face_i + 1}"
        draw.text((x1 + 4, max(0, y1 - 14)), label, fill=(0, 240, 255))
    return img


def get_bar_color(pct: float) -> tuple[str, str]:
    if pct < 30:
        return "#22c55e", "#16a34a"
    if pct < 60:
        return "#f59e0b", "#d97706"
    return "#ef4444", "#dc2626"


def result_card_html(pipeline_name: str, result, threshold: float) -> str:
    if result.error:
        return f'<div class="result-card"><p style="color:#f59e0b">{result.error}</p></div>'

    fused = result.fused_score
    face_s = result.face_score
    full_s = result.full_score
    fused_pct = round(fused * 100, 1)
    face_pct = round(face_s * 100, 1) if face_s >= 0 else None
    full_pct = round(full_s * 100, 1)
    is_ai = fused >= threshold
    label = "AI GENERATED" if is_ai else "REAL"
    label_color = "#ef4444" if is_ai else "#22c55e"
    label_icon = (
        f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
        f'background:{label_color};box-shadow:0 0 16px {label_color}99;flex-shrink:0"></span>'
    )
    bar_c1, bar_c2 = get_bar_color(fused_pct)
    face_display = f"{face_pct}%" if face_pct is not None else "—"
    face_sub = "No face branch" if face_s < 0 else "Face score"
    time_ms = max(1, int(result.processing_time * 1000))
    model_id = MODEL_IDS.get(pipeline_name, pipeline_name)

    return f"""
    <div class="result-card scanlines">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
        <div style="display:flex;align-items:center;gap:12px">
          {label_icon}
          <span style="color:{label_color};font-weight:900;font-size:1.35rem;letter-spacing:0.04em">{label}</span>
        </div>
        <span style="color:var(--text2);font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em">{pipeline_name}</span>
      </div>
      <div style="margin-bottom:14px">
        <div style="display:flex;justify-content:space-between;margin-bottom:4px">
          <span style="color:var(--text2);font-size:0.78rem;text-transform:uppercase">P(AI) fused</span>
          <span style="color:{label_color};font-weight:800;font-size:0.95rem">{fused_pct}%</span>
        </div>
        <div class="conf-bar-track">
          <div class="conf-bar-fill" style="width:{min(100,fused_pct)}%;background:linear-gradient(90deg,{bar_c1},{bar_c2})"></div>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:12px">
        <div style="background:rgba(0,240,255,0.04);border-radius:8px;padding:10px;text-align:center;border:1px solid rgba(0,240,255,0.1)">
          <div style="color:var(--cyan);font-weight:800;font-size:1.15rem">{face_display}</div>
          <div style="color:var(--text2);font-size:0.7rem;text-transform:uppercase;margin-top:2px">{face_sub}</div>
        </div>
        <div style="background:rgba(0,240,255,0.04);border-radius:8px;padding:10px;text-align:center;border:1px solid rgba(0,240,255,0.1)">
          <div style="color:var(--cyan);font-weight:800;font-size:1.15rem">{full_pct}%</div>
          <div style="color:var(--text2);font-size:0.7rem;text-transform:uppercase;margin-top:2px">Full image</div>
        </div>
        <div style="background:rgba(0,240,255,0.04);border-radius:8px;padding:10px;text-align:center;border:1px solid rgba(0,240,255,0.1)">
          <div style="color:{label_color};font-weight:800;font-size:1.15rem">{fused_pct}%</div>
          <div style="color:var(--text2);font-size:0.7rem;text-transform:uppercase;margin-top:2px">Fused</div>
        </div>
      </div>
      <div style="display:flex;justify-content:space-between;color:var(--text2);font-size:0.72rem;text-transform:uppercase">
        <span>{result.num_faces} face(s)</span>
        <span>{time_ms} ms</span>
        <span>{model_id}</span>
      </div>
    </div>
    """


def comparison_table_html(results: dict, threshold: float) -> str:
    rows = ""
    for name, res in results.items():
        if res.error:
            continue
        fused_pct = round(res.fused_score * 100, 1)
        is_ai = res.fused_score >= threshold
        label = "AI Generated" if is_ai else "Real"
        dot = "#ef4444" if is_ai else "#22c55e"
        icon = (
            f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
            f'background:{dot};box-shadow:0 0 8px {dot}88;vertical-align:middle;margin-right:6px"></span>'
        )
        bar_c1, bar_c2 = get_bar_color(fused_pct)
        rows += f"""
        <tr>
          <td style="font-weight:600">{name}</td>
          <td>{icon} {label}</td>
          <td>
            <div style="display:flex;align-items:center;gap:8px">
              <div style="flex:1;height:8px;background:rgba(255,255,255,0.06);border-radius:4px;overflow:hidden">
                <div style="width:{min(100,fused_pct)}%;height:100%;background:linear-gradient(90deg,{bar_c1},{bar_c2})"></div>
              </div>
              <span style="font-weight:700;min-width:45px;text-align:right">{fused_pct}%</span>
            </div>
          </td>
        </tr>"""
    return f"""
    <div style="margin-top:1.2rem">
      <div style="color:var(--cyan);font-weight:700;font-size:0.85rem;text-transform:uppercase;margin-bottom:8px">Pipeline comparison</div>
      <table class="cmp-table">
        <thead><tr><th>Pipeline</th><th>Verdict</th><th>P(AI) fused</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    """


# ═══════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════

catalog = pipeline_catalog()

with st.sidebar:
    st.markdown(
        """
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:4px">
      <div style="width:36px;height:36px;border-radius:10px;background:linear-gradient(135deg,#00f0ff,#0d9488);display:flex;align-items:center;justify-content:center;font-weight:900;font-size:0.65rem;letter-spacing:0.06em;color:#0a0e17">SYS</div>
      <div>
        <div style="font-weight:900;font-size:1.1rem;color:#e2e8f0">Pipeline control</div>
        <div style="font-size:0.7rem;color:#94a3b8;text-transform:uppercase">Configuration</div>
      </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)

    st.markdown("#### Active pipelines")
    _wid = _sidebar_widget_id()
    active: list[str] = []
    for name, cfg in catalog.items():
        tag = "READY" if cfg["trained"] else "WAIT"
        enabled = cfg["trained"]
        # Fresh defaults when widget id changes (WAIT→READY or after Reload); Streamlit ignores
        # `value=` once a key exists, so keys must include _wid.
        default_on = enabled and name in _DEFAULT_SIDEBAR_PIPELINES
        checked = st.checkbox(
            f"[{tag}]  {name}",
            value=default_on,
            disabled=not enabled,
            key=f"chk_{name}_{_wid}",
        )
        if checked and enabled:
            active.append(name)
    st.session_state.active_pipelines = active

    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
    st.markdown("#### Decision threshold")
    threshold = st.slider(
        "P(AI) above this → AI-generated",
        min_value=0.1,
        max_value=0.9,
        value=float(st.session_state.threshold),
        step=0.05,
        help="Uses the same rule as training fusion: fused score vs threshold.",
    )
    st.session_state.threshold = threshold
    st.caption(f"Current: ≥ {threshold:.0%} fused → AI")

    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
    st.markdown("#### Pipeline details")
    for name, cfg in catalog.items():
        c = "#22c55e" if cfg["trained"] else "#f59e0b"
        st.markdown(
            f"""
        <div class="pipe-card">
          <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">
            <span style="width:8px;height:8px;border-radius:50%;background:{c}"></span>
            <span style="font-weight:700;font-size:0.9rem;color:#e2e8f0">{name}</span>
            <span style="margin-left:auto;font-size:0.65rem;color:{c};text-transform:uppercase">{"Ready" if cfg["trained"] else "Not trained"}</span>
          </div>
          <div style="font-size:0.75rem;color:#94a3b8;line-height:1.45">{cfg["desc"]}</div>
          <div style="font-size:0.65rem;color:#64748b;margin-top:6px;text-transform:uppercase">Model: {cfg["model"]}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
    if st.button("Reload pipelines", use_container_width=True, help="Use after new checkpoints or changing PROJECT_ROOT"):
        cached_pipelines.clear()
        st.session_state["_pipeline_ck_seen"] = None
        st.session_state["pipeline_ui_bump"] = int(st.session_state.get("pipeline_ui_bump", 0)) + 1
        st.rerun()
    if st.button("Clear analysis", use_container_width=True):
        st.session_state.results = None
        st.session_state.processed_image = None
        st.session_state.processed_file_key = None
        st.session_state.anim_trigger = 0
        st.session_state.viz_boxes = None
        st.session_state.viz_scores = None
        st.rerun()


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

st.markdown(
    """
<div style="display:flex;align-items:center;gap:14px;margin-bottom:2px">
  <div style="width:44px;height:44px;border-radius:12px;background:linear-gradient(135deg,#00f0ff,#0d9488);display:flex;align-items:center;justify-content:center;font-weight:900;font-size:0.85rem;letter-spacing:0.04em;color:#0a0e17">DFX</div>
  <div>
    <div class="title-grad" style="font-size:1.8rem;line-height:1.1">DeepFake Detective</div>
    <div style="font-size:0.78rem;color:#94a3b8;text-transform:uppercase;margin-top:2px">AI-powered authenticity (FerretNet + CLIP)</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)

render_3d_assistant(
    should_wave=st.session_state.anim_trigger > 0,
    iframe_key=f"df_robot_{st.session_state.anim_trigger}",
)

uploaded_file = st.file_uploader(
    "Drop an image for analysis",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    label_visibility="collapsed",
)

pipelines = cached_pipelines(_pipeline_cache_key())
name_to_pipeline = {p.name: p for p in pipelines}

if uploaded_file is not None:
    file_key = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.get("processed_file_key") != file_key:
        image = Image.open(uploaded_file).convert("RGB")
        selected = [
            name_to_pipeline[n]
            for n in st.session_state.active_pipelines
            if n in name_to_pipeline and name_to_pipeline[n].is_available()
        ]
        with st.spinner("Running RetinaFace + model inference…"):
            results = {}
            viz_boxes, viz_scores = None, None
            for p in selected:
                results[p.name] = p.classify(image, threshold)
                r = results[p.name]
                if not r.error and r.face_boxes:
                    viz_boxes = r.face_boxes
                    viz_scores = r.face_detection_scores
            st.session_state.results = results
            st.session_state.processed_image = image
            st.session_state.processed_file_key = file_key
            st.session_state.viz_boxes = viz_boxes
            st.session_state.viz_scores = viz_scores or []
            st.session_state.anim_trigger = st.session_state.anim_trigger + 1
        st.rerun()

if st.session_state.processed_image is not None and st.session_state.results is not None:
    image = st.session_state.processed_image
    results = st.session_state.results
    threshold = st.session_state.threshold
    boxes = st.session_state.viz_boxes or []
    scores = st.session_state.viz_scores or []

    st.markdown('<div class="section-line"></div>', unsafe_allow_html=True)
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.markdown(
            '<div style="color:#00f0ff;font-weight:700;font-size:0.82rem;text-transform:uppercase;margin-bottom:8px">Input scan</div>',
            unsafe_allow_html=True,
        )
        annotated = draw_face_boxes(image, boxes, scores if len(scores) == len(boxes) else None)
        st.image(annotated, use_container_width=True)
        st.caption(
            f"{image.width}×{image.height} px  |  {len(boxes)} face(s) detected (RetinaFace)"
        )
        with st.expander("Face crop thumbnails"):
            if boxes:
                crop_cols = st.columns(min(len(boxes), 4))
                for i, bbox in enumerate(boxes[:4]):
                    x1, y1, x2, y2 = bbox
                    crop = image.crop((x1, y1, x2, y2))
                    sc = scores[i] if i < len(scores) else 0.0
                    with crop_cols[i]:
                        st.image(crop, caption=f"Face {i + 1} ({sc:.0%})" if sc else f"Face {i + 1}", use_container_width=True)
            else:
                st.info("No faces detected — models used full-image branch only.")

    with right_col:
        st.markdown(
            '<div style="color:#00f0ff;font-weight:700;font-size:0.82rem;text-transform:uppercase;margin-bottom:8px">Analysis results</div>',
            unsafe_allow_html=True,
        )
        ok_results = {k: v for k, v in results.items() if not v.error}
        err_results = {k: v for k, v in results.items() if v.error}

        for name, res in err_results.items():
            st.warning(f"**{name}:** {res.error}")

        if not ok_results:
            if not results:
                st.warning(
                    "No pipeline ran. In the sidebar, enable **[READY]** FerretNet and/or CLIP, "
                    "then click **Reload pipelines** if you added checkpoints after starting the app, "
                    "and upload the image again (or **Clear analysis** first)."
                )
            else:
                st.warning(
                    "No successful runs — every selected pipeline failed (see warnings above). "
                    "Fix errors or try **Reload pipelines** after updating checkpoints."
                )
        elif len(ok_results) == 1:
            pipe_name = next(iter(ok_results))
            res = ok_results[pipe_name]
            st.markdown(result_card_html(pipe_name, res, threshold), unsafe_allow_html=True)
            with st.expander("Detection details"):
                fs = res.face_score
                st.markdown(
                    f"""
- **Face branch:** {fs:.1%} (max over crops)  {f"— {res.num_faces} face(s)" if fs >= 0 else "_(no faces)_"}
- **Full image:** {res.full_score:.1%}
- **Fused P(AI):** {res.fused_score:.1%}
- **Threshold:** {threshold:.0%}
- **Latency:** {res.processing_time:.2f}s
"""
                )
        else:
            tab_names = list(ok_results.keys())
            tabs = st.tabs(tab_names)
            for i, pipe_name in enumerate(tab_names):
                with tabs[i]:
                    res = ok_results[pipe_name]
                    st.markdown(result_card_html(pipe_name, res, threshold), unsafe_allow_html=True)
                    with st.expander("Detection details"):
                        fs = res.face_score
                        st.markdown(
                            f"""
- **Face branch:** {fs:.1%}  {f"({res.num_faces} faces)" if fs >= 0 else "_(no faces)_"}
- **Full image:** {res.full_score:.1%}
- **Fused:** {res.fused_score:.1%}
- **Threshold:** {threshold:.0%}
- **Latency:** {res.processing_time:.2f}s
"""
                        )
            st.markdown(comparison_table_html(ok_results, threshold), unsafe_allow_html=True)

        if len(ok_results) > 1:
            verdicts = [
                "AI" if r.fused_score >= threshold else "REAL" for r in ok_results.values()
            ]
            all_agree = len(set(verdicts)) == 1
            if all_agree:
                verdict = verdicts[0]
                color = "#ef4444" if verdict == "AI" else "#22c55e"
                rgb = "239,68,68" if verdict == "AI" else "34,197,94"
                mark = "!" if verdict == "AI" else "OK"
                mark_bg = "rgba(239,68,68,0.35)" if verdict == "AI" else "rgba(34,197,94,0.35)"
                st.markdown(
                    f"""
            <div style="background:rgba({rgb},0.08);border:1px solid {color}33;border-radius:12px;padding:14px 18px;margin-top:8px;display:flex;align-items:center;gap:12px">
              <span style="display:inline-flex;align-items:center;justify-content:center;min-width:28px;height:28px;border-radius:8px;background:{mark_bg};border:1px solid {color}55;color:{color};font-size:0.7rem;font-weight:900;letter-spacing:0.06em">{mark}</span>
              <span style="color:{color};font-weight:800">All pipelines agree: {verdict}</span>
            </div>
            """,
                    unsafe_allow_html=True,
                )
            else:
                pairs = ", ".join(f"{n}: {v}" for n, v in zip(ok_results.keys(), verdicts))
                st.markdown(
                    f"""
            <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.25);border-radius:12px;padding:14px 18px;margin-top:8px">
              <span style="color:#f59e0b;font-weight:800">Pipelines disagree</span>
              <div style="color:#94a3b8;font-size:0.85rem;margin-top:4px">{pairs}</div>
            </div>
            """,
                    unsafe_allow_html=True,
                )

else:
    st.markdown(
        """
    <div style="text-align:center;padding:2.5rem 1rem">
      <div style="width:72px;height:72px;margin:0 auto 14px;border-radius:14px;border:1px dashed rgba(0,240,255,0.35);background:rgba(0,240,255,0.04);display:flex;align-items:center;justify-content:center;font-size:0.65rem;font-weight:800;letter-spacing:0.2em;color:#64748b">INPUT</div>
      <div style="color:#94a3b8;font-size:0.95rem;font-weight:600">Upload an image to run the pipeline</div>
      <div style="color:#64748b;font-size:0.8rem;margin-top:4px">Requires checkpoints under <code>lightning_logs/</code> and RetinaFace weights in <code>weights/</code></div>
    </div>
    """,
        unsafe_allow_html=True,
    )

import io
import os
import tempfile
import streamlit as st
from PIL import Image

from hybrid_detector import HybridDeepfakeDetector as DeepfakeDetector
from grad_cam import VideoGradCAM

st.set_page_config(
    page_title="Hybrid Spatial-Temporal Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_resource
def load_models():
    detector = DeepfakeDetector(
        model_path="models/fine_tuned.pt",
        gru_model_path="models/video_gru.pt",
        clip_env_path="models/clip_temporal_env.pt",
        meta_fusion_path="models/meta_fusion.pkl",
        num_video_frames=16,
    )
    grad_cam = VideoGradCAM(detector)
    return detector, grad_cam


st.title("🛡️ Hybrid Spatial-Temporal Deepfake Detector")
st.caption(
    "Dual-Stream Architecture: Quality-Gated EfficientNet-B0 Face Specialist + "
    "CLIP ViT-B/16 & 4-Layer Temporal Transformer Environment Specialist + Grad-CAM (XAI)"
)

# Load models once before rendering sidebar status
with st.spinner("Loading EfficientNet Face Ensemble, CLIP-Temporal Specialist & Grad-CAM..."):
    detector, grad_cam = load_models()

# Sidebar Controls & Weight Status
with st.sidebar:
    st.header("⚙️ Settings")
    num_frames = st.slider(
        "Video Frames to Analyze", min_value=8, max_value=32, value=16, step=4
    )
    enable_gradcam = st.checkbox(
        "Enable Grad-CAM Heatmap (XAI)",
        value=True,
        help="Image aur Video dono par manipulated face areas ko highlight karega.",
    )
    max_cam_frames = st.slider(
        "Max Grad-CAM Video Frames",
        min_value=30,
        max_value=240,
        value=90,
        step=30,
        disabled=not enable_gradcam,
    )

    st.divider()
    st.markdown("**Loaded Weights Status:**")
    for w_path in [
        "models/clip_temporal_env.pt",
        "models/fine_tuned_v1.pt",
        "models/fine_tuned.pt",
        "models/best_model.pt",
        "models/video_gru.pt",
    ]:
        status = "✅" if os.path.exists(w_path) else "⚠️ Missing"
        st.text(f"{status} {os.path.basename(w_path)}")

    if os.path.exists("models/meta_fusion.pkl"):
        st.text("✅ meta_fusion.pkl (Learned Fusion)")
    else:
        st.text("ℹ️ Adaptive Rule Fusion Active")

    env_det = getattr(detector, "env_detector", None)
    if env_det and env_det.is_ready:
        st.caption(
            f"CLIP Env Calibration → T = `{env_det.temperature:.2f}` | "
            f"Threshold = `{env_det.threshold:.3f}`"
        )

uploaded_file = st.file_uploader(
    "Test karne ke liye Image ya Video upload karein:",
    type=["jpg", "jpeg", "png", "webp", "mp4", "avi", "mov", "mkv", "webm"],
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    is_video = file_ext in (".mp4", ".avi", ".mov", ".mkv", ".webm")

    col1, col2 = st.columns([1, 1.15], gap="large")

    with col1:
        st.subheader("📂 Uploaded Media")
        if is_video:
            st.video(file_bytes)
        else:
            st.image(Image.open(io.BytesIO(file_bytes)), use_container_width=True)

    with col2:
        st.subheader("🔍 Detection Analysis")
        run_btn = st.button(
            "🚀 Run Detection", type="primary", use_container_width=True
        )

        if run_btn:
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name

                with st.spinner("Running Dual-Stream Face + CLIP Environment Analysis..."):
                    if is_video:
                        result = detector.predict_video(tmp_path, num_frames=num_frames)
                    else:
                        result = detector.predict(tmp_path)

                label = result["Label"]
                fake_conf = result["Fake Confidence"] * 100
                real_conf = result["Real Confidence"] * 100

                if label == "FAKE":
                    st.error(f"### 🚨 Prediction: FAKE ({fake_conf:.2f}% Confidence)")
                else:
                    st.success(f"### ✅ Prediction: REAL ({real_conf:.2f}% Confidence)")

                fusion_mode = result.get("Fusion Mode")
                if fusion_mode:
                    st.caption(f"**Fusion Mode:** `{fusion_mode}`")

                m1, m2 = st.columns(2)
                m1.metric("Fake Confidence", f"{fake_conf:.2f}%")
                m2.metric("Real Confidence", f"{real_conf:.2f}%")

                # Video-specific dual-stream metrics & chart
                if is_video:
                    m3, m4, m5, m6 = st.columns(4)

                    face_score = result.get("Face Stream Score")
                    if face_score is not None:
                        sharp_pct = result.get("Sharp Face Ratio", 0.0) * 100
                        m3.metric(
                            "Face Stream",
                            f"{face_score * 100:.1f}%",
                            delta=f"Sharp: {sharp_pct:.0f}%",
                            delta_color="off",
                        )
                    else:
                        m3.metric("Face Stream", "No Face")

                    env_aligned = result.get("GRU Temporal Score", 0.0) * 100
                    env_raw = result.get("Env CLIP Score (Raw)")
                    if env_raw is not None:
                        m4.metric(
                            "Env CLIP Stream",
                            f"{env_aligned:.1f}%",
                            delta=f"Raw: {env_raw * 100:.1f}%",
                            delta_color="off",
                        )
                    else:
                        m4.metric("Temporal Stream", f"{env_aligned:.1f}%")

                    m5.metric("Scene Drift", f"{result['Temporal Flicker']:.4f}")
                    m6.metric(
                        "Faces Detected",
                        f"{result['Faces Detected']} / {result['Frames Analyzed']}",
                    )

                    frame_scores = result.get("Frame Fake Scores", [])
                    if frame_scores:
                        if result.get("Faces Detected", 0) > 0:
                            st.markdown("**📈 Face Frame-by-Frame Fake Confidence**")
                            st.line_chart({"Face Fake Confidence": frame_scores})
                        else:
                            st.markdown("**📈 CLIP Temporal Attention Weights (Across Frames)**")
                            st.line_chart({"Temporal Attention": frame_scores})

                # Grad-CAM Section (Images + Videos)
                if enable_gradcam:
                    st.divider()
                    st.markdown(
                        "**🔥 Grad-CAM Explainability (Red/Yellow = High Fake Attention)**"
                    )

                    if is_video and result.get("Faces Detected", 0) == 0:
                        st.info(
                            "No clear human face detected — verdict is driven 100% by the "
                            "CLIP Global Environment Specialist (face Grad-CAM skipped)."
                        )
                    elif is_video:
                        with st.spinner("Generating Grad-CAM Heatmap Video & Previews..."):
                            out_mp4 = os.path.join(
                                tempfile.gettempdir(), "gradcam_heatmap.mp4"
                            )
                            out_mp4, previews = grad_cam.process_video_with_heatmap(
                                tmp_path,
                                output_path=out_mp4,
                                class_idx=0,
                                max_frames=max_cam_frames,
                                num_preview_frames=4,
                            )

                        if previews:
                            p_cols = st.columns(len(previews))
                            for idx, p_img in enumerate(previews):
                                p_cols[idx].image(
                                    p_img,
                                    caption=f"Sample {idx + 1}",
                                    use_container_width=True,
                                )

                        if os.path.exists(out_mp4):
                            with open(out_mp4, "rb") as f:
                                st.download_button(
                                    label="⬇️ Download Full Grad-CAM Video (.mp4)",
                                    data=f.read(),
                                    file_name=f"gradcam_{uploaded_file.name}",
                                    mime="video/mp4",
                                    use_container_width=True,
                                )
                    else:
                        with st.spinner("Generating Image Grad-CAM Heatmap..."):
                            cam_img = grad_cam.process_image_with_heatmap(
                                tmp_path, class_idx=0
                            )
                        st.image(
                            cam_img,
                            caption="Grad-CAM Attention Map",
                            use_container_width=True,
                        )

                with st.expander("🧾 Raw JSON Output"):
                    st.json(result)

            except Exception as e:
                st.error(f"Error during analysis: {e}")

            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
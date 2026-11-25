# 🛡️ Hybrid Spatial-Temporal Deepfake Detector

An AI-powered web application for detecting manipulated images, face-swap deepfakes, and full-scene AI-generated videos with visual Grad-CAM explainability.

## ✨ Main Features

- **Image & Video Detection:** Supports common image (`.jpg`, `.jpeg`, `.png`, `.webp`) and video (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`) formats.
- **Dual-Stream Video Analysis:**
  - **Face Specialist:** Uses Dual-Scale MTCNN and an EfficientNet-B0 ensemble to catch facial manipulations and face-swaps.
  - **Environment Specialist:** Uses CLIP ViT-B/16 with a Temporal Transformer on full frames to catch full-scene AI videos (even when no human face is visible).
- **Quality-Gated Adaptive Fusion:** Automatically balances face and environment predictions based on face visibility and sharpness, preventing blurry crops from causing false alarms.
- **Grad-CAM Explainability (XAI):** Highlights suspected manipulated face regions on images and generates downloadable heatmap preview videos.
- **Interactive Streamlit UI:** Upload media, adjust analysis frames, toggle heatmaps, and view frame-by-frame confidence charts in the browser.

## 📁 Project Structure

````text
├── app.py                # Streamlit web interface
├── hybrid_detector.py    # Dual-stream Face + Environment detector
├── clip_env_model.py     # CLIP + Temporal Transformer specialist
├── detector.py           # EfficientNet face ensemble & Bi-GRU fallback
├── grad_cam.py           # Image & video Grad-CAM visualizer
├── network.py            # Model architecture utilities
├── requirements.txt      # Python dependencies
└── models/               # Trained model checkpoints (.pt)



## to clone
git clone <your-repo-url>
cd <your-repo-folder>


## install dependencies

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt

```Add Model Weights
Create a models/ folder in the project root and place your trained .pt checkpoints inside it:

models/clip_temporal_env.pt (Global Environment Specialist)

models/fine_tuned_v1.pt (Primary Face Model)

models/fine_tuned.pt (Secondary Face Model)

models/best_model.pt (Base Face Model)

models/video_gru.pt (Optional Bi-GRU fallback)

````

""RUN:)""
streamlit run app.py



"""
features :"""
Analyze Media:
Open the local URL shown in your terminal.

    Check the sidebar to confirm your model weights are loaded, and adjust the frame count or Grad-CAM toggle if needed.

    Upload an image or video and click Run Detection.

    Review the REAL / FAKE verdict, stream breakdown, frame chart, and Grad-CAM heatmap.

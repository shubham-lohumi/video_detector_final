import os
import pickle
import cv2
import numpy as np
from PIL import Image

from detector import DeepfakeDetector, VIDEO_EXTENSIONS
from clip_env_model import CLIPEnvDetector


class HybridDeepfakeDetector(DeepfakeDetector):
    """
    Production Dual-Stream Hybrid Detector:
    - Stream 1 (Local Face Specialist): 3-Model EfficientNet-B0 Ensemble + Dual-Scale MTCNN
      guarded by face confidence (>=0.88), minimum face resolution (>=64px), and sharpness filter.
    - Stream 2 (Global Environment Specialist): CLIP ViT-B/16 (768-dim) + 4-Layer Temporal Transformer
      running on full, aspect-ratio-preserving video frames.
    - Stream 3 (Adaptive Meta-Fusion): Uses a learned LogisticRegression meta-classifier if present
      ('models/meta_fusion.pkl'), or calibrated quality-gated asymmetric fusion otherwise.
    """
    def __init__(
        self,
        model_path="models/fine_tuned.pt",
        gru_model_path="models/video_gru.pt",
        clip_env_path="models/clip_temporal_env.pt",
        meta_fusion_path="models/meta_fusion.pkl",
        num_video_frames=16,
    ):
        super().__init__(
            model_path=model_path,
            gru_model_path=gru_model_path,
            num_video_frames=num_video_frames,
        )
        self.env_detector = CLIPEnvDetector(checkpoint_path=clip_env_path, device=self.device)
        self.meta_model = None
        if meta_fusion_path and os.path.exists(meta_fusion_path):
            with open(meta_fusion_path, "rb") as f:
                self.meta_model = pickle.load(f)
            print(f"✅ Loaded Learned Meta-Fusion Classifier: '{meta_fusion_path}'")

    def _assess_face_quality(self, pil_img, min_prob=0.88, min_face_px=48):
        """
        Checks if a frame contains a real, sharp, sufficiently large human face.
        Prevents tiny compressed meme/Reels boxes or logos from triggering false positives.
        """
        try:
            boxes, probs = self.mtcnn_20.detect(pil_img)
        except Exception:
            return False, 0.0, 0.0

        if boxes is None or probs is None or len(boxes) == 0:
            return False, 0.0, 0.0

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = int(np.argmax(areas))
        prob = float(probs[idx]) if probs[idx] is not None else 0.0
        x1, y1, x2, y2 = boxes[idx]
        bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)

        if prob < min_prob or min(bw, bh) < min_face_px:
            return False, prob, 0.0

        # Measure Laplacian sharpness strictly inside the detected face box
        w, h = pil_img.size
        crop = pil_img.crop((max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))))
        gray = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return True, prob, sharpness

    @staticmethod
    def _normalize_prob_around_threshold(p, thresh):
        """Maps calibrated probability p so that `thresh` aligns cleanly to 0.50."""
        if p <= thresh:
            return 0.50 * (p / max(thresh, 1e-5))
        return 0.50 + 0.50 * ((p - thresh) / max(1.0 - thresh, 1e-5))

    def predict_video(self, video_path, num_frames=None):
        num_frames = num_frames or self.num_video_frames
        pil_frames = self._extract_video_frames(video_path, num_frames=num_frames)

        # ---------------- STREAM 1: QUALITY-GATED FACE SPECIALIST ----------------
        sharp_face_scores = []
        all_face_scores = []
        sharpness_values = []

        for frame_img in pil_frames:
            is_valid, _, sharpness = self._assess_face_quality(frame_img)
            if not is_valid:
                continue

            batch, _ = self._prepare_face_crops(frame_img, min_face_prob=0.85)
            fake_score, _, _ = self._predict_single_batch(batch)
            all_face_scores.append(fake_score)
            sharpness_values.append(sharpness)

            # Ignore motion-blurred / heavily compressed face crops (sharpness < 35)
            if sharpness >= 35.0:
                sharp_face_scores.append(fake_score)

        faces_detected = len(all_face_scores)
        valid_face_ratio = len(sharp_face_scores) / max(1, len(pil_frames))
        mean_sharpness = float(np.mean(sharpness_values)) if sharpness_values else 0.0

        if len(sharp_face_scores) >= 2:
            face_scores_used = sharp_face_scores
        else:
            face_scores_used = all_face_scores

        if len(face_scores_used) >= 1:
            arr = np.array(face_scores_used, dtype=np.float32)
            # Median-dominant score resists occasional compression spikes
            p_face = float(0.65 * np.median(arr) + 0.35 * np.mean(arr))
        else:
            p_face = None

        # ---------------- STREAM 2: CLIP-TEMPORAL ENVIRONMENT SPECIALIST ----------------
        env_res = self.env_detector.predict_frames(pil_frames) if self.env_detector.is_ready else None

        if env_res is not None:
            p_env_raw = env_res["env_fake_prob"]
            env_thresh = env_res["env_threshold"]
            p_env_aligned = self._normalize_prob_around_threshold(p_env_raw, env_thresh)
            temporal_flicker = env_res["scene_drift"]
            env_attn = env_res["env_attn_weights"]
        else:
            # Graceful fallback to base Bi-GRU detector if CLIP checkpoint isn't downloaded yet
            return super().predict_video(video_path, num_frames=num_frames)

        # ---------------- STREAM 3: ADAPTIVE META-FUSION ----------------
        has_reliable_face = (p_face is not None) and (
            valid_face_ratio >= 0.15
            or (len(sharp_face_scores) >= 2 and mean_sharpness >= 80.0)
        )

        if self.meta_model is not None and has_reliable_face:
            meta_x = np.array([[
                p_face,
                p_env_raw,
                valid_face_ratio,
                min(mean_sharpness / 300.0, 1.0),
                temporal_flicker,
            ]], dtype=np.float32)
            final_fake_conf = float(self.meta_model.predict_proba(meta_x)[0, 1])
            fusion_mode = "LEARNED_META_FUSION"

        elif not has_reliable_face:
            # No clean/sharp human face -> 100% Environment CLIP-Temporal Specialist
            final_fake_conf = p_env_aligned
            fusion_mode = "ENVIRONMENT_ONLY_CLIP"

        else:
            # Quality-weighted asymmetric fusion:
            # 1. Very sharp face strongly REAL (<= 0.15): require overwhelming raw CLIP (>= 0.95) to override
            # 2. Overwhelming raw CLIP (>= 0.93): full-scene AI video (trust Env)
            # 3. Sharp face strongly FAKE (>= 0.72): face-swap deepfake (trust Face)
            # 4. Otherwise blend weighted by face reliability
            face_trust = min(0.55, 0.25 + 0.35 * valid_face_ratio)

            if p_face <= 0.15 and mean_sharpness >= 100.0 and p_env_raw < 0.95:
                final_fake_conf = 0.45 * p_env_aligned + 0.55 * p_face
            elif p_env_raw >= 0.93:
                final_fake_conf = max(p_env_aligned, 0.75 * p_env_aligned + 0.25 * p_face)
            elif p_face >= 0.72 and mean_sharpness >= 60.0 and valid_face_ratio >= 0.25:
                final_fake_conf = max(p_face, 0.65 * p_face + 0.35 * p_env_aligned)
            else:
                final_fake_conf = (1.0 - face_trust) * p_env_aligned + face_trust * p_face
            fusion_mode = "DUAL_STREAM_ADAPTIVE"

        final_fake_conf = float(min(max(final_fake_conf, 0.01), 0.99))
        final_real_conf = 1.0 - final_fake_conf
        label = "FAKE" if final_fake_conf > 0.50 else "REAL"

        return {
            "Label": label,
            "Real Confidence": round(final_real_conf, 4),
            "Fake Confidence": round(final_fake_conf, 4),
            "Media Type": "VIDEO",
            "Fusion Mode": fusion_mode,
            "Frames Analyzed": len(pil_frames),
            "Faces Detected": faces_detected,
            "Sharp Face Ratio": round(valid_face_ratio, 3),
            "Mean Face Sharpness": round(mean_sharpness, 1),
            "Face Stream Score": round(p_face, 4) if p_face is not None else None,
            "Env CLIP Score (Raw)": round(p_env_raw, 4),
            "GRU Temporal Score": round(p_env_aligned, 4),  # Keeps app.py metric card working
            "Temporal Flicker": round(temporal_flicker, 4),
            "Frame Fake Scores": [round(float(s), 4) for s in (face_scores_used if face_scores_used else env_attn)],
        }
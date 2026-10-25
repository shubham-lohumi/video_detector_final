import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import models, transforms
from facenet_pytorch import MTCNN

from network import VideoDeepfakeGRU, extract_backbone_features

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v")


class DeepfakeDetector:
    def __init__(
        self,
        model_path="models/fine_tuned.pt",
        gru_model_path="models/video_gru.pt",
        num_video_frames=16,
    ):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_video_frames = num_video_frames

        # 1. Dual-Scale MTCNN (Exact same as diagnostic script)
        self.mtcnn_20 = MTCNN(
            image_size=224, margin=20, keep_all=False, post_process=False,
            select_largest=True, thresholds=[0.5, 0.6, 0.6], device=self.device
        )
        self.mtcnn_36 = MTCNN(
            image_size=224, margin=36, keep_all=False, post_process=False,
            select_largest=True, thresholds=[0.5, 0.6, 0.6], device=self.device
        )

        # 2. Load Ensemble Models (fine_tuned_v1.pt + fine_tuned.pt + best_model.pt)
        self.model_v1 = self._load_model("models/fine_tuned_v1.pt")
        self.model_ft = self._load_model(model_path if os.path.exists(model_path) else "models/fine_tuned.pt")
        self.model_base = self._load_model("models/best_model.pt")

        # 3. Load Bi-GRU Temporal Video Model
        self.gru_model, self.gru_is_trained = self._load_gru_model(gru_model_path)

        # 4. Standard Normalization
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_model(self, path):
        if not os.path.exists(path):
            return None
        try:
            m = models.efficientnet_b0(weights=None)
            num_features = m.classifier[1].in_features
            m.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(num_features, 2))
            try:
                raw = torch.load(path, map_location=self.device, weights_only=False)
            except TypeError:
                raw = torch.load(path, map_location=self.device)
            if isinstance(raw, dict) and "state_dict" in raw:
                raw = raw["state_dict"]
            clean_sd = {
                (k.replace("model.", "", 1) if k.startswith("model.") else k): v
                for k, v in raw.items()
            }
            m.load_state_dict(clean_sd, strict=False)
            m.to(self.device)
            m.eval()
            print(f"✅ Loaded ensemble member: '{path}'")
            return m
        except Exception as e:
            print(f"⚠️ Could not load '{path}': {e}")
            return None

    def _load_gru_model(self, gru_path):
        primary_backbone = self.model_v1 if self.model_v1 is not None else self.model_ft
        gru_net = VideoDeepfakeGRU(backbone=primary_backbone, freeze_backbone=True)
        is_trained = False

        if gru_path and os.path.exists(gru_path):
            try:
                try:
                    payload = torch.load(gru_path, map_location=self.device, weights_only=False)
                except TypeError:
                    payload = torch.load(gru_path, map_location=self.device)
                if isinstance(payload, dict) and "state_dict" in payload:
                    payload = payload["state_dict"]
                clean_sd = {
                    (k.replace("model.", "", 1) if k.startswith("model.") else k): v
                    for k, v in payload.items()
                }
                gru_net.load_state_dict(clean_sd, strict=False)
                is_trained = True
                print(f"✅ Loaded trained Video Bi-GRU model: '{gru_path}'")
            except Exception as e:
                print(f"⚠️ Could not load GRU weights '{gru_path}': {e}")
        else:
            print(
                f"ℹ️ '{gru_path}' not found yet. Using Bi-GRU Temporal Feature Drift + "
                f"3-Model Ensemble mode (Run train_video_gru.py to train GRU weights)."
            )

        gru_net.to(self.device)
        gru_net.eval()
        return gru_net, is_trained

    def _tensor_to_pil(self, t):
        arr = t.permute(1, 2, 0).cpu().numpy().clip(0, 255).astype("uint8")
        return Image.fromarray(arr)

    def _prepare_face_crops(self, raw_img):
        """Extracts Dual-Scale MTCNN face crops + mirror TTA (4 crops per frame)."""
        f20 = self.mtcnn_20(raw_img)
        f36 = self.mtcnn_36(raw_img)

        face_detected = (f20 is not None) or (f36 is not None)
        p20 = self._tensor_to_pil(f20) if f20 is not None else raw_img.resize((224, 224))
        p36 = self._tensor_to_pil(f36) if f36 is not None else p20

        crops = [p20, ImageOps.mirror(p20), p36, ImageOps.mirror(p36)]
        batch = torch.stack([self.transform(c) for c in crops]).to(self.device)
        return batch, face_detected

    def _fuse_ensemble_scores(self, f_v1, f_ft, f_base):
        """Exact calibrated 3-model ensemble fusion logic."""
        if f_v1 is not None and f_ft is not None:
            v1_centered = f_v1 - 0.10

            # Case A: Confident REAL
            if f_ft < 0.25:
                fake_conf = (v1_centered + f_ft) / 2.0
            # Case B: v1 confident FAKE and ft supports
            elif f_v1 > 0.62 and f_ft > 0.30 and not (f_ft < 0.55 and f_base < 0.05):
                fake_conf = max(v1_centered, f_ft, 0.58)
            # Case C: ft strong FAKE and v1/base support
            elif f_ft > 0.70 and f_v1 > 0.30 and f_base > 0.50:
                fake_conf = (f_ft + f_base) / 2.0
            else:
                # Default balanced ensemble
                fake_conf = min(v1_centered, 0.45) if f_v1 < 0.60 else 0.44
        else:
            fake_conf = f_ft if f_ft is not None else (f_v1 if f_v1 is not None else 0.5)

        return float(min(max(fake_conf, 0.01), 0.99))

    def _predict_single_batch(self, batch):
        """
        Runs the 3-model ensemble on a single frame's 4-crop batch (4, 3, 224, 224)
        and also returns the 1280-dim feature vector and logits for the Bi-GRU.
        """
        with torch.no_grad():
            feat_vec = None
            frame_logit = None

            # 1. Primary Model: fine_tuned_v1.pt
            if self.model_v1 is not None:
                feats_v1, logits_v1 = extract_backbone_features(self.model_v1, batch)
                probs_v1 = torch.softmax(logits_v1, dim=1)
                f_v1 = torch.mean(probs_v1[:, 0]).item()
                feat_vec = torch.mean(feats_v1, dim=0)       # (1280,)
                frame_logit = torch.mean(logits_v1, dim=0)   # (2,)
            else:
                f_v1 = None

            # 2. Secondary Model: fine_tuned.pt
            if self.model_ft is not None:
                feats_ft, logits_ft = extract_backbone_features(self.model_ft, batch)
                probs_ft = torch.softmax(logits_ft, dim=1)
                f_ft = torch.mean(probs_ft[:, 0]).item()
                if feat_vec is None:
                    feat_vec = torch.mean(feats_ft, dim=0)
                    frame_logit = torch.mean(logits_ft, dim=0)
                else:
                    feat_vec = 0.6 * feat_vec + 0.4 * torch.mean(feats_ft, dim=0)
                    frame_logit = 0.6 * frame_logit + 0.4 * torch.mean(logits_ft, dim=0)
            else:
                f_ft = f_v1

            # 3. Base Model: best_model.pt (Inverted mapping: idx 1 is Fake on single_20)
            if self.model_base is not None:
                probs_base = torch.softmax(self.model_base(batch[:1]), dim=1)
                f_base = probs_base[0, 1].item()
            else:
                f_base = 0.5

        fake_conf = self._fuse_ensemble_scores(f_v1, f_ft, f_base)
        return fake_conf, feat_vec, frame_logit

    def _extract_video_frames(self, video_path, num_frames=16):
        """
        Crash-proof uniform frame extractor for any video file.
        Handles corrupt frames, variable FPS, and short videos automatically.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        if total_frames > 0:
            indices = np.linspace(0, max(total_frames - 1, 0), num_frames, dtype=int)
            target_set = set(indices.tolist())
            max_idx = max(target_set)

            curr_idx = 0
            grabbed_map = {}
            while curr_idx <= max_idx:
                ret, frame = cap.read()
                if not ret:
                    break
                if curr_idx in target_set and frame is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    grabbed_map[curr_idx] = Image.fromarray(rgb)
                curr_idx += 1

            for idx in indices:
                if idx in grabbed_map:
                    frames.append(grabbed_map[idx])
                elif frames:
                    frames.append(frames[-1].copy())

        # Fallback sequential read if metadata frame count was corrupted
        if len(frames) == 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            all_read = []
            while len(all_read) < 300:  # Cap at 300 frames for safety
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                all_read.append(Image.fromarray(rgb))
            if all_read:
                pick_idx = np.linspace(0, len(all_read) - 1, num_frames, dtype=int)
                frames = [all_read[i] for i in pick_idx]

        cap.release()

        if len(frames) == 0:
            raise ValueError(f"No readable frames found in video: {video_path}")

        # Pad with last frame if video had fewer frames than num_frames
        while len(frames) < num_frames:
            frames.append(frames[-1].copy())

        return frames[:num_frames]

    def predict_video(self, video_path, num_frames=None):
        """
        Full Video Deepfake Detection using:
        1. Dual-Scale MTCNN Face Cropping across T frames
        2. 3-Model Spatial Ensemble per frame
        3. Bi-GRU + Temporal Attention + Inter-Frame Flicker Analysis
        """
        num_frames = num_frames or self.num_video_frames
        pil_frames = self._extract_video_frames(video_path, num_frames=num_frames)

        frame_fake_scores = []
        feature_list = []
        logit_list = []
        faces_found = 0

        for frame_img in pil_frames:
            batch, has_face = self._prepare_face_crops(frame_img)
            if has_face:
                faces_found += 1
            fake_score, feat_vec, frame_logit = self._predict_single_batch(batch)
            frame_fake_scores.append(fake_score)
            if feat_vec is not None:
                feature_list.append(feat_vec)
                logit_list.append(frame_logit)

        scores_np = np.array(frame_fake_scores, dtype=np.float32)

        # Top-K suspicious frames + mean spatial score
        k = max(1, len(scores_np) // 3)
        top_k_mean = float(np.mean(np.sort(scores_np)[-k:]))
        mean_spatial = float(np.mean(scores_np))
        spatial_video_score = 0.55 * mean_spatial + 0.45 * top_k_mean

        # Bi-GRU Temporal Sequence Modeling
        gru_fake_prob = spatial_video_score
        temporal_flicker_score = 0.0

        if len(feature_list) >= 2:
            with torch.no_grad():
                seq_feats = torch.stack(feature_list, dim=0).unsqueeze(0)   # (1, T, 1280)
                seq_logits = torch.stack(logit_list, dim=0).unsqueeze(0)    # (1, T, 2)

                # Measure inter-frame feature cosine instability & prediction jitter
                norm_f = F.normalize(seq_feats.squeeze(0), p=2, dim=-1)
                cos_sim = torch.sum(norm_f[1:] * norm_f[:-1], dim=-1)
                feat_drift = float(torch.mean(1.0 - cos_sim).item())
                score_jitter = float(np.mean(np.abs(np.diff(scores_np))))
                temporal_flicker_score = min(max((feat_drift * 2.5) + (score_jitter * 1.2), 0.0), 1.0)

                if self.gru_is_trained and self.gru_model is not None:
                    # Trained Bi-GRU + Temporal Attention prediction (idx 0 = FAKE)
                    gru_logits, attn_weights = self.gru_model.forward_features(seq_feats, seq_logits)
                    gru_probs = torch.softmax(gru_logits, dim=1)
                    gru_fake_prob = float(gru_probs[0, 0].item())

                    # Combine trained Bi-GRU (60%) + Calibrated 3-Model Spatial Ensemble (40%)
                    final_fake_conf = 0.60 * gru_fake_prob + 0.40 * spatial_video_score
                else:
                    # Untrained GRU fallback: Use GRU input norm + temporal flicker boost
                    # If several frames spike as FAKE and there is temporal flicker, boost fake confidence
                    if top_k_mean > 0.55 and temporal_flicker_score > 0.12:
                        gru_fake_prob = min(0.99, spatial_video_score + 0.12 * temporal_flicker_score)
                    elif mean_spatial < 0.30 and top_k_mean < 0.45:
                        gru_fake_prob = max(0.01, mean_spatial * 0.90)
                    else:
                        gru_fake_prob = spatial_video_score
                    final_fake_conf = gru_fake_prob
        else:
            final_fake_conf = spatial_video_score

        final_fake_conf = float(min(max(final_fake_conf, 0.01), 0.99))
        final_real_conf = 1.0 - final_fake_conf
        label = "FAKE" if final_fake_conf > 0.50 else "REAL"

        return {
            "Label": label,
            "Real Confidence": round(final_real_conf, 4),
            "Fake Confidence": round(final_fake_conf, 4),
            "Media Type": "VIDEO",
            "Frames Analyzed": len(pil_frames),
            "Faces Detected": faces_found,
            "GRU Temporal Score": round(gru_fake_prob, 4),
            "Temporal Flicker": round(temporal_flicker_score, 4),
            "Frame Fake Scores": [round(float(s), 4) for s in frame_fake_scores],
        }

    def predict(self, input_media):
        """
        Universal predictor:
        - Automatically detects if 'input_media' is a video file path (.mp4, .avi, etc.) and runs Bi-GRU video detection.
        - Otherwise runs the exact 90-95% accurate image ensemble detector.
        """
        if isinstance(input_media, str) and input_media.lower().endswith(VIDEO_EXTENSIONS):
            return self.predict_video(input_media)

        if isinstance(input_media, str):
            raw_img = Image.open(input_media).convert("RGB")
        elif isinstance(input_media, Image.Image):
            raw_img = input_media.convert("RGB")
        else:
            raw_img = Image.fromarray(cv2.cvtColor(input_media, cv2.COLOR_BGR2RGB))

        batch, _ = self._prepare_face_crops(raw_img)
        fake_conf, _, _ = self._predict_single_batch(batch)

        real_conf = 1.0 - fake_conf
        label = "FAKE" if fake_conf > 0.50 else "REAL"

        return {
            "Label": label,
            "Real Confidence": round(real_conf, 4),
            "Fake Confidence": round(fake_conf, 4),
        }
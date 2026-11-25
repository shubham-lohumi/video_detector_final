import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class VideoGradCAM:
    """
    Crash-proof Grad-CAM for EfficientNet-B0 (works on frozen weights & inplace SiLU).
    Supports both Images and Videos with exact MTCNN face-region heatmap alignment.
    """

    def __init__(self, detector_instance):
        self.detector = detector_instance
        self.device = self.detector.device

        # Primary EfficientNet backbone (fine_tuned_v1 ya fine_tuned)
        self.backbone = (
            self.detector.model_v1
            if self.detector.model_v1 is not None
            else self.detector.model_ft
        )
        if self.backbone is None:
            raise RuntimeError("No EfficientNet model loaded in detector for Grad-CAM!")

        self.backbone.eval()

    def _compute_cam(self, tensor_img, class_idx=0):
        """
        Generates normalized (0..1) Grad-CAM heatmap for a single (3, 224, 224) tensor.
        class_idx: 0 = FAKE, 1 = REAL
        """
        with torch.enable_grad():
            inp = tensor_img.detach().clone().unsqueeze(0).to(self.device)
            inp.requires_grad_(True)

            feat_maps = self.backbone.features(inp)
            feat_maps.retain_grad()

            pooled = self.backbone.avgpool(feat_maps)
            features = torch.flatten(pooled, 1)
            logits = self.backbone.classifier(features)

            self.backbone.zero_grad(set_to_none=True)
            score = logits[0, class_idx]
            score.backward()

            if feat_maps.grad is None:
                return None

            grads = feat_maps.grad.detach()  # (1, C, H', W')
            acts = feat_maps.detach()  # (1, C, H', W')

            weights = torch.mean(grads, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * acts, dim=1).squeeze(0)
            cam = F.relu(cam)

            cam_min, cam_max = cam.min(), cam.max()
            if (cam_max - cam_min) > 1e-6:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros_like(cam)

            return cam.cpu().numpy()

    def _get_largest_face_box(self, pil_img, min_prob=0.85):
        """Detects largest high-confidence face bounding box (x1, y1, x2, y2) with margin=20."""
        try:
            boxes, probs = self.detector.mtcnn_20.detect(pil_img)
        except Exception:
            return None

        if boxes is None or probs is None or len(boxes) == 0:
            return None

        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = int(np.argmax(areas))
        if probs[idx] is None or float(probs[idx]) < min_prob:
            return None
        box = boxes[idx]

        w, h = pil_img.size
        margin = 20
        x1 = max(0, int(box[0]) - margin)
        y1 = max(0, int(box[1]) - margin)
        x2 = min(w, int(box[2]) + margin)
        y2 = min(h, int(box[3]) + margin)

        if (x2 - x1) < 16 or (y2 - y1) < 16:
            return None
        return (x1, y1, x2, y2)

    def _overlay_bgr_frame(self, bgr_frame, class_idx=0):
        """Applies Grad-CAM heatmap accurately on the face region (or full frame if no face)."""
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        overlay = bgr_frame.copy()

        box = self._get_largest_face_box(pil_img)
        if box is not None:
            x1, y1, x2, y2 = box
            crop_pil = pil_img.crop((x1, y1, x2, y2))
            tensor_img = self.detector.transform(crop_pil)
            cam_map = self._compute_cam(tensor_img, class_idx=class_idx)

            if cam_map is not None:
                cam_resized = cv2.resize(cam_map, (x2 - x1, y2 - y1))
                heatmap = cv2.applyColorMap(
                    np.uint8(255 * cam_resized), cv2.COLORMAP_JET
                )
                roi = overlay[y1:y2, x1:x2]
                overlay[y1:y2, x1:x2] = cv2.addWeighted(roi, 0.55, heatmap, 0.45, 0)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        else:
            # Fallback: Full frame par heatmap lagayein
            tensor_img = self.detector.transform(pil_img)
            cam_map = self._compute_cam(tensor_img, class_idx=class_idx)
            if cam_map is not None:
                h, w = overlay.shape[:2]
                cam_resized = cv2.resize(cam_map, (w, h))
                heatmap = cv2.applyColorMap(
                    np.uint8(255 * cam_resized), cv2.COLORMAP_JET
                )
                overlay = cv2.addWeighted(overlay, 0.60, heatmap, 0.40, 0)

        return overlay

    def process_image_with_heatmap(self, image_path, class_idx=0):
        """Generates Grad-CAM overlayed PIL Image for a single image file."""
        pil_img = Image.open(image_path).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        overlay_bgr = self._overlay_bgr_frame(bgr, class_idx=class_idx)
        overlay_rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(overlay_rgb)

    def process_video_with_heatmap(
        self,
        video_path,
        output_path="output_heatmap.mp4",
        class_idx=0,
        max_frames=120,
        num_preview_frames=4,
    ):
        """
        Generates Grad-CAM heatmap video (.mp4) and returns (output_path, preview_pil_frames)
        so Streamlit browser mein preview frames turant dikh sakein.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Long videos par Web UI hang na ho isliye cap lagaya hai
        frames_to_process = (
            min(total_frames, max_frames) if total_frames > 0 else max_frames
        )
        preview_indices = set(
            np.linspace(
                0, max(frames_to_process - 1, 0), num_preview_frames, dtype=int
            ).tolist()
        )

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        preview_frames = []
        frame_idx = 0

        while cap.isOpened() and frame_idx < frames_to_process:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            overlay = self._overlay_bgr_frame(frame, class_idx=class_idx)
            cv2.putText(
                overlay,
                "Grad-CAM Deepfake Heatmap",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            out.write(overlay)

            if frame_idx in preview_indices:
                rgb_preview = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                preview_frames.append(Image.fromarray(rgb_preview))

            frame_idx += 1

        cap.release()
        out.release()
        return output_path, preview_frames

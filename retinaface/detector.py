import torch
import numpy as np
from PIL import Image
from typing import Optional

from retinaface.config import cfg_re50
from retinaface.retinaface_model import RetinaFace
from retinaface.prior_box import PriorBox
from retinaface.box_utils import decode, decode_landm
from retinaface.py_cpu_nms import py_cpu_nms


class RetinaFaceDetector:
    """High-level wrapper for RetinaFace face detection on MPS/CPU."""

    def __init__(
        self,
        weights_path: str,
        device: str = "mps",
        confidence_threshold: float = 0.02,
        nms_threshold: float = 0.4,
        top_k: int = 5000,
        keep_top_k: int = 750,
        vis_threshold: float = 0.6,
        min_face_size: int = 30,
    ):
        self.device = torch.device(device if torch.backends.mps.is_available() or device == "cpu" else "cpu")
        self.cfg = cfg_re50
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.keep_top_k = keep_top_k
        self.vis_threshold = vis_threshold
        self.min_face_size = min_face_size

        # Build model
        self.model = RetinaFace(cfg=self.cfg, phase='test')
        self._load_weights(weights_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_weights(self, weights_path: str):
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        # Strip 'module.' prefix from DataParallel
        cleaned = {}
        for k, v in state_dict.items():
            name = k[len("module."):] if k.startswith("module.") else k
            cleaned[name] = v
        self.model.load_state_dict(cleaned, strict=False)

    @torch.no_grad()
    def detect(self, image: Image.Image) -> list[dict]:
        """
        Detect faces in a PIL Image.

        Returns list of dicts with keys:
            'bbox': (x1, y1, x2, y2) in original image coordinates
            'confidence': float
            'landmarks': list of 5 (x, y) tuples
        """
        img_width, img_height = image.size

        # Convert PIL RGB to numpy BGR (match original OpenCV-based training)
        img_np = np.array(image)[:, :, ::-1].astype(np.float32)
        img_np -= (104, 117, 123)

        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        scale = torch.Tensor([img_width, img_height, img_width, img_height]).to(self.device)

        loc, conf, landms = self.model(img_tensor)

        # Generate priors
        priorbox = PriorBox(self.cfg, image_size=(img_height, img_width))
        priors = priorbox.forward().to(self.device)
        prior_data = priors.data

        # Decode
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()

        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        landms_decoded = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img_width, img_height] * 5).to(self.device)
        landms_decoded = landms_decoded * scale1
        landms_decoded = landms_decoded.cpu().numpy()

        # Filter by confidence
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms_decoded = landms_decoded[inds]
        scores = scores[inds]

        # Keep top-K before NMS
        order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms_decoded = landms_decoded[order]
        scores = scores[order]

        # NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        landms_decoded = landms_decoded[keep]

        # Keep top-K after NMS
        dets = dets[:self.keep_top_k, :]
        landms_decoded = landms_decoded[:self.keep_top_k]

        # Build results
        results = []
        for i in range(dets.shape[0]):
            if dets[i, 4] < self.vis_threshold:
                continue
            x1, y1, x2, y2 = dets[i, :4].astype(int)
            face_w = x2 - x1
            face_h = y2 - y1
            if face_w < self.min_face_size or face_h < self.min_face_size:
                continue
            landmarks = []
            for j in range(5):
                lx = int(landms_decoded[i, 2 * j])
                ly = int(landms_decoded[i, 2 * j + 1])
                landmarks.append((lx, ly))
            results.append({
                'bbox': (x1, y1, x2, y2),
                'confidence': float(dets[i, 4]),
                'landmarks': landmarks,
            })
        return results

    def detect_faces(
        self,
        image: Image.Image,
        crop_size: int = 256,
        margin: float = 0.20,
        max_faces: Optional[int] = None,
    ) -> list[Image.Image]:
        """
        Detect faces and return cropped PIL Images.

        Args:
            image: Input PIL Image
            crop_size: Output size for each face crop
            margin: Fraction to expand bbox (e.g., 0.20 = 20% expansion)
            max_faces: Maximum number of faces to return (None = all)

        Returns:
            List of PIL Images, each crop_size x crop_size
        """
        detections = self.detect(image)
        if max_faces is not None:
            detections = detections[:max_faces]

        img_width, img_height = image.size
        crops = []
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            face_w = x2 - x1
            face_h = y2 - y1

            # Expand bbox by margin
            dx = int(face_w * margin)
            dy = int(face_h * margin)
            x1 = max(0, x1 - dx)
            y1 = max(0, y1 - dy)
            x2 = min(img_width, x2 + dx)
            y2 = min(img_height, y2 + dy)

            crop = image.crop((x1, y1, x2, y2)).resize((crop_size, crop_size), Image.BILINEAR)
            crops.append(crop)
        return crops

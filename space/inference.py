"""DeepSentinel inference: face detection, preprocessing, prediction, and Grad-CAM."""

import os
from dataclasses import dataclass

import cv2
import numpy as np
import timm
import torch
import torchvision.transforms as T
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

MODEL_REPO = os.environ.get("MODEL_REPO", "adichaudhary/deepsentinel")
MODEL_FILE = os.environ.get("MODEL_FILE", "deepsentinel_effb4.pt")
LOCAL_WEIGHTS = os.environ.get("LOCAL_WEIGHTS")

# Must match the evaluation transform used in training.
SOURCE_SIZE = 260
CROP_SIZE = 208
IMG_SIZE = 224
FACE_MARGIN = 0.25
THRESHOLD = 0.5

_transform = T.Compose([
    T.CenterCrop(CROP_SIZE),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

_face_detector = cv2.CascadeClassifier(
    os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
)


@dataclass
class Prediction:
    label: str
    p_fake: float
    face_detected: bool
    face_box: list[int] | None
    face: Image.Image
    heatmap: Image.Image | None = None


def _weights_path() -> str:
    if LOCAL_WEIGHTS:
        return LOCAL_WEIGHTS
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)


def load_model(device: str = "cpu") -> torch.nn.Module:
    model = timm.create_model("efficientnet_b4", pretrained=False, num_classes=1)
    model.load_state_dict(torch.load(_weights_path(), map_location=device))
    return model.to(device).eval()


def crop_face(image: Image.Image) -> tuple[Image.Image, list[int] | None]:
    """Crop the largest detected face with a margin. Falls back to a center square crop."""
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = _face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))

    h, w = gray.shape
    if len(faces):
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        side = int(max(fw, fh) * (1 + 2 * FACE_MARGIN))
        cx, cy = x + fw // 2, y + fh // 2
        x1, y1 = max(0, cx - side // 2), max(0, cy - side // 2)
        x2, y2 = min(w, x1 + side), min(h, y1 + side)
        box = [int(x1), int(y1), int(x2), int(y2)]
    else:
        side = min(h, w)
        x1, y1 = (w - side) // 2, (h - side) // 2
        x2, y2 = x1 + side, y1 + side
        box = None

    face = Image.fromarray(rgb[y1:y2, x1:x2]).resize((SOURCE_SIZE, SOURCE_SIZE), Image.BILINEAR)
    return face, box


class DeepSentinel:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_model(self.device)
        self.cam = GradCAM(model=self.model, target_layers=[self.model.conv_head])

    @torch.no_grad()
    def _score(self, x: torch.Tensor) -> float:
        return torch.sigmoid(self.model(x)).item()

    def predict(self, image: Image.Image, with_heatmap: bool = False) -> Prediction:
        face, box = crop_face(image)
        x = _transform(face).unsqueeze(0).to(self.device)
        p_fake = self._score(x)

        heatmap = None
        if with_heatmap:
            grayscale = self.cam(input_tensor=x, targets=[BinaryClassifierOutputTarget(1)])[0]
            view = T.CenterCrop(CROP_SIZE)(face).resize((IMG_SIZE, IMG_SIZE))
            base = np.asarray(view, dtype=np.float32) / 255.0
            heatmap = Image.fromarray(show_cam_on_image(base, grayscale, use_rgb=True))

        return Prediction(
            label="fake" if p_fake >= THRESHOLD else "real",
            p_fake=round(p_fake, 4),
            face_detected=box is not None,
            face_box=box,
            face=face,
            heatmap=heatmap,
        )

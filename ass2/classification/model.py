"""
Load task_2-1 ResNet18 checkpoint and classify sign crops (same preprocessing as training eval).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

from classification.network import ResNet18

CLASS_NAMES = {
    0: "Stop",
    1: "Turn right",
    2: "Turn left",
    3: "Ahead only",
    4: "Roundabout mandatory",
}

IMG_SIZE = 32
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

_DEFAULT_CKPT = Path(__file__).resolve().parent.parent.parent / "Traffic-Sign-Student" / "Traffic-Sign" / "checkpoint" / "best_model.pth"


def _eval_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def _strip_module_prefix(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict
    first = next(iter(state_dict.keys()))
    if first.startswith("module."):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


class TrafficSignClassifier:
    def __init__(self, checkpoint_path: str | Path | None = None, device: torch.device | None = None):
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = _eval_transform()
        self.model = ResNet18(num_classes=5).to(self.device)
        self.model.eval()

        ckpt_path = Path(checkpoint_path) if checkpoint_path else _DEFAULT_CKPT
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Classifier checkpoint not found: {ckpt_path}\n"
                "Train on task_2-1 or copy best_model.pth there."
            )

        try:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path, map_location=self.device)
        raw_sd = ckpt.get("model", ckpt)
        if not isinstance(raw_sd, dict):
            raise ValueError("Checkpoint must contain 'model' state_dict or be a raw state_dict.")
        self.model.load_state_dict(_strip_module_prefix(raw_sd), strict=True)

    @torch.inference_mode()
    def classify_sign_crop(self, sign_bgr: np.ndarray | None) -> tuple[int | None, str | None, float | None]:
        """
        Returns (class_id, class_name, confidence) or (None, None, None) if no valid crop.
        """
        if sign_bgr is None or sign_bgr.size == 0:
            return None, None, None
        if sign_bgr.ndim != 3 or sign_bgr.shape[2] != 3:
            return None, None, None

        rgb = sign_bgr[:, :, ::-1].copy()
        pil = Image.fromarray(rgb.astype(np.uint8))
        x = self.transform(pil).unsqueeze(0).to(self.device)
        logits = self.model(x)
        prob = torch.softmax(logits, dim=1)
        conf, pred = prob.max(dim=1)
        cid = int(pred.item())
        return cid, CLASS_NAMES.get(cid, f"Unknown ({cid})"), float(conf.item())

    def classify_cone_data(self, cone_data: list[dict]) -> None:
        """Add predicted_label, predicted_class_name, classification_confidence to each dict (in-place)."""
        for row in cone_data:
            crop = row.get("sign_crop")
            cid, name, conf = self.classify_sign_crop(crop)
            row["predicted_label"] = cid
            row["predicted_class_name"] = name
            row["classification_confidence"] = conf

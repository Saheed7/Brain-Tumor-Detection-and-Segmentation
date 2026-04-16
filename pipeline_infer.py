from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.data.transforms import CLAHETransform
from src.models.attention_resunet import AttentionResUNet
from src.models.detector import ResNet50Detector
from src.utils.io import ensure_dir, load_yaml
from src.utils.profiling import measure_inference_time
from src.utils.reproducibility import resolve_device, set_seed


def preprocess_for_detector(image: np.ndarray, image_size: int) -> torch.Tensor:
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).unsqueeze(0)


def preprocess_for_segmenter(image: np.ndarray, image_size: int) -> torch.Tensor:
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    image = (image - 0.5) / 0.5
    return torch.from_numpy(image).unsqueeze(0).unsqueeze(0)


def main(config_path: str, input_dir: str, output_dir: str) -> None:
    cfg = load_yaml(config_path)
    set_seed(cfg["seed"])
    device = resolve_device("cuda")
    output_path = ensure_dir(output_dir)

    detector = ResNet50Detector(pretrained=False).to(device)
    detector_ckpt = torch.load(cfg["detector"]["checkpoint"], map_location=device)
    detector.load_state_dict(detector_ckpt["model_state"])
    detector.eval()

    segmenter = AttentionResUNet().to(device)
    segmenter_ckpt = torch.load(cfg["segmenter"]["checkpoint"], map_location=device)
    segmenter.load_state_dict(segmenter_ckpt["model_state"])
    segmenter.eval()

    clahe_cfg = cfg["preprocessing"]["clahe"]
    clahe = CLAHETransform(clip_limit=clahe_cfg["clip_limit"], tile_grid_size=tuple(clahe_cfg["tile_grid_size"])) if clahe_cfg["enabled"] else None

    for image_path in sorted(Path(input_dir).iterdir()):
        if not image_path.is_file():
            continue
        gray = np.array(Image.open(image_path).convert("L"))
        if clahe is not None:
            gray = clahe(gray)

        det_tensor = preprocess_for_detector(gray, cfg["detector"]["image_size"]).to(device)
        with torch.no_grad():
            logits = detector(det_tensor)
            probs = torch.softmax(logits, dim=1)[0, 1].item()

        result = {"filename": image_path.name, "tumor_probability": probs}
        if probs >= cfg["detector"]["threshold"]:
            seg_tensor = preprocess_for_segmenter(gray, cfg["segmenter"]["image_size"]).to(device)
            with torch.no_grad():
                seg_logits = segmenter(seg_tensor)
                seg_mask = (torch.sigmoid(seg_logits) >= 0.5).float()[0, 0].cpu().numpy() * 255
            cv2.imwrite(str(output_path / f"{image_path.stem}_mask.png"), seg_mask.astype(np.uint8))
            result["segmented"] = True
        else:
            result["segmented"] = False

        with open(output_path / f"{image_path.stem}.txt", "w", encoding="utf-8") as f:
            for key, value in result.items():
                f.write(f"{key}: {value}\n")

    sample = torch.randn(1, 3, cfg["detector"]["image_size"], cfg["detector"]["image_size"], device=device)
    print("Detector inference time (ms):", measure_inference_time(detector, sample, device))
    sample = torch.randn(1, 1, cfg["segmenter"]["image_size"], cfg["segmenter"]["image_size"], device=device)
    print("Segmenter inference time (ms):", measure_inference_time(segmenter, sample, device))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    main(args.config, args.input_dir, args.output_dir)

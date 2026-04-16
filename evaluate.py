from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import BrainTumorDetectionDataset, BrainTumorSegmentationDataset
from src.data.transforms import get_detection_transforms, get_segmentation_transforms
from src.models.attention_resunet import AttentionResUNet
from src.models.detector import ResNet50Detector
from src.models.losses import FocalTverskyLoss
from src.training.train_detector import evaluate as evaluate_detector
from src.training.train_segmenter import evaluate as evaluate_segmenter
from src.utils.io import load_yaml
from src.utils.reproducibility import resolve_device, set_seed


def main(task: str, config_path: str) -> None:
    cfg = load_yaml(config_path)
    set_seed(cfg["seed"])
    device = resolve_device(cfg["hardware"].get("device", "cuda"))

    if task == "detector":
        tfms = get_detection_transforms(cfg["data"]["image_size"], False, cfg["preprocessing"]["normalize_mean"], cfg["preprocessing"]["normalize_std"])
        ds = BrainTumorDetectionDataset(root=str(Path(cfg["data"]["root"]) / "test"), transform=tfms)
        loader = DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=False)
        model = ResNet50Detector(cfg["model"]["num_classes"], cfg["model"]["pretrained"], cfg["model"]["dropout"]).to(device)
        ckpt = torch.load(Path(cfg["output_dir"]) / cfg["save_name"], map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(evaluate_detector(model, loader, device))
    else:
        tfms = get_segmentation_transforms(cfg["data"]["image_size"], False, cfg["preprocessing"]["normalize_mean"], cfg["preprocessing"]["normalize_std"])
        ds = BrainTumorSegmentationDataset(str(Path(cfg["data"]["root"]) / "test" / "images"), str(Path(cfg["data"]["root"]) / "test" / "masks"), transform=tfms)
        loader = DataLoader(ds, batch_size=cfg["data"]["batch_size"], shuffle=False)
        model = AttentionResUNet(cfg["model"]["in_channels"], cfg["model"]["out_channels"], cfg["model"]["base_channels"], cfg["model"]["use_attention"]).to(device)
        ckpt = torch.load(Path(cfg["output_dir"]) / cfg["save_name"], map_location=device)
        model.load_state_dict(ckpt["model_state"])
        criterion = FocalTverskyLoss(cfg["training"]["alpha"], cfg["training"]["beta"], cfg["training"]["gamma"])
        print(evaluate_segmenter(model, loader, criterion, device))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["detector", "segmenter"], required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.task, args.config)

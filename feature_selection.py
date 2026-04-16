from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import BrainTumorDetectionDataset
from src.data.transforms import get_detection_transforms
from src.models.detector import ResNet50Detector
from src.utils.io import ensure_dir, load_yaml
from src.utils.reproducibility import resolve_device, set_seed


def run_feature_selection(config_path: str, stage: str = "train") -> None:
    cfg = load_yaml(config_path)
    set_seed(cfg["seed"])
    device = resolve_device(cfg["hardware"].get("device", "cuda"))
    output_dir = ensure_dir(cfg["output_dir"])

    val_tfms = get_detection_transforms(
        image_size=cfg["data"]["image_size"],
        train=False,
        mean=cfg["preprocessing"]["normalize_mean"],
        std=cfg["preprocessing"]["normalize_std"],
    )
    clahe_cfg = cfg["preprocessing"]["clahe"]
    dataset = BrainTumorDetectionDataset(
        root=str(Path(cfg["data"]["root"]) / stage),
        transform=val_tfms,
        use_clahe=clahe_cfg["enabled"],
        clahe_clip_limit=clahe_cfg["clip_limit"],
        clahe_tile_grid_size=tuple(clahe_cfg["tile_grid_size"]),
        grayscale_to_rgb=cfg["data"].get("grayscale_to_rgb", True),
    )
    loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size"], shuffle=False)

    model = ResNet50Detector(num_classes=cfg["model"]["num_classes"], pretrained=cfg["model"]["pretrained"], dropout=cfg["model"]["dropout"]).to(device)
    model.eval()

    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Extracting GAP features"):
            images = images.to(device)
            features = model.extract_gap_features(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)

    if cfg["feature_selection"].get("smote", True) and stage == "train":
        X, y = SMOTE(random_state=cfg["seed"]).fit_resample(X, y)

    estimator = LogisticRegression(max_iter=1000, solver="liblinear", random_state=cfg["seed"])
    selector = RFE(estimator=estimator, n_features_to_select=cfg["feature_selection"]["rfe_num_features"], step=0.1)
    X_rfe = selector.fit_transform(X, y)

    pca = PCA(n_components=cfg["feature_selection"]["pca_variance"], random_state=cfg["seed"])
    X_pca = pca.fit_transform(X_rfe)

    np.save(output_dir / f"{stage}_gap_features.npy", X)
    np.save(output_dir / f"{stage}_labels.npy", y)
    np.save(output_dir / f"{stage}_features_rfe_pca.npy", X_pca)
    joblib.dump(selector, output_dir / "detector_feature_selector.joblib")
    joblib.dump(pca, output_dir / "detector_pca.joblib")
    print(f"Original shape: {X.shape}")
    print(f"After RFE: {X_rfe.shape}")
    print(f"After PCA: {X_pca.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stage", type=str, default="train", choices=["train", "val", "test"])
    args = parser.parse_args()
    run_feature_selection(args.config, args.stage)

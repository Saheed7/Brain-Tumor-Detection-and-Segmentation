from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import BrainTumorDetectionDataset
from src.data.transforms import get_detection_transforms
from src.models.detector import ResNet50Detector
from src.utils.checkpointing import save_checkpoint
from src.utils.io import ensure_dir, load_yaml
from src.utils.metrics import compute_classification_metrics
from src.utils.profiling import compute_flops_params, measure_inference_time
from src.utils.reproducibility import get_torch_generator, resolve_device, seed_worker, set_seed


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_logits = []
    all_targets = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    preds = torch.argmax(logits, dim=1).numpy()
    metrics = compute_classification_metrics(targets.numpy(), preds)
    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "false_negative_rate": metrics.false_negative_rate,
    }


def train(config_path: str) -> None:
    cfg = load_yaml(config_path)
    set_seed(cfg["seed"])
    device = resolve_device(cfg["hardware"].get("device", "cuda"))
    output_dir = ensure_dir(cfg["output_dir"])

    train_tfms = get_detection_transforms(
        image_size=cfg["data"]["image_size"],
        train=True,
        mean=cfg["preprocessing"]["normalize_mean"],
        std=cfg["preprocessing"]["normalize_std"],
    )
    val_tfms = get_detection_transforms(
        image_size=cfg["data"]["image_size"],
        train=False,
        mean=cfg["preprocessing"]["normalize_mean"],
        std=cfg["preprocessing"]["normalize_std"],
    )

    clahe_cfg = cfg["preprocessing"]["clahe"]
    ds_train = BrainTumorDetectionDataset(
        root=str(Path(cfg["data"]["root"]) / "train"),
        transform=train_tfms,
        use_clahe=clahe_cfg["enabled"],
        clahe_clip_limit=clahe_cfg["clip_limit"],
        clahe_tile_grid_size=tuple(clahe_cfg["tile_grid_size"]),
        grayscale_to_rgb=cfg["data"].get("grayscale_to_rgb", True),
    )
    ds_val = BrainTumorDetectionDataset(
        root=str(Path(cfg["data"]["root"]) / "val"),
        transform=val_tfms,
        use_clahe=clahe_cfg["enabled"],
        clahe_clip_limit=clahe_cfg["clip_limit"],
        clahe_tile_grid_size=tuple(clahe_cfg["tile_grid_size"]),
        grayscale_to_rgb=cfg["data"].get("grayscale_to_rgb", True),
    )
    ds_test = BrainTumorDetectionDataset(
        root=str(Path(cfg["data"]["root"]) / "test"),
        transform=val_tfms,
        use_clahe=clahe_cfg["enabled"],
        clahe_clip_limit=clahe_cfg["clip_limit"],
        clahe_tile_grid_size=tuple(clahe_cfg["tile_grid_size"]),
        grayscale_to_rgb=cfg["data"].get("grayscale_to_rgb", True),
    )

    generator = get_torch_generator(cfg["seed"])
    loader_train = DataLoader(ds_train, batch_size=cfg["data"]["batch_size"], shuffle=True,
                              num_workers=cfg["hardware"]["num_workers"], worker_init_fn=seed_worker, generator=generator)
    loader_val = DataLoader(ds_val, batch_size=cfg["data"]["batch_size"], shuffle=False,
                            num_workers=cfg["hardware"]["num_workers"])
    loader_test = DataLoader(ds_test, batch_size=cfg["data"]["batch_size"], shuffle=False,
                             num_workers=cfg["hardware"]["num_workers"])

    model = ResNet50Detector(
        num_classes=cfg["model"]["num_classes"],
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    class_weights = None
    if cfg["training"].get("use_class_weights", False):
        y = np.array(ds_train.targets)
        weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])

    best_val_loss = float("inf")
    patience = 0

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        running_loss = 0.0
        pbar = tqdm(loader_train, desc=f"Detector Epoch {epoch + 1}/{cfg['training']['epochs']}")
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=loss.item())

        val_metrics = evaluate(model, loader_val, device)
        val_loss = val_metrics["loss"]
        print({"train_loss": running_loss / max(len(ds_train), 1), **val_metrics})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            save_checkpoint({
                "model_state": model.state_dict(),
                "config": cfg,
                "val_metrics": val_metrics,
            }, str(output_dir), cfg["save_name"])
        else:
            patience += 1
            if patience >= cfg["training"]["early_stopping_patience"]:
                print("Early stopping triggered.")
                break

    ckpt = torch.load(Path(output_dir) / cfg["save_name"], map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, loader_test, device)
    print("Detector test metrics:", test_metrics)

    sample, _ = next(iter(loader_test))
    sample = sample[:1].to(device)
    print("Detector profile:", compute_flops_params(model, sample))
    print("Detector inference time (ms):", measure_inference_time(model, sample, device))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(args.config)

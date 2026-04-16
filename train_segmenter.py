from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import BrainTumorSegmentationDataset
from src.data.transforms import get_segmentation_transforms
from src.models.attention_resunet import AttentionResUNet
from src.models.losses import FocalTverskyLoss
from src.utils.checkpointing import save_checkpoint
from src.utils.io import ensure_dir, load_yaml
from src.utils.metrics import segmentation_metrics_from_logits
from src.utils.profiling import compute_flops_params, measure_inference_time
from src.utils.reproducibility import get_torch_generator, resolve_device, seed_worker, set_seed


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    dice_scores = []
    iou_scores = []
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += loss.item() * images.size(0)
        metrics = segmentation_metrics_from_logits(logits, masks)
        dice_scores.append(metrics["dice"])
        iou_scores.append(metrics["iou"])
    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "dice": float(sum(dice_scores) / max(len(dice_scores), 1)),
        "iou": float(sum(iou_scores) / max(len(iou_scores), 1)),
    }


def train(config_path: str) -> None:
    cfg = load_yaml(config_path)
    set_seed(cfg["seed"])
    device = resolve_device(cfg["hardware"].get("device", "cuda"))
    output_dir = ensure_dir(cfg["output_dir"])

    train_tfms = get_segmentation_transforms(
        image_size=cfg["data"]["image_size"],
        train=True,
        mean=cfg["preprocessing"]["normalize_mean"],
        std=cfg["preprocessing"]["normalize_std"],
    )
    val_tfms = get_segmentation_transforms(
        image_size=cfg["data"]["image_size"],
        train=False,
        mean=cfg["preprocessing"]["normalize_mean"],
        std=cfg["preprocessing"]["normalize_std"],
    )

    clahe_cfg = cfg["preprocessing"]["clahe"]
    ds_train = BrainTumorSegmentationDataset(
        images_dir=str(Path(cfg["data"]["root"]) / "train" / "images"),
        masks_dir=str(Path(cfg["data"]["root"]) / "train" / "masks"),
        transform=train_tfms,
        use_clahe=clahe_cfg["enabled"],
        clahe_clip_limit=clahe_cfg["clip_limit"],
        clahe_tile_grid_size=tuple(clahe_cfg["tile_grid_size"]),
    )
    ds_val = BrainTumorSegmentationDataset(
        images_dir=str(Path(cfg["data"]["root"]) / "val" / "images"),
        masks_dir=str(Path(cfg["data"]["root"]) / "val" / "masks"),
        transform=val_tfms,
        use_clahe=clahe_cfg["enabled"],
        clahe_clip_limit=clahe_cfg["clip_limit"],
        clahe_tile_grid_size=tuple(clahe_cfg["tile_grid_size"]),
    )
    ds_test = BrainTumorSegmentationDataset(
        images_dir=str(Path(cfg["data"]["root"]) / "test" / "images"),
        masks_dir=str(Path(cfg["data"]["root"]) / "test" / "masks"),
        transform=val_tfms,
        use_clahe=clahe_cfg["enabled"],
        clahe_clip_limit=clahe_cfg["clip_limit"],
        clahe_tile_grid_size=tuple(clahe_cfg["tile_grid_size"]),
    )

    generator = get_torch_generator(cfg["seed"])
    loader_train = DataLoader(ds_train, batch_size=cfg["data"]["batch_size"], shuffle=True,
                              num_workers=cfg["hardware"]["num_workers"], worker_init_fn=seed_worker, generator=generator)
    loader_val = DataLoader(ds_val, batch_size=cfg["data"]["batch_size"], shuffle=False,
                            num_workers=cfg["hardware"]["num_workers"])
    loader_test = DataLoader(ds_test, batch_size=cfg["data"]["batch_size"], shuffle=False,
                             num_workers=cfg["hardware"]["num_workers"])

    model = AttentionResUNet(
        in_channels=cfg["model"]["in_channels"],
        out_channels=cfg["model"]["out_channels"],
        base_channels=cfg["model"]["base_channels"],
        use_attention=cfg["model"]["use_attention"],
    ).to(device)
    criterion = FocalTverskyLoss(
        alpha=cfg["training"]["alpha"],
        beta=cfg["training"]["beta"],
        gamma=cfg["training"]["gamma"],
    )
    optimizer = Adam(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])

    best_val_loss = float("inf")
    patience = 0

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        running_loss = 0.0
        pbar = tqdm(loader_train, desc=f"Segmenter Epoch {epoch + 1}/{cfg['training']['epochs']}")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=loss.item())

        val_metrics = evaluate(model, loader_val, criterion, device)
        val_loss = val_metrics["loss"]
        print({"train_loss": running_loss / max(len(ds_train), 1), **val_metrics})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
            save_checkpoint({"model_state": model.state_dict(), "config": cfg, "val_metrics": val_metrics},
                            str(output_dir), cfg["save_name"])
        else:
            patience += 1
            if patience >= cfg["training"]["early_stopping_patience"]:
                print("Early stopping triggered.")
                break

    ckpt = torch.load(Path(output_dir) / cfg["save_name"], map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, loader_test, criterion, device)
    print("Segmenter test metrics:", test_metrics)

    sample, _ = next(iter(loader_test))
    sample = sample[:1].to(device)
    print("Segmenter profile:", compute_flops_params(model, sample))
    print("Segmenter inference time (ms):", measure_inference_time(model, sample, device))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(args.config)

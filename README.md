# Brain-Tumor-Detection-and-Segmentation
Two-Stage Brain Tumor Detection and Segmentation
A GitHub-ready reference implementation for a two-stage MRI brain tumor analysis pipeline:
1. **Detection stage**: ResNet-50 classifier for tumor / non-tumor screening.
2. **Segmentation stage**: Attention Residual U-Net with Focal Tversky Loss.
3. **Optional feature-selection pipeline**: Global Average Pooling embeddings + SMOTE in feature space + RFE + PCA.

This repository is structured to support reproducible experiments, deterministic runs, and public release of code and trained weights.
## Repository structure
```text
brain_tumor_two_stage_repo/
├── README.md
├── requirements.txt
├── REPRODUCIBILITY_CHECKLIST.md
├── LICENSE
├── docs/
│   └── project_notes.md
├── models/
│   └── .gitkeep
├── scripts/
│   ├── train_detector.sh
│   ├── train_segmenter.sh
│   ├── run_full_pipeline.sh
│   └── extract_features.sh
├── src/
│   ├── configs/
│   │   ├── detector.yaml
│   │   ├── segmenter.yaml
│   │   └── pipeline.yaml
│   ├── data/
│   │   ├── dataset.py
│   │   └── transforms.py
│   ├── models/
│   │   ├── attention_resunet.py
│   │   ├── detector.py
│   │   └── losses.py
│   ├── training/
│   │   ├── train_detector.py
│   │   ├── train_segmenter.py
│   │   ├── evaluate.py
│   │   ├── pipeline_infer.py
│   │   └── feature_selection.py
│   ├── utils/
│   │   ├── checkpointing.py
│   │   ├── metrics.py
│   │   ├── reproducibility.py
│   │   ├── profiling.py
│   │   └── io.py
│   └── __init__.py
└── tests/
    └── smoke_test.py
```

## Supported data layout

The code expects a **2D slice-level** dataset layout. Example:

```text
data/
├── detection/
│   ├── train/
│   │   ├── tumor/
│   │   └── no_tumor/
│   ├── val/
│   │   ├── tumor/
│   │   └── no_tumor/
│   └── test/
│       ├── tumor/
│       └── no_tumor/
└── segmentation/
    ├── train/
    │   ├── images/
    │   └── masks/
    ├── val/
    │   ├── images/
    │   └── masks/
    └── test/
        ├── images/
        └── masks/
```

For patient-wise evaluation, prepare the splits before training so that slices from the same patient never appear in more than one split.

## Quick start

### 1) Create environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

### 2) Train detector

```bash
python -m src.training.train_detector --config src/configs/detector.yaml
```

### 3) Extract detector features and run optional SMOTE + RFE + PCA

```bash
python -m src.training.feature_selection --config src/configs/detector.yaml --stage train
```

### 4) Train segmenter

```bash
python -m src.training.train_segmenter --config src/configs/segmenter.yaml
```
### 5) Run full two-stage inference

```bash
python -m src.training.pipeline_infer --config src/configs/pipeline.yaml \
  --input_dir /path/to/test/images \
  --output_dir outputs/pipeline
```
## Deterministic and reproducible runs

The repository includes:
- global seed setting
- deterministic PyTorch backend configuration
- config-driven experiments
- saved metrics and checkpoints
- reproducibility checklist
- profiling hooks for FLOPs and inference time

To reproduce a run exactly, keep the same:

- code commit hash
- Python and package versions
- GPU / CUDA environment
- seed value
- dataset split files
- config YAML files

## Default hyperparameters

### Detection

- Backbone: ResNet-50
- Input size: 224 × 224
- Optimizer: Adam
- Learning rate: 1e-4
- Batch size: 16
- Epochs: 50
- Early stopping patience: 10
- Loss: Cross entropy with optional class weighting

### Segmentation

- Model: Attention Residual U-Net
- Input size: 240 × 240
- Optimizer: Adam
- Learning rate: 5e-5
- Batch size: 5
- Epochs: 100
- Early stopping patience: 10
- Loss: Focal Tversky Loss

## Notes on the manuscript-aligned pipeline

This repository matches the paper's core design but keeps a few parts modular because manuscripts often evolve during revision:

- **SMOTE** is applied in feature space, not on raw MRI images.
- **RFE + PCA** is implemented as an optional detector-side module for ablation and feature-analysis.
- **Segmentation** is trained end-to-end without external feature selection.
- **Pipeline gating** uses the detector probability threshold before sending an image to the segmenter.
## Releasing trained models

```text
models/
├── detector_best.pt
├── segmenter_best.pt
├── detector_feature_selector.joblib
└── detector_pca.joblib
```

## Citation

## License

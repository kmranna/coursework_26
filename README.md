# ML Pipeline for Urban Infrastructure Violation Verification

End-to-end machine learning system that automates the visual verification of urban infrastructure repair tasks based on before/after photographs. Built as a course paper at HSE Faculty of Computer Science, BPAD program.

The pipeline takes an Excel file with task descriptions and photo URLs as input, and produces an Excel file with `ACCEPT` / `REJECT` / `HUMAN_REVIEW` decisions for each task.

## Key results

Evaluated on a held-out test set of 6,567 manually annotated pairs (UNSEEN by any model during training):

| Metric                | Specialized (3 models) | Unified (single model) |
|-----------------------|:----------------------:|:----------------------:|
| Auto-accuracy         | 96.66 %                | **97.35 %**            |
| Critical FP rate      | 19.15 %                | **2.13 %**             |
| Human review rate     | 6.55 %                 | **5.09 %**             |

The unified single-model approach trained on a larger mixed corpus outperforms the three-model specialized ensemble on all key metrics. See the course paper for full discussion.

## Architecture

```
Input XLSX → Module 1: Category Router → Module 2: Location Verification (Siamese ResNet50)
                                       → Module 3: Quality Assessment (Siamese ResNet50)
                                       → Module 4: Decision Router → Output XLSX
```

## Requirements

```bash
pip install -r requirements.txt
```

Core dependencies: Python 3.10, PyTorch 2.x, torchvision, pandas, Pillow, matplotlib, requests, openpyxl. CUDA-capable GPU recommended (RTX 3070 or better).

For CLIP-based anomaly detection (`detect_anomalies.py`), the `transformers` library is additionally required.

## Usage

### Run the pipeline on a new Excel file

```bash
python mvp_pipeline.py input.xlsx --output decisions.xlsx --mode unified
```

Modes: `unified` (recommended) or `specialized`.

### Reproduce the evaluation

```bash
python evaluate_pipeline.py --csv pair_decisions.csv --modes specialized unified
```

This produces `evaluation_summary.txt` and `evaluation_results.csv` with per-pair predictions.

### Train from scratch

```bash
# Download photos and prepare train/val splits for location verification
python prepare_training_data.py asphalt.xlsx --sample 10000 --workers 32

# Train one location-verification model per category
python train_siamese.py --train-csv train_asphalt.csv --val-csv val_asphalt.csv \
    --epochs 10 --batch-size 32 --workers 0 --out-dir runs/asphalt

# Prepare quality-assessment splits from the location splits
python prepare_quality_data.py --train-csv train_asphalt.csv --val-csv val_asphalt.csv \
    --out-train train_qa_asphalt.csv --out-val val_qa_asphalt.csv

# Train one quality-assessment model per category
python train_quality.py --train-csv train_qa_asphalt.csv --val-csv val_qa_asphalt.csv \
    --epochs 10 --batch-size 32 --workers 0 --out-dir runs/qa_asphalt
```

## Repository structure

### Data preparation and cleaning
- `filter_pairs.py` — Tkinter GUI for manual pair annotation (produces `pair_decisions.csv`).
- `clean_dataset.py` — automated filtering of non-image URLs.
- `fix_bad_decisions.py` — automatic detection of annotation inconsistencies.
- `detect_anomalies.py` — CLIP-based detection of suspicious pairs.
- `review_suspicious.py` — second-pass GUI for reviewing flagged pairs.
- `download_rejected.py` — utility for downloading manually-rejected pairs.
- `prepare_training_data.py` — XLSX → train/val CSVs, with photo download.
- `prepare_quality_data.py` — converts location-verification splits to quality-assessment format.
- `visualize_augmentation.py` — diagnostic utility for inspecting augmented pairs.

### Model training
- `train_siamese.py` — training script for Location Verification models.
- `train_quality.py` — training script for Quality Assessment models (incl. `SafeGeometricAug` class).
- `plot_results.py` — generates training-curve and confusion-matrix figures.

### Pipeline and evaluation
- `mvp_pipeline.py` — main CLI: input XLSX → output XLSX with decisions.
- `evaluate_pipeline.py` — end-to-end evaluation comparing specialized vs unified modes on `pair_decisions.csv`, with SEEN/UNSEEN data-leakage handling.
- `evaluate_on_val.py` — alternative evaluation on the held-out validation splits.

### Data and model artifacts
- `pair_decisions.csv` — 8,000 manually annotated pairs (ground truth).
- `train_*.csv`, `val_*.csv` — Location Verification train/val splits per category.
- `train_qa_*.csv`, `val_qa_*.csv` — Quality Assessment train/val splits per category.
- `runs/asphalt/`, `runs/lawn/`, `runs/curb/` — specialized Location Verification model weights.
- `runs/siamese_full/` — unified Location Verification model weights.
- `runs/qa_asphalt/`, `runs/qa_lawn/`, `runs/qa_curb/` — Quality Assessment model weights.
- `evaluation_summary.txt`, `evaluation_results.csv` — final evaluation outputs.

Each `runs/*/` directory contains `best_model.pt` (model weights), `training_log.csv` (per-epoch metrics), and `args.json` (hyperparameters used).

### Documentation
- `violation_classification_report.pdf` — full course paper.
- `requirements.txt` — Python dependencies.

## Decision thresholds

The Decision Router uses fixed thresholds chosen to err on the side of caution:

| Condition                              | Decision         |
|----------------------------------------|------------------|
| `location_prob < 0.30`                 | REJECT           |
| `quality_prob < 0.30`                  | REJECT           |
| `location_prob > 0.95` AND `quality_prob > 0.80` | ACCEPT |
| otherwise                              | HUMAN_REVIEW     |

Thresholds can be overridden via CLI flags `--accept-loc`, `--accept-qa`, `--reject-loc`, `--reject-qa`.

## License

Code released for academic purposes. See LICENSE.

## Author

Anna Rodionova, BPAD232  
Higher School of Economics, Faculty of Computer Science  
Course paper, 2026

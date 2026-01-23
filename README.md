# CAFA 6 Protein Function Prediction

Kaggle Competition: [CAFA 6 Protein Function Prediction](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction)

## Task

Predict protein function (Gene Ontology terms) from amino acid sequences.

- **Train**: 82,404 proteins, 537,028 GO term annotations
- **Test**: 224,309 proteins
- **Unique GO terms**: 26,125 (multi-label classification)

## Deadlines

- Entry: 2026-01-26
- Final: 2026-02-02

## Submission History

| Version | Approach | # Terms | Score | Notes |
|---------|----------|---------|-------|-------|
| v1 | LightGBM (per-term) | 82 | 0.114 | >= 500 occurrences |
| v2 | v1 + GO expansion | 82 | 0.115 | Ancestor term propagation |
| v3 | NN MLP (all-at-once) | 668 | **0.165** | Best score |
| v4 | v3 + GO expansion | 668 | pending | File too large (224MB) |

## Key Learnings

### 1. Embedding Approach Works
- Used ESM-Cambrian pre-computed embeddings (320 dimensions)
- Much faster than computing embeddings ourselves
- Download: `esm-cambrian-model/protein_embeddings_*.npy`

### 2. NN >> LightGBM for Multi-label
- **LightGBM per-term**: Trains separate model for each GO term. Very slow (hours for 668 terms)
- **NN all-at-once**: Single model predicts all labels simultaneously. Fast (30 epochs in minutes)
- Score improvement: 0.114 → 0.165 (+45%)

### 3. GO Hierarchy Expansion
- If predicting a child term, also predict all ancestor terms
- Helps with evaluation metric (F-max)
- Caution: Can cause row explosion (224MB file for 224K proteins)

### 4. ID Format Gotcha
- Train embedding IDs: `sp|A0A0C5B5G6|MOTSC_HUMAN`
- train_terms.tsv IDs: `A0A0C5B5G6`
- Need to extract middle part: `id.split('|')[1]`

## Structure

```
cafa6-protein/
├── notebooks/          # Experiments
├── src/
│   ├── baseline.py     # LightGBM baseline (v1)
│   ├── baseline_v2.py  # LightGBM with more terms
│   ├── nn_baseline.py  # PyTorch MLP (best)
│   ├── parse_go.py     # GO ontology parser
│   └── expand_with_go.py # GO hierarchy expansion
├── data/               # Data (gitignored)
└── README.md
```

## Model Architecture (NN Baseline)

```python
MultiLabelMLP:
  Linear(320, 512) -> BatchNorm -> ReLU -> Dropout(0.3)
  Linear(512, 512) -> BatchNorm -> ReLU -> Dropout(0.3)
  Linear(512, num_labels)

Loss: BCEWithLogitsLoss
Optimizer: AdamW
Scheduler: CosineAnnealing
Epochs: 30
Batch: 512
```

## Setup

```bash
# Download competition data
kaggle competitions download -c cafa-6-protein-function-prediction -p data/
unzip data/cafa-6-protein-function-prediction.zip -d data/

# Download ESM-Cambrian embeddings
kaggle datasets download -d adriansanz/esm-cambrian-model -p data/embeddings/
unzip data/embeddings/esm-cambrian-model.zip -d data/embeddings/
```

## Run

```bash
# NN Baseline (recommended)
python src/nn_baseline.py

# LightGBM Baseline (slower)
python src/baseline.py

# Expand with GO hierarchy
python src/expand_with_go.py
```

## Next Steps

- [ ] Increase model capacity (deeper/wider network)
- [ ] Try attention mechanisms
- [ ] Ensemble NN + LightGBM
- [ ] Use label embeddings (dual-tower architecture)
- [ ] Tune threshold for submission

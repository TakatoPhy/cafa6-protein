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

## Current Status

**Rank: 45 / 2113 (Top 2.1%) - Silver Medal Zone**

| Date | Score | Rank | Approach |
|------|-------|------|----------|
| 2026-01-28 | **0.387** | **45** | Merged SOTA + ESM2 notebooks |
| 2026-01-28 | 0.386 | 88 | SOTA 27Jan notebook |
| 2026-01-27 | 0.375 | 371 | GOA + ProtT5 ensemble |
| 2026-01-24 | 0.208 | - | ESM2 + GO expansion |

**Total improvement: 0.114 → 0.387 (+239%)**

## Submission History

| Version | Approach | # Terms | Score | Notes |
|---------|----------|---------|-------|-------|
| v1 | LightGBM (per-term) | 82 | 0.114 | >= 500 occurrences |
| v2 | v1 + GO expansion | 82 | 0.115 | Ancestor term propagation |
| v3 | NN MLP (320 dim) | 668 | 0.165 | ESM-Cambrian embeddings |
| v4 | v3 + GO expansion | 668 | error | File too large (224MB) |
| v5 | NN MLP (1280 dim) | 1170 | 0.200 | ESM2-650M embeddings |
| v6 | v5 + GO expansion | 1170+ | 0.208 | Optimized expansion |
| v7 | Public notebook merge | - | 0.375 | GOA + ProtT5 ensemble |
| v8 | SOTA 27Jan | - | 0.386 | jakupymeraj notebook |
| **v9** | **Merged SOTA + ESM2** | - | **0.387** | **Average ensemble** |

## Key Learnings

### 1. Embedding Quality Matters Most
- **ESM-Cambrian** (320 dim): Score 0.165
- **ESM2-650M** (1280 dim): Score 0.200 (+21%)
- Higher dimension embeddings capture more protein information

### 2. NN >> LightGBM for Multi-label
- **LightGBM per-term**: Trains separate model for each GO term. Very slow (hours for 668 terms)
- **NN all-at-once**: Single model predicts all labels simultaneously. Fast (30 epochs in minutes)

### 3. GO Hierarchy Expansion
- If predicting a child term, also predict all ancestor terms
- Helps with F-max metric (+4% improvement)
- Must filter low-confidence ancestors to avoid file explosion

### 4. ID Format Gotcha
- Train embedding IDs: `sp|A0A0C5B5G6|MOTSC_HUMAN`
- train_terms.tsv IDs: `A0A0C5B5G6`
- Need to extract middle part: `id.split('|')[1]`

## Structure

```
cafa6-protein/
├── notebooks/              # Experiments
├── src/
│   ├── baseline.py         # LightGBM baseline (v1)
│   ├── baseline_v2.py      # LightGBM with more terms
│   ├── nn_baseline.py      # PyTorch MLP (ESM-Cambrian)
│   ├── nn_esm2.py          # PyTorch MLP (ESM2-650M) [BEST]
│   ├── parse_go.py         # GO ontology parser
│   ├── expand_with_go.py   # GO hierarchy expansion
│   └── expand_go_optimized.py # Optimized GO expansion
├── data/                   # Data (gitignored)
└── README.md
```

## Model Architecture (NN ESM2)

```python
MultiLabelMLP (Best: nn_esm2.py):
  Linear(1280, 1024) -> BatchNorm -> ReLU -> Dropout(0.3)
  Linear(1024, 1024) -> BatchNorm -> ReLU -> Dropout(0.3)
  Linear(1024, 512)  -> BatchNorm -> ReLU -> Dropout(0.3)
  Linear(512, num_labels)

Loss: BCEWithLogitsLoss
Optimizer: AdamW (weight_decay=0.01)
Scheduler: CosineAnnealing
Epochs: 30
Batch: 256
```

## Setup

```bash
# Download competition data
kaggle competitions download -c cafa-6-protein-function-prediction -p data/
unzip data/cafa-6-protein-function-prediction.zip -d data/

# Download ESM2-650M embeddings (recommended)
kaggle datasets download -d seddiktrk/cafa6-protein-embeddings-esm2 -p data/embeddings/esm2/ --unzip

# Or ESM-Cambrian embeddings (smaller)
kaggle datasets download -d adriansanz/esm-cambrian-model -p data/embeddings/ --unzip
```

## Run

```bash
# Best: ESM2 + GO expansion
python src/nn_esm2.py
python src/expand_go_optimized.py

# Old baseline
python src/nn_baseline.py
```

## Next Steps

- [ ] Multi-embedding ensemble (ESM2 + T5 + Ankh)
- [ ] Attention mechanisms / Transformer
- [ ] Label embeddings (dual-tower architecture)
- [ ] Hyperparameter tuning

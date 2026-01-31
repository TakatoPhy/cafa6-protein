# CAFA 6 Protein Function Prediction

Kaggle Competition: [CAFA 6 Protein Function Prediction](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction)

## Task

Predict protein function (Gene Ontology terms) from amino acid sequences.

- **Train**: 82,404 proteins, 537,028 GO term annotations
- **Test**: 224,309 proteins
- **Unique GO terms**: 26,125 (multi-label classification)

## Deadlines

- Entry: 2026-01-26
- Final: 2026-02-02 (TOMORROW!)

## Current Status

**Rank: ~61 / 2168 (Top 2.8%) - Silver Medal Zone**

**Gold Medal Target: 0.408 (Need +0.021 improvement)**

| Date | Score | Rank | Approach |
|------|-------|------|----------|
| 2026-01-28 | **0.387** | **45** | Merged SOTA + ESM2 notebooks |
| 2026-01-31 | 0.386 | ~61 | 3-model merge (no improvement) |
| 2026-01-28 | 0.386 | 88 | SOTA 27Jan notebook |
| 2026-01-27 | 0.375 | 371 | GOA + ProtT5 ensemble |
| 2026-01-24 | 0.208 | - | ESM2 + GO expansion |

**Total improvement: 0.114 → 0.387 (+239%)**

---

## Day 2 Progress (2026-01-31)

### 新規実装

#### 1. ProtBoost (LightGBM)
CAFA5 2位手法の簡易版。ESM2埋め込みをPCA圧縮してLightGBMで学習。

```
src/protboost_simple.py  - 500 GO terms, Val AUC: 0.7884
src/protboost_1000.py    - 1000 GO terms, Val AUC: ~0.78
```

設定:
- PCA: 1280 → 256次元
- LightGBM: num_leaves=31, learning_rate=0.1
- 各GO termに対してbinaryモデルを学習

#### 2. Dual-Tower Architecture
タンパク質とGO termを同じ埋め込み空間にマップして内積で予測。

```
src/dual_tower.py        - 2000 GO terms, Val Loss: 0.4254
```

アーキテクチャ:
```python
ProteinEncoder: Linear(1280→512) → LayerNorm → ReLU → Dropout(0.1)
                Linear(512→512)  → LayerNorm → ReLU → Dropout(0.1)
                Linear(512→256)
GOTermEncoder:  Embedding(2000, 256)
Prediction:     dot_product(protein_vec, go_vec) * temperature
```

出力: 448M行 (12GB) → フィルタリングで56M行に削減

#### 3. アンサンブル手法
```
src/geometric_mean.py    - 幾何平均 sqrt(s1 * s2)
src/rank_average.py      - ランクベース平均（スコア分布の差を吸収）
src/optimize_weights.py  - SOTA/ESM2の重み最適化
src/blend_dt_filtered.py - Dual-Tower用メモリ効率版ブレンド
src/blend_dt_simple.py   - pandas版ブレンド
src/blend_dual_tower.py  - ストリーミング版ブレンド
src/blend_dual_tower_streaming.py
```

### 準備済み提出ファイル（リモートサーバー）

11個の提出ファイルをリモートサーバーに準備:

| Priority | File | Method | Expected |
|----------|------|--------|----------|
| ⭐⭐⭐ | geometric_mean.tsv | 幾何平均 | 分布改善 |
| ⭐⭐⭐ | rank_average.tsv | ランク平均 | 分布正規化 |
| ⭐⭐⭐ | blend_dual_tower_30.tsv | DT 30% blend | 新モデル効果 |
| ⭐⭐ | blend_protboost1000_30.tsv | PB 30% blend | LightGBM追加 |
| ⭐⭐ | weighted_sota60_esm240.tsv | SOTA重視 | 安定選択 |
| ⭐ | weighted_sota40_esm260.tsv | ESM2重視 | - |
| ⭐ | weighted_sota70_esm230.tsv | SOTA高重み | - |
| - | protboost_500.tsv | PB単体 | ベースライン |
| - | protboost_1000.tsv | PB拡張 | ベースライン |
| - | blend_protboost500_30.tsv | PB500 blend | - |
| - | current_best.tsv | 現在最高 | 0.387保証 |

### リモートサーバー構成

```bash
# 接続
ssh taka@100.75.229.83

# ファイル場所
~/cafa6-protein/submissions/  # 11個の提出ファイル（8.3GB）
~/cafa6-protein/notebooks/    # 公開ノートブック（12GB）
~/cafa6-protein/data/embeddings/  # ESM2埋め込み（11GB）

# 提出コマンド
cd ~/cafa6-protein/submissions
kaggle competitions submit -c cafa-6-protein-function-prediction \
  -f FILE.tsv -m "MESSAGE"
```

---

## Key Learnings

### Day 2 の重要な学び

#### 1. フィルタリングは絶対にするな
**提出ファイルをフィルタリングするとスコアが激減する**
- 0.387 → 0.279 (-28%) という壊滅的な結果
- 公開ノートブックはTrain+Test両方を出力している
- Top-K制限やテストセットフィルタは適用しない
- シンプルな平均マージが最も安全

#### 2. 大規模ファイルの処理
- Dual-Tower出力: 448M行（12GB）
- メモリに載らない場合はストリーミング処理
- threshold=0.5でフィルタリング → 12.6%に削減
- baselineを先にメモリロード → DTをストリーム処理

#### 3. GPU活用
- RTX 4060 (8GB) で十分に学習可能
- Dual-Tower: 10エポック約10分
- ProtBoost: 500 terms約5分、1000 terms約15分

### 過去の学び

#### Embedding Quality Matters Most
- **ESM-Cambrian** (320 dim): Score 0.165
- **ESM2-650M** (1280 dim): Score 0.200 (+21%)
- Higher dimension embeddings capture more protein information

#### NN >> LightGBM for Multi-label
- **LightGBM per-term**: Trains separate model for each GO term. Very slow (hours for 668 terms)
- **NN all-at-once**: Single model predicts all labels simultaneously. Fast (30 epochs in minutes)

#### GO Hierarchy Expansion
- If predicting a child term, also predict all ancestor terms
- Helps with F-max metric (+4% improvement)
- Must filter low-confidence ancestors to avoid file explosion

#### ID Format Gotcha
- Train embedding IDs: `sp|A0A0C5B5G6|MOTSC_HUMAN`
- train_terms.tsv IDs: `A0A0C5B5G6`
- Need to extract middle part: `id.split('|')[1]`

---

## Day 3 Plan (2026-02-01)

### 提出スケジュール（5回/日）

| # | 時間 | 提出 | 目的 |
|---|------|------|------|
| 1 | 9:00 | geometric_mean.tsv | 幾何平均の効果確認 |
| 2 | 11:00 | rank_average.tsv | ランク平均の効果確認 |
| 3 | 14:00 | blend_dual_tower_30.tsv | Dual-Tower効果確認 |
| 4 | 17:00 | 結果に応じて最適化 | - |
| 5 | 20:00 | Final Selection | 最終選択 |

### 提出判断基準

```
geometric > 0.390 → geometric系を深掘り
rank_avg > 0.390  → rank系を深掘り
dual_tower > 0.390 → DT重みを調整
全部 < 0.387     → current_best.tsvで安全策
```

---

## Structure

```
cafa6-protein/
├── notebooks/              # Experiments
├── src/
│   ├── baseline.py         # LightGBM baseline (v1)
│   ├── baseline_v2.py      # LightGBM with more terms
│   ├── nn_baseline.py      # PyTorch MLP (ESM-Cambrian)
│   ├── nn_esm2.py          # PyTorch MLP (ESM2-650M)
│   ├── parse_go.py         # GO ontology parser
│   ├── expand_with_go.py   # GO hierarchy expansion
│   ├── expand_go_optimized.py # Optimized GO expansion
│   ├── protboost_simple.py # LightGBM 500 terms [NEW]
│   ├── protboost_1000.py   # LightGBM 1000 terms [NEW]
│   ├── dual_tower.py       # Dual-Tower Architecture [NEW]
│   ├── geometric_mean.py   # Geometric mean ensemble [NEW]
│   ├── rank_average.py     # Rank-based averaging [NEW]
│   ├── optimize_weights.py # Weight optimization [NEW]
│   ├── blend_dt_filtered.py # Memory-efficient DT blend [NEW]
│   ├── blend_dt_simple.py  # Pandas DT blend [NEW]
│   ├── blend_dual_tower.py # Streaming DT blend [NEW]
│   └── blend_dual_tower_streaming.py # Stream blend v2 [NEW]
├── data/                   # Data (gitignored, on remote)
├── submissions/            # Submissions (gitignored, on remote)
└── README.md
```

## Model Architecture

### NN ESM2 (Best single model)

```python
MultiLabelMLP:
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

### Dual-Tower (Day 2)

```python
ProteinEncoder:
  Linear(1280, 512) -> LayerNorm -> ReLU -> Dropout(0.1)
  Linear(512, 512)  -> LayerNorm -> ReLU -> Dropout(0.1)
  Linear(512, 256)  -> L2 Normalize

GOTermEncoder:
  Embedding(2000, 256) -> L2 Normalize

Prediction: dot_product * temperature
Loss: BCEWithLogitsLoss
Epochs: 10
```

## Setup

```bash
# Download competition data
kaggle competitions download -c cafa-6-protein-function-prediction -p data/
unzip data/cafa-6-protein-function-prediction.zip -d data/

# Download ESM2-650M embeddings (recommended)
kaggle datasets download -d seddiktrk/cafa6-protein-embeddings-esm2 -p data/embeddings/esm2/ --unzip
```

## Run

```bash
# Best: ESM2 + GO expansion
python src/nn_esm2.py
python src/expand_go_optimized.py

# ProtBoost
python src/protboost_simple.py

# Dual-Tower (requires GPU)
python src/dual_tower.py
```

## Remote Submission

```bash
# SSH to remote
ssh taka@100.75.229.83

# Submit
cd ~/cafa6-protein/submissions
kaggle competitions submit -c cafa-6-protein-function-prediction \
  -f geometric_mean.tsv -m "Geometric mean ensemble"
```

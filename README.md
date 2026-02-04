# CAFA 6 Protein Function Prediction

**Silver Medal** | Final Score: **0.387** | Top 2.8% (61st / 2168 teams)

[Competition Page](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction)

## Overview

[CAFA 6](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction) (Critical Assessment of Functional Annotation) challenges participants to predict protein function using Gene Ontology (GO) terms from amino acid sequences.

- **Train**: 82,404 proteins with 537,028 GO annotations
- **Test**: 224,309 proteins
- **GO terms**: 26,125 unique terms (multi-label classification)
- **Metric**: F-max (threshold-optimized F1 across all GO terms)
- **Duration**: 2026-01-24 to 2026-02-02 (10 days)

### Result

| Item | Value |
|------|-------|
| Final Score | **0.387** |
| Medal | Silver |
| Rank | 61 / 2168 (Top 2.8%) |
| Total Submissions | 22 |
| Best Approach | Arithmetic mean of two public SOTA notebooks |

## Approach Timeline

| Day | What I Tried | Score | Takeaway |
|-----|-------------|-------|----------|
| 1 | LightGBM per-term baseline | 0.114 | Too few high-confidence predictions |
| 2 | Neural network (ESM-Cambrian 320d) | 0.165 | Embedding quality matters |
| 3 | ESM2-650M embeddings + GO expansion | 0.208 | Better embeddings = better score |
| 4 | Public notebook (GOA+ProtT5) | 0.375 | Public notebooks >> custom models |
| 5 | SOTA notebook (jakupymeraj) | 0.386 | Single best public notebook |
| 6 | SOTA + ESM2 arithmetic mean | **0.387** | Simple merge of top-2 notebooks |
| 7-10 | Blending, ProtBoost, Taxon features | No improvement | Blending hurts in CAFA |

## What Worked

1. **Using public SOTA notebooks directly** - The single most impactful decision. jakupymeraj's notebook alone scored 0.386; datasciencegrad's ESM2 notebook scored 0.378.

2. **Simple arithmetic mean of two high-scoring notebooks** - Merging SOTA (0.386) + ESM2 (0.378) with equal weights yielded our best score of 0.387.

3. **Max-merge for diversity** - Taking the maximum prediction across models (SOTA + antonoof) scored 0.381. Less effective than arithmetic mean but still reasonable.

## What Didn't Work

1. **GO hierarchy propagation** - Score crashed to 0.005. CAFA's evaluation system handles GO propagation internally, so doing it yourself causes double-propagation.

2. **Blending with non-SOTA models** - Every attempt to blend in weaker models (Taxon, ProtBoost, baselines) reduced the score. CAFA's F-max metric heavily penalizes false positives, and blending dilutes high-confidence predictions.

3. **Weighted average + Top-K filtering** - Dropped to 0.279-0.291. Filtering by test set proteins or applying top-K limits removed valid predictions.

4. **Custom baselines (LightGBM, simple NN)** - All scored 0.14x. The issue: fewer than 1M predictions with score >= 0.9.

5. **ProtBoost (CAFA-5 2nd place method)** - OOM crashes during training (2001/4500 terms). When completed, only scored 0.294 standalone and hurt the ensemble when blended with SOTA.

## Key Findings

### The High-Score Count Law

The number of predictions with confidence >= 0.9 is the single strongest predictor of LB score:

| High-Score Count (>=0.9) | Expected LB |
|--------------------------|-------------|
| < 1M | 0.14x (baseline, do not submit) |
| 5-8M | 0.35-0.37 |
| 10M+ | 0.38+ |

### Blending Is a Trap

In most Kaggle competitions, ensembling improves scores. In CAFA 6, it usually hurts:

- SOTA 95% + Taxon 5%: 0.368 (worse than SOTA alone at 0.386)
- SOTA 70% + ESM2 30%: 0.368 (worse than equal-weight merge at 0.387)
- Any blend with baseline models: 0.14x

The reason: F-max penalizes false positives aggressively. Blending dilutes the sharp, high-confidence predictions from SOTA models.

### GO Propagation Is Forbidden

CAFA's evaluation handles ancestor term propagation internally. If you propagate GO terms in your submission, they get propagated again during evaluation, causing catastrophic score collapse (0.387 -> 0.005).

## Solution Architecture

```
Final Submission (0.387):
  = arithmetic_mean(SOTA_notebook, ESM2_notebook)

  SOTA_notebook (0.386):
    - jakupymeraj's public notebook
    - 53M rows, 10.5M high-score predictions

  ESM2_notebook (0.378):
    - datasciencegrad's ESM2 notebook
    - 53M rows, similar distribution
```

No custom training, no feature engineering. Just merging two well-calibrated public notebooks.

## Repository Structure

```
cafa6-protein/
├── README.md                     # This file
├── CAFA6_FINAL_REPORT.md        # Detailed post-mortem (Japanese)
├── FINDINGS.md                   # Analysis and leaderboard insights
├── SUBMISSION_HISTORY.md         # All 22 submissions with scores
├── WORK_LOG.md                   # Daily work log
├── COMPETITION_RULES.md          # Competition rules reference
│
├── src/                          # Core scripts
│   ├── baseline.py               # LightGBM per-term baseline (v1)
│   ├── nn_esm2.py                # PyTorch MLP with ESM2-650M embeddings
│   ├── nn_baseline.py            # PyTorch MLP with ESM-Cambrian embeddings
│   ├── parse_go.py               # Gene Ontology OBO parser
│   ├── expand_go_optimized.py    # GO hierarchy expansion (don't use for submission!)
│   ├── simple_merge.py           # Arithmetic mean merge
│   ├── geometric_mean.py         # Geometric mean merge
│   ├── rank_average.py           # Rank-based averaging
│   ├── blend_*.py                # Various blending strategies (all failed)
│   ├── protboost_*.py            # ProtBoost implementation (CAFA-5 2nd place)
│   ├── dual_tower.py             # Dual-tower architecture experiment
│   ├── optimize_*.py             # Weight optimization scripts
│   └── filter_submission.py      # Submission filtering (don't use!)
│
├── scripts/                      # Utilities
│   ├── validate_submission.py    # Pre-submission validation
│   ├── submission_stats.py       # Submission file statistics & LB prediction
│   ├── merge_submissions.py      # Merge multiple submission files
│   ├── safe_blend.py             # Safe blending with validation
│   ├── protboost_4500.py         # Full ProtBoost (4500 terms)
│   ├── protboost_4500_batch.py   # Batch version for memory constraints
│   ├── gcn_stacking.py           # GCN-based stacking (ProtBoost component)
│   ├── stacking_mlp.py           # MLP stacking
│   ├── prostt5_embeddings.py     # ProstT5 embedding extraction
│   └── go_max_propagation.py     # GO max propagation
│
├── notebooks/                    # Jupyter notebooks
│   ├── cafa_ensemble_378/        # Ensemble notebook (0.378)
│   ├── cafa_optimization/        # Optimization notebook
│   ├── esm2_3785/                # ESM2 notebook (0.378)
│   ├── goa_propagation/          # GOA propagation notebook
│   ├── goa_prott5_370/           # GOA+ProtT5 notebook (0.370)
│   └── external/                 # Downloaded public notebooks
│
├── kaggle_notebook/              # Kaggle submission notebook
├── experiments/                  # Experiment logs
└── cache/                        # Cached statistics
```

## How to Reproduce

### Setup

```bash
# Clone
git clone https://github.com/TakatoPhy/cafa6-protein.git
cd cafa6-protein

# Download competition data
kaggle competitions download -c cafa-6-protein-function-prediction -p data/
unzip data/cafa-6-protein-function-prediction.zip -d data/

# Download ESM2-650M embeddings
kaggle datasets download -d seddiktrk/cafa6-protein-embeddings-esm2 \
  -p data/embeddings/esm2/ --unzip
```

### Run Custom Models (for reference)

```bash
# ESM2 neural network (our best custom model, ~0.208)
python src/nn_esm2.py

# GO expansion (adds ~0.008, but DO NOT use for final submission)
python src/expand_go_optimized.py

# LightGBM baseline (~0.114)
python src/baseline.py
```

### Reproduce Best Submission

The best submission was created by downloading two public Kaggle notebooks and averaging their outputs:

```bash
# 1. Download outputs from:
#    - jakupymeraj's SOTA notebook (0.386)
#    - datasciencegrad's ESM2 notebook (0.378)
#    Place their submission.tsv files in notebooks/external/

# 2. Merge
python src/simple_merge.py \
  notebooks/external/sota_27jan_output/submission.tsv \
  notebooks/external/esm2_386_new/submission.tsv \
  -o submission.tsv

# 3. Validate before submitting
python scripts/validate_submission.py submission.tsv --category sota_blend
```

## Submission History

All 22 submissions ordered by score:

| Score | Approach | Category |
|-------|----------|----------|
| **0.387** | **SOTA + ESM2 arithmetic mean** | **sota_blend** |
| 0.386 | SOTA notebook (jakupymeraj) | single |
| 0.386 | 3-model merge (SOTA + ESM2 + CAFA378) | ensemble |
| 0.386 | SOTA + ESM2 merged (variant) | sota_blend |
| 0.381 | Max merge (SOTA + antonoof) | sota_blend |
| 0.378 | ESM2 notebook (datasciencegrad) | single |
| 0.368 | SOTA 95% + Taxon 5% | sota_blend |
| 0.368 | SOTA 70% + ESM2 30% | sota_blend |
| 0.367 | Triple blend (50/40/10) | sota_blend |
| 0.364 | 3-model rank ensemble (x2) | ensemble |
| 0.362 | antonoof notebook | single |
| 0.356 | 3-model merge (SOTA + ESM2 + KTDK) | ensemble |
| 0.291 | 5-model weighted avg top-60 | weighted |
| 0.279 | 3-model weighted avg top-60 (x2) | weighted |
| 0.141 | Baseline + GO propagation | baseline |
| 0.141 | Baseline + ProtBoost blend | baseline |
| 0.141 | Abhishek baseline | baseline |
| 0.141 | GOA + ProtT5 ensemble | baseline |
| 0.140 | Baseline + NB153 blend | baseline |

## Lessons for CAFA 7

If you're participating in CAFA 7, here's what I'd do differently:

1. **Start with public notebooks, not custom models.** I spent 3 days building custom models (0.114 -> 0.208) when I could have started with public notebooks at 0.37+. Survey the landscape first.

2. **Do NOT propagate GO terms in your submission.** CAFA's evaluation handles this. Adding propagation yourself will destroy your score.

3. **Do NOT blend aggressively.** F-max heavily penalizes false positives. Simple arithmetic mean of 2 similarly-performing models is the ceiling. Adding weaker models always hurts.

4. **Check the high-score count (predictions >= 0.9).** If it's below 1M, your submission will score 0.14x regardless. This is the single best pre-submission sanity check.

5. **Use the validation script.** `scripts/validate_submission.py` categorizes your submission and predicts the score range. Use it before every submission.

6. **Don't filter submissions by test proteins.** Public notebooks output predictions for both train and test sets. Filtering to test-only causes a -28% score drop (0.387 -> 0.279).

7. **ProtBoost (CAFA-5 2nd place) needs serious compute.** Minimum 4.5 days on 1 GPU for 4500 terms. Plan accordingly if you want to implement it.

## References

- [CAFA 6 Competition](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction)
- [CAFA-5 2nd Place: ProtBoost](https://arxiv.org/abs/2412.04529) - GCN stacking with 29 features
- [jakupymeraj's SOTA Notebook](https://www.kaggle.com/code/jakupymeraj) - 0.386 LB
- [datasciencegrad's ESM2 Notebook](https://www.kaggle.com/code/datasciencegrad) - 0.378 LB
- [100 Experiments: What Works & Doesn't](https://www.kaggle.com/code/ravishah1/cafa-6-100-experiments-what-works-doesn-t) - Comprehensive experiment log

## License

MIT License. See [LICENSE](LICENSE).

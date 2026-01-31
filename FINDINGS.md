# CAFA6 Competition Findings

## Current Status (2026-02-01)

| Item | Value |
|------|-------|
| Best Score | 0.387 (45位) |
| Target | 0.400+ |
| Deadline | 2026-02-02 23:59 UTC |
| Remaining Submissions | 10 (5/day) |

## Leaderboard Analysis

| Rank | Team | Score |
|------|------|-------|
| 1 | Mixture of Experts | 0.472 |
| 2 | WePredictProteins | 0.441 |
| 3 | Guoliang&Born4 | 0.440 |
| 20 | yukiZ | 0.404 |

Gap to 0.40: **+0.013 needed**

## Available Resources

### Downloaded Notebooks

| Name | Score | Rows | Path |
|------|-------|------|------|
| sota_27jan | 0.386 | 53M | notebooks/external/sota_27jan_output/ |
| esm2_386_new | 0.386 | 53M | notebooks/external/esm2_386_new/ |
| cafa_ensemble_378 | 0.378 | 55.5M | notebooks/external/cafa_ensemble_378/ |
| goa_propagation | ? | 54.6M | notebooks/external/goa_propagation/ |
| cafa_tuning | ? | 55.5M | notebooks/external/cafa_tuning/ |

### Prepared Submission Files

| File | Size | Content |
|------|------|---------|
| merged_sota_esm2_386_new.tsv.gz | 287M | SOTA + ESM2 new arithmetic mean |
| geometric_mean_new.tsv.gz | 291M | SOTA + ESM2 new geometric mean |
| 4model_merge.tsv.gz | 388M | 4-model arithmetic mean |
| 5model_merge.tsv.gz | 393M | 5-model arithmetic mean |
| 5model_geometric.tsv.gz | 471M | 5-model geometric mean |

## Key Insights from Previous Experiments

### What Works
- GOA baseline: 0.336
- GOA 60% + ProtT5 40%: 0.374
- ESM-2 Embedding: 0.378

### What DOES NOT Work (CRITICAL)
- **GO propagation**: Crashed to 0.005 - CAFA evaluation handles this internally
- **Adding novel predictions**: Always hurts - F-max penalizes false positives heavily
- **Filtering low-confidence**: Worse (0.323 vs 0.336)
- **GOA as 1/3 features**: Overfits (0.318)
- **Evidence weighting**: Breaks calibration (0.234)

## CAFA-5 Winner Analysis

### 1st Place: GOCurator
- Text mining + literature retrieval
- Protein language models
- Structural information
- GORetrieval (GO-protein matching)

### 2nd Place: ProtBoost (0.582)
Source: https://arxiv.org/abs/2412.04529

**29 Features for GCN:**
- 5 base models x 4 variants = 20 features
  - Logit prediction
  - Two GO hierarchy propagation variants
  - Prior flag (absent in training)
- Electronic GO annotations = 1 feature
- Trainable GO embeddings = 8 features

**Base Models:**
- Py-Boost (4500 targets, ~2h/fold)
- Logistic Regression (GPU, 2-10h/fold)
- 2-layer NN (1h/fold)

**Key Insight:**
- GOA as 1/29 feature = balanced
- GOA as 1/3 feature = overfits (our failure)

**Training Time:**
- Minimum: 4.5 days (1 GPU)
- Optimal: 2 days (3 GPUs)

## External Data Allowed

Competition rules Section 2.6 confirms external data is allowed if:
- Publicly available
- Accessible to all participants
- Free or reasonable cost

**Usable External Data:**
- UniProt / UniProtKB
- Gene Ontology (GO hierarchy)
- QuickGO (electronic annotations)
- InterPro (domain info)
- STRING (protein interactions)

## Strategy for Remaining Submissions

### Day 3 (2/1) - Exploration
1. merged_sota_esm2_386_new - baseline with new ESM2
2. 5model_merge - maximum diversity
3. 5model_geometric - geometric mean effect
4. Based on results, adjust
5. Best candidate refinement

### Day 4 (2/2) - Final
6. Check for new notebooks
7. Develop Day 3 best method
8. Try simplified ProtBoost if time
9. Final tuning
10. Final Selection

## Realistic Expectations

| Scenario | Score | Probability |
|----------|-------|-------------|
| 5-model ensemble improvement | 0.388-0.390 | High |
| New high-score notebook appears | 0.390-0.395 | Medium |
| Simplified ProtBoost works | 0.395-0.400 | Low |
| 0.40+ breakthrough | 0.400+ | Very Low |

**Bottom line:** 
- 0.40 requires either new SOTA notebooks or ProtBoost-level implementation
- Focus on defending position and incremental improvement
- Do not break current 0.387 with risky experiments

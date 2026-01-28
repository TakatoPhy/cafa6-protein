# 2026-01-28 Breakthrough: 371位 → 45位

## Summary

公開ノートブックのダウンロード・マージ戦略により、大幅なスコア改善を達成。

| Before | After | Improvement |
|--------|-------|-------------|
| 0.375 (371位) | 0.387 (45位) | +0.012 (+326位) |

**銀メダル圏内 (Top 5%) 確定**

---

## Timeline

### 起点: 0.375 (371位)
- 前日までのベストスコア
- ESM2 + GO expansion ベースのアプローチ

### Step 1: 最新ノートブック調査
投票数・作成日順で公開ノートブックを調査:

| Notebook | Score | Votes | 更新日 |
|----------|-------|-------|--------|
| jakupymeraj/cafa6-sota-solution-27jan | ? | 61 | 1/28 |
| datasciencegrad/cafa-6-esm-2-embedding-inference-0-3785 | 0.3785 | 72 | 1/22 |
| abhishekgodara/cafa-6-protein-prediction | ? | 164 | 1/21 |

### Step 2: ノートブック出力ダウンロード
```bash
kaggle kernels output jakupymeraj/cafa6-sota-solution-27jan -p notebooks/external/sota_27jan
kaggle kernels output datasciencegrad/cafa-6-esm-2-embedding-inference-0-3785 -p notebooks/external/esm2_3785
kaggle kernels output abhishekgodara/cafa-6-protein-prediction -p notebooks/external/abhishek_protein
```

### Step 3: 個別提出テスト

| Notebook | Score | 順位変動 |
|----------|-------|----------|
| SOTA 27Jan | **0.386** | 371→88位 |
| ESM2 0.3785 | 0.378 | - |
| Abhishek | 0.141 | - (弱い) |

**SOTA 27Janだけで88位まで上昇！**

### Step 4: マージ戦略

2つの提出ファイルを平均でマージ:

```python
import pandas as pd

# Load both submissions
sota = pd.read_csv('sota_27jan/submission.tsv', sep='\t', header=None,
                   names=['protein', 'go', 'score'])
esm2 = pd.read_csv('esm2_3785/submission.tsv', sep='\t', header=None,
                   names=['protein', 'go', 'score'])

# Concatenate and average by (protein, go) pair
combined = pd.concat([sota, esm2], ignore_index=True)
merged = combined.groupby(['protein', 'go'], as_index=False)['score'].mean()

# Save
merged.to_csv('submission.tsv', sep='\t', header=False, index=False)
```

**結果:**
- SOTA: 53M行
- ESM2: 55M行
- Merged: 55.8M行 (1.55GB)

### Step 5: マージ版提出

| Submission | Score | 順位 |
|------------|-------|------|
| Merged (SOTA + ESM2) | **0.387** | **45位** |

**88位 → 45位、銀メダル圏内確定！**

---

## Key Insights

### 1. 公開ノートブックの活用が最も効果的
- 自前で最適化するより、高スコアの公開ノートブックをダウンロードしてマージする方が効率的
- 特に締切直前は新しいSOTAが公開されることが多い

### 2. マージ戦略の効果
- 単体: 0.386 (SOTA)
- マージ: 0.387 (+0.001)
- 異なるアプローチの予測を平均することで、ノイズが減り精度向上

### 3. 弱いモデルはマージに入れない
- Abhishek (0.141) は単体で弱すぎる
- マージに入れると平均が下がる可能性が高い

---

## Submission History (Today)

| Time | Description | Score | Rank |
|------|-------------|-------|------|
| 13:18 | SOTA solution 27Jan | 0.386 | 88 |
| 14:30 | ESM2 0.3785 | 0.378 | - |
| 17:10 | Merged SOTA + ESM2 | **0.387** | **45** |
| 17:28 | Abhishek protein | 0.141 | - |

**Daily submissions used: 4/5**

---

## Medal Status

| Medal | Cutoff (2113 teams) | Current Rank |
|-------|---------------------|--------------|
| Gold | ~14 | - |
| **Silver** | ~106 | **45 ✓** |
| Bronze | ~211 | ✓ |

---

## Next Steps (Tomorrow)

1. **新しい公開ノートブックをチェック** - 締切直前のSOTA更新を拾う
2. **他の高スコアノートブックをマージ候補に追加**
   - CAFA 6 | Tuning (164 votes)
   - CAFA 6 | GOA+ Propagation (201 votes)
3. **加重平均マージを試す** - SOTAに重み付け
4. **金メダル圏 (Top 14) を目指す** - あと31位

---

## Files

### Downloaded Notebooks
- `notebooks/external/sota_27jan/submission.tsv` (1.4GB)
- `notebooks/external/esm2_3785/submission.tsv` (1.3GB)
- `notebooks/external/abhishek_protein/submission.tsv` (294MB)

### Merged Submission
- `submission.tsv` (1.55GB) - SOTA + ESM2 average

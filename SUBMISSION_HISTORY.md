# CAFA6 全提出履歴

## 提出一覧

| 説明 | 実際LB |
|------|--------|
| SOTA+ESM2 merged - secure 0.387 baseline | 0.386 |
| Max merge SOTA+antonoof (12.8M high scores, no Taxon) | 0.381 |
| SOTA*0.95 + Taxon*0.05 blend | 0.368 |
| baseline + GO max propagation | 0.141 |
| baseline*0.8 + protboost_1000*0.2 (safe blend) | 0.141 |
| baseline*0.7 + NB153*0.3 (safe blend) | 0.140 |
| 3-model rank ensemble + GO propagation | 0.364 |
| 3-model rank ensemble + GO propagation | 0.364 |
| 3-model merge: SOTA + ESM2 3785 + KTDK | 0.356 |
| 3-model merge: SOTA + ESM2 3785 + CAFA 378 | 0.386 |
| Merged SOTA 27Jan + ESM2 3785 | 0.386 |
| 5-model ensemble (weighted avg, top-60) | 0.291 |
| 3-model ensemble (weighted avg, top-60) | 0.279 |
| 3-model ensemble (weighted avg, top-60) | 0.279 |
| cafa-optimization (antonoof) direct | 0.362 |
| Abhishek protein prediction | 0.141 |
| Merged SOTA 27Jan + ESM2 0.3785 (average) | **0.387** |
| ESM2 0.3785 (datasciencegrad) | 0.378 |
| SOTA solution 27Jan (jakupymeraj) | 0.386 |
| GOA+ProtT5 ensemble with GO propagation | 0.141 |

## スコア分布

| LBスコア | 件数 | 
|----------|------|
| 0.387 | 1 (ベスト) |
| 0.386 | 4 |
| 0.381 | 1 |
| 0.378 | 1 |
| 0.368 | 1 |
| 0.364 | 2 |
| 0.362 | 1 |
| 0.356 | 1 |
| 0.291 | 1 |
| 0.279 | 2 |
| 0.141 | 4 |
| 0.140 | 1 |

## パターン分析

**0.38x以上 (成功):**
- SOTA単体
- SOTA+ESM2マージ
- Max merge

**0.35-0.37 (中程度):**
- rank ensemble
- antonoof単体
- 3-model merge (一部)

**0.14x (失敗):**
- baseline系全て
- GO propagation使用
- GOA+ProtT5 ensemble

## 教訓

1. baseline系は全滅 (0.14x)
2. GO propagationを使うと壊れる
3. SOTA単体またはSOTA+ESM2が最強
4. weighted avg, top-60は悪化する

---

## 検証スクリプト v3

パス: scripts/validate_v3.py

使い方: python3 scripts/validate_v3.py file.tsv --category カテゴリ

カテゴリ別予測:
- baseline: 0.140-0.141 (提出禁止)
- weighted: 0.279-0.291 (低スコア警告)
- ensemble: 0.356-0.364 (中程度)
- single: 0.362-0.386 (良好)
- sota_blend: 0.368-0.387 (最良)

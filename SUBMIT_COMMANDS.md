# CAFA6 提出コマンド

## 提出枠リセット
- **2/1 9:00 JST** (UTC 00:00) に5回リセット

## リモートからKaggle提出

```bash
# リモートサーバーに接続
ssh taka@100.75.229.83

# Kaggle認証確認
kaggle competitions list

# 提出コマンド (リモートで実行)
cd ~/cafa6-protein/submissions

# 1. 幾何平均 (新手法)
kaggle competitions submit -c cafa-6-protein-function-prediction \
  -f geometric_mean.tsv -m "Geometric mean: SOTA + ESM2"

# 2. ランク平均 (新手法)
kaggle competitions submit -c cafa-6-protein-function-prediction \
  -f rank_average.tsv -m "Rank average: SOTA + ESM2"

# 3. 加重平均 SOTA重視
kaggle competitions submit -c cafa-6-protein-function-prediction \
  -f weighted_sota60_esm240.tsv -m "Weighted: SOTA 60% + ESM2 40%"

# 4. ProtBoost 1000 ブレンド (完成後)
kaggle competitions submit -c cafa-6-protein-function-prediction \
  -f blend_protboost1000_30.tsv -m "Blend: ProtBoost1000 30% + SOTA+ESM2 70%"

# 5. 結果を見て調整
```

## ProtBoost 1000 完成後のブレンド作成

```bash
ssh taka@100.75.229.83
cd ~/cafa6-protein
source ~/ml-env/bin/activate
python src/blend_protboost_1000.py 0.3
```

## スコア確認

```bash
kaggle competitions submissions -c cafa-6-protein-function-prediction | head -10
```

## 現在のベストスコア
- **0.387** (SOTA + ESM2 単純平均)

## 金メダルライン
- **0.407** (15位付近)

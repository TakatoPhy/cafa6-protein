# CAFA6 0.40+ Execution Guide

## 現在のステータス (2026-02-01 更新)

| ファイル | サイズ | 状態 | 期待スコア |
|---------|--------|------|-----------|
| current_best (submission.tsv) | 280 MB | ✅ 利用可能 | 0.387 (baseline) |
| cafa_opt | 1.3 GB | ✅ 利用可能 | - |
| goa | 1.6 GB | ✅ 利用可能 | - |
| rank_ensemble | 1.5 GB | ✅ 生成済み | 0.388-0.392 |
| rank_ensemble_propagated | 1.5 GB (296 MB gz) | ✅ 生成済み | 0.390-0.395 |
| final_ensemble | 1.5 GB (401 MB gz) | ✅ 生成済み | 0.389-0.394 |

## 提出ファイル (submissions/)

```
rank_ensemble_propagated.tsv.gz  - 296 MB  ⭐推奨
final_ensemble.tsv.gz            - 401 MB
```

## 48時間タイムライン

### Phase 1: 即効性アンサンブル (0-8h)

```bash
# 1. 5モデルランクアンサンブル
python scripts/rank_ensemble_5model.py submissions/rank_ensemble.tsv

# 2. GO階層Max伝播
python scripts/go_max_propagation.py submission.tsv submissions/propagated.tsv

# 3. 提出
kaggle competitions submit -c cafa-6-protein-function-prediction \
  -f submissions/rank_ensemble.tsv.gz -m "3-model rank ensemble"
```

### Phase 2: ProtBoost 4500 (8-24h)

```bash
# リモートサーバーで実行 (GPU必要)
ssh taka@100.75.229.83
cd ~/cafa6-protein

# ProtBoost 4500 terms版
python scripts/protboost_4500.py --embedding esm2_650M

# ブレンド
python scripts/final_ensemble.py --models current_best,protboost_4500 --weights 0.7,0.3
```

### Phase 3: GCNスタッキング (24-40h)

```bash
# 訓練
python scripts/gcn_stacking.py --train --top-k 5000

# 予測
python scripts/gcn_stacking.py --predict

# または、MLPスタッカー（軽量版）
python scripts/stacking_mlp.py --train
python scripts/stacking_mlp.py --predict
```

### Phase 4: 最終調整 (40-48h)

```bash
# 全モデル統合
python scripts/final_ensemble.py --propagate

# 圧縮
gzip -c submissions/final_ensemble.tsv > submissions/final_ensemble.tsv.gz

# 最終提出
kaggle competitions submit -c cafa-6-protein-function-prediction \
  -f submissions/final_ensemble.tsv.gz -m "Final: all models + GO propagation"
```

## スクリプト一覧

| スクリプト | 用途 | 入力 | 出力 |
|-----------|------|------|------|
| rank_ensemble_5model.py | ランク正規化アンサンブル | 複数.tsv | rank_ensemble.tsv |
| go_max_propagation.py | GO階層伝播 | .tsv | propagated.tsv |
| protboost_4500.py | ProtBoost拡張版 | embeddings | protboost_4500.tsv |
| stacking_mlp.py | MLPスタッキング | 複数.tsv | stacking_mlp.tsv |
| gcn_stacking.py | GCNスタッキング | 複数.tsv + GO graph | gcn_stacking.tsv |
| final_ensemble.py | 最終アンサンブル | 複数.tsv | final_ensemble.tsv |

## 提出コマンド

```bash
# リモートサーバーへファイル転送
scp submissions/final_ensemble.tsv.gz taka@100.75.229.83:~/cafa6-protein/submissions/

# リモートで提出
ssh taka@100.75.229.83 "cd ~/cafa6-protein/submissions && \
  kaggle competitions submit -c cafa-6-protein-function-prediction \
  -f final_ensemble.tsv.gz -m 'Final ensemble'"
```

## 期待スコア

| モデル | 期待スコア | 備考 |
|--------|-----------|------|
| current_best | 0.387 | ベースライン |
| + rank_ensemble | 0.389-0.392 | +0.002-0.005 |
| + GO propagation | 0.390-0.393 | +0.001-0.003 |
| + ProtBoost 4500 | 0.392-0.398 | +0.002-0.008 |
| + GCN stacking | 0.395-0.405 | +0.003-0.010 |

## トラブルシューティング

### メモリ不足
```bash
# ProtBoostのterms数を減らす
python scripts/protboost_4500.py --terms 3000

# GCNのtop-kを減らす
python scripts/gcn_stacking.py --top-k 3000
```

### GPU利用
```bash
# ProtBoostでGPU使用
python scripts/protboost_4500.py --gpu
```

### ファイルが大きすぎる
```bash
# 提出前に圧縮
gzip -c submission.tsv > submission.tsv.gz
```

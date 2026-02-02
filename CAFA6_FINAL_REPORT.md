# CAFA6 Protein Function Prediction - 最終報告書

**期間:** 2026-01-24 〜 2026-02-02
**最終スコア:** 0.387 (LB)
**目標:** 0.40+（未達成）

---

## 最終結果

| 項目 | 値 |
|------|-----|
| ベストスコア | **0.387** |
| ベスト提出 | SOTA+ESM2 merged (arithmetic mean) |
| 総提出数 | 22回 |
| 成功率 | 77% (17/22が0.14x以上) |

---

## 全提出履歴

### 0.38x以上（成功）

| 提出 | 構成 | LB |
|------|------|-----|
| SOTA+ESM2 merged | SOTA + ESM2 arithmetic mean | **0.387** |
| SOTA単体 | jakupymeraj notebook | 0.386 |
| 3-model merge (SOTA+ESM2+CAFA) | 3モデル平均 | 0.386 |
| Merged SOTA+ESM2 | 別配合 | 0.386 |
| Max merge SOTA+antonoof | max取り | 0.381 |
| ESM2単体 | datasciencegrad | 0.378 |

### 0.35-0.37（中程度）

| 提出 | 構成 | LB |
|------|------|-----|
| SOTA*0.95 + Taxon*0.05 | Taxonブレンド | 0.368 |
| weighted_sota70_esm230 | SOTA 70% + ESM2 30% | 0.368 |
| triple_50_40_10 | SOTA 50% + antonoof 40% + Taxon 10% | 0.367 |
| 3-model rank ensemble | ランクアンサンブル | 0.364 |
| antonoof単体 | 公開notebook | 0.362 |
| 3-model merge | SOTA+ESM2+KTDK | 0.356 |

### 0.27-0.29（低スコア）

| 提出 | 構成 | LB |
|------|------|-----|
| 5-model weighted avg top-60 | weighted平均 | 0.291 |
| 3-model weighted avg top-60 | weighted平均 | 0.279 |

### 0.14x（失敗）

| 提出 | 構成 | LB |
|------|------|-----|
| baseline + GO propagation | GO伝播 | 0.141 |
| baseline*0.8 + protboost*0.2 | ブレンド | 0.141 |
| Abhishek baseline | baseline | 0.141 |
| GOA+ProtT5 ensemble | アンサンブル | 0.141 |
| baseline*0.7 + NB153*0.3 | ブレンド | 0.140 |

---

## 試した手法と結果

### 成功した手法

1. **公開SOTA Notebook直接使用**
   - jakupymeraj: 0.386
   - datasciencegrad ESM2: 0.378
   - 教訓: 公開notebookの直接使用が最も効率的

2. **SOTAベースの軽量マージ**
   - SOTA + ESM2 arithmetic mean: 0.387
   - max merge: 0.381
   - 教訓: 似たスコア分布のモデル同士のマージは有効

### 失敗した手法

1. **GO階層伝播**
   - LBが0.005まで崩壊
   - 原因: CAFA評価が内部で伝播を処理するため二重伝播になる

2. **ブレンド比率の調整**
   - SOTA比率を下げるほど悪化
   - SOTA 95%: 0.368, SOTA 70%: 0.368, SOTA 50%: 0.367
   - 教訓: SOTAを薄めることに意味はない

3. **weighted average + top-N選択**
   - 0.279-0.291に低下
   - 原因: false positiveが増加しF-maxが低下

4. **baseline系の使用**
   - 全て0.14x
   - 原因: 高スコア(>=0.9)の予測数が少なすぎる

5. **ProtBoost 4500ターム**
   - OOMで2回失敗（2001/4500でクラッシュ）
   - 完了しても0.294で単体使用不可
   - SOTAとブレンドしても悪化

6. **Taxon features追加**
   - 期待した改善なし
   - SOTAとの相関が低い(0.146)のは良いが、予測精度が低い

---

## 得られた知見

### 重要な発見

1. **高スコア(>=0.9)の数が決定的**
   - 1M未満 → 0.14x確定
   - 5M以上 → 0.35+期待可能
   - 10M以上 → 0.38+期待可能

2. **ブレンドは基本的に悪化する**
   - CAFA6のF-max指標はfalse positiveに敏感
   - 異なる分布のモデルを混ぜると高スコア予測が薄まる
   - SOTAを薄めるほど悪化

3. **GO伝播は禁止**
   - CAFA評価システムが内部で処理
   - 自前で伝播すると二重になりスコア崩壊

4. **公開notebookの直接使用が最強**
   - 自作モデルより公開SOTAの方が高スコア
   - 時間を自作に使うより、良いnotebookを探す方が効率的

### 予測精度

カテゴリベースの予測が最も正確:

| カテゴリ | 予測範囲 | 的中率 |
|---------|---------|--------|
| baseline | 0.140-0.141 | 100% |
| weighted | 0.279-0.291 | 100% |
| ensemble | 0.356-0.364 | 100% |
| sota_blend | 0.368-0.387 | 100% |

回帰式（LB = -0.622*mean + 0.580）は外れることが多い。

---

## ファイル構成

```
cafa6-protein/
  README.md                    # プロジェクト概要
  CAFA6_FINAL_REPORT.md       # この報告書
  SUBMISSION_HISTORY.md        # 提出履歴
  FINDINGS.md                  # 発見・分析
  WORK_LOG.md                  # 作業ログ
  COMPETITION_RULES.md         # コンペルール

  src/                         # メインスクリプト
    baseline.py                # ベースライン
    blend_*.py                 # ブレンドスクリプト
    geometric_mean.py          # 幾何平均
    rank_average.py            # ランク平均
    protboost_*.py             # ProtBoost実装

  scripts/                     # ユーティリティ
    validate_v3.py             # 検証スクリプト（カテゴリ版）
    safe_blend.py              # 安全なブレンド
    max_merge.py               # max合成

  submissions/                 # 提出ファイル
    merged_sota_esm2_386_new.tsv  # ベスト (0.387)
    triple_50_40_10.tsv        # 最終日1
    weighted_sota70_esm230.tsv # 最終日2

  notebooks/                   # Jupyter notebooks
    external/                  # ダウンロードした公開notebook

  data/                        # データ
    embeddings/                # ESM2等の埋め込み

  models/                      # 訓練済みモデル
```

---

## 反省点と次回への教訓

### やるべきだったこと

1. **早期に公開notebookを網羅的に調査**
   - 自作に時間を使いすぎた
   - 最初から高スコアnotebookを探すべきだった

2. **検証スクリプトを最初から整備**
   - 予測式を早期に確立していれば無駄な提出を避けられた

3. **ProtBoostの実装を早期に着手**
   - CAFA5で2位の手法だが、時間不足で完成しなかった
   - 締切1週間前には着手すべき

### やってはいけなかったこと

1. **GO伝播の実装**
   - 完全に時間の無駄だった

2. **baseline系の提出**
   - 5回も0.14xを出した
   - 高スコア数を確認してから提出すべきだった

3. **ブレンド比率の探索**
   - SOTA 95%, 92%, 91%, 90%と試したが全て悪化
   - 最初の1回で「ブレンド=悪化」と学ぶべきだった

---

## 技術的メモ

### 提出時の注意

1. **gzip圧縮で400エラーが出ることがある**
   - 原因不明
   - 非圧縮(.tsv)なら通る

2. **ファイル名は何でもOK**
   - submission.tsvでなくても提出可能

3. **Kaggle CLIのsubmissions確認がバグっている**
   - TypeError: got an unexpected keyword argument 'page_number'
   - Web UIで確認する必要あり

### 検証スクリプトの使い方

```bash
# カテゴリベース検証
python3 scripts/validate_v3.py file.tsv --category sota_blend

# カテゴリ:
# - baseline: 0.140-0.141 (提出禁止)
# - weighted: 0.279-0.291 (低スコア警告)
# - ensemble: 0.356-0.364 (中程度)
# - single: 0.362-0.386 (良好)
# - sota_blend: 0.368-0.387 (最良)
```

---

## 結論

**最終スコア 0.387** で終了。目標の0.40には届かなかった。

主な敗因:
1. ブレンドで改善できるという誤った仮説に時間を費やした
2. ProtBoost実装が間に合わなかった
3. 公開notebookの調査が遅かった

次回参加時は、公開リソースの調査を最優先し、自作モデルは時間があれば程度に留める。

---

*生成日: 2026-02-02*
*Co-Authored-By: Claude Opus 4.5*

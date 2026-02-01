#!/usr/bin/env python3
"""
提出前検証スクリプト v2

このスクリプトを通過しない限り、提出してはいけない。

0.140事件の教訓:
- スコア平均が同じでも、分布が変わるとF-maxが激減する
- 高スコア予測(>=0.9)の数が重要
- スコア分布のヒストグラムを詳細にチェックする必要がある

Usage:
    python scripts/validate_submission.py new_submission.tsv

Checks:
    1. フォーマット検証
    2. 行数チェック（ベースラインと比較）
    3. タンパク質数チェック
    4. スコア分布チェック（平均・標準偏差）
    5. 高スコア予測数チェック（重要！）
    6. スコア分布ヒストグラム詳細比較
    7. (protein, term)ペア完全一致確認
    8. 同一予測のスコア比較
"""
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

BASE_DIR = Path(__file__).parent.parent
BASELINE_PATH = BASE_DIR / 'submission.tsv'

# 許容範囲（厳格化）
MAX_ROW_RATIO = 1.5          # 行数が1.5倍以上は警告
MAX_PROTEIN_RATIO = 1.2      # タンパク質数が1.2倍以上は警告
SCORE_DIFF_THRESHOLD = 0.10  # 同一予測のスコア差が0.10以上は警告

# 高スコア予測の許容減少率（重要！）
HIGH_SCORE_THRESHOLDS = {
    0.9: 0.20,  # >=0.9の予測が20%以上減少したらエラー
    0.7: 0.15,  # >=0.7の予測が15%以上減少したらエラー
    0.5: 0.20,  # >=0.5の予測が20%以上減少したらエラー
}

# スコア分布ヒストグラムの許容変化率
HISTOGRAM_CHANGE_THRESHOLD = 0.30  # 各区間で30%以上変化したらエラー


def load_submission(path: Path) -> dict:
    """Load submission and return stats."""
    scores = defaultdict(dict)
    row_count = 0
    score_values = []
    pairs = set()

    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            try:
                protein, go_term, score = parts[0], parts[1], float(parts[2])
                if not go_term.startswith('GO:'):
                    continue
                scores[protein][go_term] = score
                score_values.append(score)
                pairs.add((protein, go_term))
                row_count += 1
            except ValueError:
                continue

    return {
        'scores': scores,
        'row_count': row_count,
        'protein_count': len(scores),
        'score_values': score_values,
        'pairs': pairs,
    }


def validate(new_path: Path) -> tuple[bool, list[str]]:
    """
    Validate new submission against baseline.

    Returns:
        (passed, messages)
    """
    errors = []
    warnings = []

    print(f"\n{'='*60}")
    print(f"提出前検証 v2: {new_path.name}")
    print(f"{'='*60}\n")

    # 1. ファイル存在チェック
    if not new_path.exists():
        errors.append(f"ファイルが存在しない: {new_path}")
        return False, errors

    if not BASELINE_PATH.exists():
        errors.append(f"ベースラインが存在しない: {BASELINE_PATH}")
        return False, errors

    # 2. ロード
    print("Loading baseline...")
    baseline = load_submission(BASELINE_PATH)
    print(f"  行数: {baseline['row_count']:,}")
    print(f"  タンパク質数: {baseline['protein_count']:,}")

    print("\nLoading new submission...")
    new = load_submission(new_path)
    print(f"  行数: {new['row_count']:,}")
    print(f"  タンパク質数: {new['protein_count']:,}")

    # 3. 行数チェック
    print("\n--- 行数チェック ---")
    row_ratio = new['row_count'] / baseline['row_count']
    print(f"比率: {row_ratio:.2f}x")
    if row_ratio > 2.0 or row_ratio < 0.5:
        errors.append(f"行数がベースラインと大きく異なる (比率: {row_ratio:.2f}x)")
    elif row_ratio > MAX_ROW_RATIO or row_ratio < 0.7:
        warnings.append(f"行数がベースラインと異なる (比率: {row_ratio:.2f}x)")
    else:
        print("OK")

    # 4. タンパク質数チェック
    print("\n--- タンパク質数チェック ---")
    protein_ratio = new['protein_count'] / baseline['protein_count']
    print(f"比率: {protein_ratio:.2f}x")
    if protein_ratio > 1.5 or protein_ratio < 0.7:
        errors.append(f"タンパク質数がベースラインと大きく異なる (比率: {protein_ratio:.2f}x)")
    elif protein_ratio > MAX_PROTEIN_RATIO or protein_ratio < 0.9:
        warnings.append(f"タンパク質数がベースラインと異なる (比率: {protein_ratio:.2f}x)")
    else:
        print("OK")

    # 5. (protein, term)ペア完全一致確認
    print("\n--- (protein, term)ペア一致確認 ---")
    missing_pairs = baseline['pairs'] - new['pairs']
    extra_pairs = new['pairs'] - baseline['pairs']
    print(f"  baselineにあって新規にない: {len(missing_pairs):,}")
    print(f"  新規にあってbaselineにない: {len(extra_pairs):,}")

    if len(missing_pairs) > 0:
        errors.append(f"baselineの予測が{len(missing_pairs):,}件欠落している")
    if len(extra_pairs) > baseline['row_count'] * 0.1:
        warnings.append(f"baselineにない予測が{len(extra_pairs):,}件追加されている")
    if len(missing_pairs) == 0 and len(extra_pairs) == 0:
        print("OK (完全一致)")

    # 6. スコア分布チェック（平均・標準偏差）
    print("\n--- スコア分布チェック ---")
    baseline_scores = np.array(baseline['score_values'])
    new_scores = np.array(new['score_values'])

    baseline_mean = np.mean(baseline_scores)
    new_mean = np.mean(new_scores)
    baseline_std = np.std(baseline_scores)
    new_std = np.std(new_scores)

    print(f"ベースライン: mean={baseline_mean:.3f}, std={baseline_std:.3f}")
    print(f"新規:         mean={new_mean:.3f}, std={new_std:.3f}")

    if abs(new_mean - baseline_mean) > 0.15:
        errors.append(f"スコア平均がベースラインと大きく異なる ({baseline_mean:.3f} → {new_mean:.3f})")
    elif abs(new_mean - baseline_mean) > 0.05:
        warnings.append(f"スコア平均がベースラインと異なる ({baseline_mean:.3f} → {new_mean:.3f})")
    else:
        print("OK")

    # 7. 高スコア予測数チェック（重要！）
    print("\n--- 高スコア予測数チェック（重要）---")
    high_score_ok = True
    for threshold, max_decrease in HIGH_SCORE_THRESHOLDS.items():
        baseline_count = np.sum(baseline_scores >= threshold)
        new_count = np.sum(new_scores >= threshold)

        if baseline_count > 0:
            change_rate = (new_count - baseline_count) / baseline_count
            status = "OK" if abs(change_rate) <= max_decrease else "❌"
            print(f"  >={threshold}: {baseline_count:,} → {new_count:,} ({change_rate:+.1%}) {status}")

            if change_rate < -max_decrease:
                errors.append(f"高スコア予測(>={threshold})が{-change_rate:.1%}減少 (許容: {max_decrease:.0%})")
                high_score_ok = False
            elif change_rate > max_decrease:
                warnings.append(f"高スコア予測(>={threshold})が{change_rate:.1%}増加")

    if high_score_ok:
        print("  全体: OK")

    # 8. スコア分布ヒストグラム詳細比較
    print("\n--- スコア分布ヒストグラム ---")
    bins = [0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    baseline_hist, _ = np.histogram(baseline_scores, bins=bins)
    new_hist, _ = np.histogram(new_scores, bins=bins)

    histogram_ok = True
    for i in range(len(bins) - 1):
        b_count = baseline_hist[i]
        n_count = new_hist[i]
        if b_count > 0:
            change_rate = (n_count - b_count) / b_count
            status = "" if abs(change_rate) <= HISTOGRAM_CHANGE_THRESHOLD else "⚠️"
            print(f"  [{bins[i]:.1f}-{bins[i+1]:.1f}): {b_count:>8,} → {n_count:>8,} ({change_rate:+6.1%}) {status}")

            if abs(change_rate) > 0.5:  # 50%以上変化はエラー
                errors.append(f"スコア区間[{bins[i]:.1f}-{bins[i+1]:.1f})が{change_rate:+.1%}変化")
                histogram_ok = False
            elif abs(change_rate) > HISTOGRAM_CHANGE_THRESHOLD:
                warnings.append(f"スコア区間[{bins[i]:.1f}-{bins[i+1]:.1f})が{change_rate:+.1%}変化")

    # 9. 同一予測のスコア比較
    print("\n--- 同一予測のスコア比較 ---")
    sample_proteins = list(baseline['scores'].keys())[:10]
    large_diffs = []
    score_decreases = 0
    score_increases = 0

    for protein in sample_proteins:
        if protein not in new['scores']:
            continue
        for term in list(baseline['scores'][protein].keys())[:5]:
            if term not in new['scores'][protein]:
                continue
            old_score = baseline['scores'][protein][term]
            new_score = new['scores'][protein][term]
            diff = new_score - old_score

            if diff < -0.01:
                score_decreases += 1
            elif diff > 0.01:
                score_increases += 1

            if abs(diff) > 0.05:  # 表示は0.05以上の差のみ
                print(f"  {protein[:12]}... {term}: {old_score:.2f} → {new_score:.2f} ({diff:+.2f})")

            if abs(diff) > SCORE_DIFF_THRESHOLD:
                large_diffs.append((protein, term, old_score, new_score))

    print(f"\n  サンプル内: 減少={score_decreases}, 増加={score_increases}")

    if len(large_diffs) > 5:
        warnings.append(f"{len(large_diffs)}件の予測でスコアが大きく変化 (>{SCORE_DIFF_THRESHOLD})")

    if score_decreases > score_increases * 3:
        warnings.append(f"スコア減少が増加より大幅に多い ({score_decreases} vs {score_increases})")

    # 10. 結果表示
    print(f"\n{'='*60}")
    print("検証結果")
    print(f"{'='*60}")

    if errors:
        print("\n❌ エラー (提出禁止):")
        for e in errors:
            print(f"  - {e}")

    if warnings:
        print("\n⚠️ 警告 (要確認):")
        for w in warnings:
            print(f"  - {w}")

    passed = len(errors) == 0

    if passed and not warnings:
        print("\n✅ 全チェック通過")
    elif passed:
        print("\n⚠️ 警告あり。提出前に確認してください。")
    else:
        print("\n❌ 検証失敗。提出しないでください。")

    print(f"{'='*60}\n")

    return passed, errors + warnings


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/validate_submission.py new_submission.tsv")
        print("\nこのスクリプトを通過しない限り、提出してはいけない。")
        sys.exit(1)

    new_path = Path(sys.argv[1])
    passed, messages = validate(new_path)

    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()

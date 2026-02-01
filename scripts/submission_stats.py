#!/usr/bin/env python3
"""
提出ファイルの統計情報をキャッシュ & LBスコア予測スクリプト

回帰式: LB = -0.622 * mean + 0.580 (R²=0.996)

Usage:
    # LBスコアを予測
    python scripts/submission_stats.py predict file.tsv

    # 統計を計算してキャッシュ
    python scripts/submission_stats.py cache file1.tsv file2.tsv ...

    # キャッシュされた統計を比較
    python scripts/submission_stats.py compare baseline.tsv new.tsv

    # 全キャッシュを表示（予測LB付き）
    python scripts/submission_stats.py list
"""
import sys
import json
import hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np

CACHE_DIR = Path(__file__).parent.parent / 'cache' / 'submission_stats'


def compute_stats(path: Path) -> dict:
    """ファイルを読み込んで統計情報を計算"""
    print(f"Computing stats for {path.name}...")

    scores = []
    proteins = set()
    terms = set()
    row_count = 0

    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i % 1000000 == 0 and i > 0:
                print(f"  {i:,} rows processed...")

            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            try:
                protein, go_term, score = parts[0], parts[1], float(parts[2])
                if not go_term.startswith('GO:'):
                    continue
                scores.append(score)
                proteins.add(protein)
                terms.add(go_term)
                row_count += 1
            except ValueError:
                continue

    scores = np.array(scores)

    # ヒストグラム
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(scores, bins=bins)

    stats = {
        'file': str(path),
        'file_size_mb': path.stat().st_size / 1024 / 1024,
        'row_count': row_count,
        'protein_count': len(proteins),
        'term_count': len(terms),
        'score_mean': float(np.mean(scores)),
        'score_std': float(np.std(scores)),
        'score_min': float(np.min(scores)),
        'score_max': float(np.max(scores)),
        'high_score_counts': {
            '>=0.9': int(np.sum(scores >= 0.9)),
            '>=0.8': int(np.sum(scores >= 0.8)),
            '>=0.7': int(np.sum(scores >= 0.7)),
            '>=0.5': int(np.sum(scores >= 0.5)),
        },
        'histogram': {f'{bins[i]:.1f}-{bins[i+1]:.1f}': int(hist[i]) for i in range(len(hist))},
    }

    print(f"  Done: {row_count:,} rows, {len(proteins):,} proteins")
    return stats


def get_cache_path(file_path: Path) -> Path:
    """ファイルパスからキャッシュパスを生成"""
    # ファイル名とサイズでハッシュを作成
    key = f"{file_path.name}_{file_path.stat().st_size}"
    hash_str = hashlib.md5(key.encode()).hexdigest()[:8]
    return CACHE_DIR / f"{file_path.stem}_{hash_str}.json"


def cache_stats(paths: list[Path]):
    """統計情報を計算してキャッシュ"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for path in paths:
        if not path.exists():
            print(f"File not found: {path}")
            continue

        cache_path = get_cache_path(path)
        if cache_path.exists():
            print(f"Already cached: {path.name} -> {cache_path.name}")
            continue

        stats = compute_stats(path)
        with open(cache_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Cached: {cache_path.name}")


def load_stats(path: Path) -> dict:
    """キャッシュから統計を読み込む（なければ計算）"""
    cache_path = get_cache_path(path)

    if cache_path.exists():
        with open(cache_path, 'r') as f:
            return json.load(f)

    # キャッシュがなければ計算
    stats = compute_stats(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(stats, f, indent=2)
    return stats


def compare_stats(baseline_path: Path, new_path: Path):
    """2つのファイルの統計を比較"""
    print(f"\nLoading stats...")
    baseline = load_stats(baseline_path)
    new = load_stats(new_path)

    print(f"\n{'='*60}")
    print(f"比較: {baseline_path.name} vs {new_path.name}")
    print(f"{'='*60}\n")

    # 基本統計
    print("--- 基本統計 ---")
    print(f"{'項目':<20} {'Baseline':>15} {'New':>15} {'変化':>10}")
    print(f"{'-'*60}")

    for key in ['row_count', 'protein_count', 'term_count']:
        b_val = baseline[key]
        n_val = new[key]
        ratio = n_val / b_val if b_val > 0 else 0
        print(f"{key:<20} {b_val:>15,} {n_val:>15,} {ratio:>9.2f}x")

    print(f"\n--- スコア分布 ---")
    print(f"{'項目':<20} {'Baseline':>15} {'New':>15} {'変化':>10}")
    print(f"{'-'*60}")
    print(f"{'mean':<20} {baseline['score_mean']:>15.4f} {new['score_mean']:>15.4f} {new['score_mean']-baseline['score_mean']:>+10.4f}")
    print(f"{'std':<20} {baseline['score_std']:>15.4f} {new['score_std']:>15.4f} {new['score_std']-baseline['score_std']:>+10.4f}")

    print(f"\n--- 高スコア予測数 ---")
    print(f"{'閾値':<20} {'Baseline':>15} {'New':>15} {'変化率':>10}")
    print(f"{'-'*60}")
    for threshold in ['>=0.9', '>=0.8', '>=0.7', '>=0.5']:
        b_val = baseline['high_score_counts'][threshold]
        n_val = new['high_score_counts'][threshold]
        change = (n_val - b_val) / b_val * 100 if b_val > 0 else 0
        print(f"{threshold:<20} {b_val:>15,} {n_val:>15,} {change:>+9.1f}%")

    print(f"\n--- ヒストグラム ---")
    print(f"{'区間':<20} {'Baseline':>15} {'New':>15} {'変化率':>10}")
    print(f"{'-'*60}")
    for bin_range in baseline['histogram']:
        b_val = baseline['histogram'][bin_range]
        n_val = new['histogram'].get(bin_range, 0)
        change = (n_val - b_val) / b_val * 100 if b_val > 0 else 0
        print(f"{bin_range:<20} {b_val:>15,} {n_val:>15,} {change:>+9.1f}%")


def predict_lb(stats: dict) -> float:
    """統計情報からLBスコアを予測（R²=0.996の回帰式）"""
    # LB = -0.6221 * mean + 0.5798
    return -0.6221 * stats['score_mean'] + 0.5798


def list_cache():
    """キャッシュされた統計を一覧表示"""
    if not CACHE_DIR.exists():
        print("No cache found")
        return

    print(f"\n{'='*80}")
    print("キャッシュ一覧")
    print(f"{'='*80}\n")
    print(f"{'ファイル':<30} {'行数':>12} {'mean':>8} {'>=0.9':>10} {'予測LB':>8}")
    print(f"{'-'*80}")

    for cache_file in sorted(CACHE_DIR.glob('*.json')):
        with open(cache_file, 'r') as f:
            stats = json.load(f)
        name = Path(stats['file']).name[:28]
        pred_lb = predict_lb(stats)
        print(f"{name:<30} {stats['row_count']:>12,} {stats['score_mean']:>8.3f} {stats['high_score_counts']['>=0.9']:>10,} {pred_lb:>8.3f}")


def predict_file(path: Path):
    """ファイルのLBスコアを予測"""
    stats = load_stats(path)
    pred_lb = predict_lb(stats)

    print(f"\n{'='*60}")
    print(f"LBスコア予測: {path.name}")
    print(f"{'='*60}\n")

    print(f"統計情報:")
    print(f"  行数:       {stats['row_count']:,}")
    print(f"  タンパク質: {stats['protein_count']:,}")
    print(f"  GO term数:  {stats['term_count']:,}")
    print(f"  スコア平均: {stats['score_mean']:.4f}")
    print(f"  スコア標準偏差: {stats['score_std']:.4f}")
    print(f"  高スコア予測 (>=0.9): {stats['high_score_counts']['>=0.9']:,}")
    print(f"  高スコア予測 (>=0.7): {stats['high_score_counts']['>=0.7']:,}")

    print(f"\n{'='*60}")
    print(f"予測LBスコア: {pred_lb:.3f}")
    print(f"{'='*60}")
    print(f"(回帰式: LB = -0.622 * mean + 0.580, R²=0.996)")

    # 信頼区間的な警告
    if stats['score_mean'] < 0.2 or stats['score_mean'] > 0.75:
        print(f"\n⚠️ 警告: mean={stats['score_mean']:.3f} は訓練データの範囲外")
        print(f"   訓練データ範囲: mean ∈ [0.326, 0.706]")

    return pred_lb


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == 'cache':
        paths = [Path(p) for p in sys.argv[2:]]
        cache_stats(paths)
    elif cmd == 'compare':
        if len(sys.argv) < 4:
            print("Usage: python submission_stats.py compare baseline.tsv new.tsv")
            sys.exit(1)
        compare_stats(Path(sys.argv[2]), Path(sys.argv[3]))
    elif cmd == 'list':
        list_cache()
    elif cmd == 'predict':
        if len(sys.argv) < 3:
            print("Usage: python submission_stats.py predict file.tsv")
            sys.exit(1)
        predict_file(Path(sys.argv[2]))
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == '__main__':
    main()

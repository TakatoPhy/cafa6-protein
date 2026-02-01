#!/usr/bin/env python3
"""
最新高スコアノートブックを即座に取り込むスクリプト

Usage:
    # 高スコアノートブック一覧を表示
    python scripts/fetch_latest_notebook.py list

    # ノートブック出力をダウンロード
    python scripts/fetch_latest_notebook.py fetch <kernel_ref>

    # ダウンロード → 予測LB確認 まで一括
    python scripts/fetch_latest_notebook.py quick <kernel_ref>

Examples:
    python scripts/fetch_latest_notebook.py list
    python scripts/fetch_latest_notebook.py quick jakupymeraj/cafa6-new-sota
"""
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / 'data' / 'notebooks_output'
COMPETITION = 'cafa-6-protein-function-prediction'


def run_cmd(cmd: list, capture=True) -> str:
    """Run command and return output."""
    result = subprocess.run(cmd, capture_output=capture, text=True)
    if result.returncode != 0 and capture:
        print(f"Error: {result.stderr}", file=sys.stderr)
    return result.stdout if capture else ""


def list_notebooks(sort_by='voteCount', limit=15):
    """List top notebooks for the competition."""
    print(f"\n{'='*80}")
    print(f"CAFA6 ノートブック一覧 (by {sort_by}, top {limit})")
    print(f"{'='*80}\n")

    cmd = [
        'kaggle', 'kernels', 'list',
        '--competition', COMPETITION,
        '--sort-by', sort_by,
        '--page-size', str(limit),
    ]
    output = run_cmd(cmd)

    # Display raw output (formatted by kaggle CLI)
    print(output)

    # Also show by date
    print(f"\n{'='*80}")
    print(f"最新ノートブック (by dateRun)")
    print(f"{'='*80}\n")

    cmd_date = [
        'kaggle', 'kernels', 'list',
        '--competition', COMPETITION,
        '--sort-by', 'dateRun',
        '--page-size', '10',
    ]
    output_date = run_cmd(cmd_date)
    print(output_date)

    print(f"\n💡 使い方:")
    print(f"   python scripts/fetch_latest_notebook.py quick <kernel_ref>")
    print(f"\n📌 既知の高スコア:")
    print(f"   jakupymeraj/cafa6-sota-solution-27jan          (0.386)")
    print(f"   datasciencegrad/cafa-6-esm-2-embedding-inference-0-386 (0.386)")
    print(f"   antonoof/cafa-ensemble-0-378                   (0.378)")


def fetch_notebook(kernel_ref: str) -> Path:
    """Download notebook output."""
    # Create output directory
    safe_name = kernel_ref.replace('/', '_')
    output_path = OUTPUT_DIR / safe_name
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"ダウンロード中: {kernel_ref}")
    print(f"{'='*60}\n")

    cmd = [
        'kaggle', 'kernels', 'output',
        kernel_ref,
        '-p', str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Find submission file
    submission_files = list(output_path.glob('*.tsv')) + list(output_path.glob('*.csv'))

    if submission_files:
        print(f"\n✅ ダウンロード完了: {submission_files[0]}")
        return submission_files[0]
    else:
        print(f"\n❌ 提出ファイルが見つからない")
        print(f"   ダウンロード先: {output_path}")
        return None


def predict_lb(file_path: Path) -> float:
    """Run LB prediction on file."""
    print(f"\n{'='*60}")
    print(f"LB予測中: {file_path.name}")
    print(f"{'='*60}\n")

    cmd = [
        'python3', str(BASE_DIR / 'scripts' / 'submission_stats.py'),
        'predict', str(file_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)

    # Extract predicted LB from output
    for line in result.stdout.split('\n'):
        if '予測LBスコア:' in line:
            try:
                lb = float(line.split(':')[1].strip())
                return lb
            except:
                pass
    return None


def quick_fetch(kernel_ref: str):
    """Fetch notebook and predict LB in one step."""
    print(f"\n{'#'*60}")
    print(f"# クイック取り込み: {kernel_ref}")
    print(f"{'#'*60}")

    # Step 1: Download
    file_path = fetch_notebook(kernel_ref)
    if not file_path:
        return

    # Step 2: Predict LB
    pred_lb = predict_lb(file_path)

    # Step 3: Summary
    print(f"\n{'='*60}")
    print(f"サマリー")
    print(f"{'='*60}")
    print(f"  ノートブック: {kernel_ref}")
    print(f"  ファイル: {file_path}")
    print(f"  予測LB: {pred_lb:.3f}" if pred_lb else "  予測LB: 計算失敗")

    if pred_lb and pred_lb >= 0.38:
        print(f"\n🎉 高スコア！マージ候補です")
        print(f"   マージコマンド例:")
        print(f"   python scripts/merge_submissions.py {file_path} <other.tsv> -o merged.tsv")
    elif pred_lb and pred_lb >= 0.35:
        print(f"\n⚠️ そこそこのスコア。単体提出 or マージで使えるかも")
    else:
        print(f"\n❌ 低スコア。使わない方が良い")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == 'list':
        list_notebooks()
    elif cmd == 'fetch':
        if len(sys.argv) < 3:
            print("Usage: fetch <kernel_ref>")
            sys.exit(1)
        fetch_notebook(sys.argv[2])
    elif cmd == 'quick':
        if len(sys.argv) < 3:
            print("Usage: quick <kernel_ref>")
            sys.exit(1)
        quick_fetch(sys.argv[2])
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == '__main__':
    main()

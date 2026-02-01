"""
ProtBoost 4500 terms - Batch Processing Version

Memory-efficient version that processes terms in batches:
1. Train 500 terms
2. Generate predictions for all test proteins
3. Append to output file
4. Free memory
5. Next batch

This avoids OOM by not keeping all 4500 models in memory.

Usage:
    python scripts/protboost_4500_batch.py [--gpu]
    python scripts/protboost_4500_batch.py --resume 1000  # Resume from batch starting at term 1000
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import lightgbm as lgb
import warnings
import argparse
import pickle
import gc
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
EMBEDDINGS_DIR = DATA_DIR / 'embeddings' / 'multi'
OUTPUT_DIR = BASE_DIR / 'submissions'
MODELS_DIR = BASE_DIR / 'models'

# Aspect configuration
ASPECT_CONFIG = {
    'P': 3000,  # Biological Process
    'F': 1000,  # Molecular Function
    'C': 500,   # Cellular Component
}

# Model configuration
PCA_DIM = 256
SEED = 42
BATCH_SIZE = 500  # Process 500 terms at a time


def load_embeddings(name: str):
    """Load embeddings and IDs."""
    emb_dir = EMBEDDINGS_DIR / name
    train_emb = np.load(emb_dir / 'train_sequences_emb.npy')
    train_ids = np.load(emb_dir / 'train_sequences_ids.npy', allow_pickle=True)
    test_emb = np.load(emb_dir / 'testsuperset_emb.npy')
    test_ids = np.load(emb_dir / 'testsuperset_ids.npy', allow_pickle=True)

    # Extract UniProt IDs
    train_ids = np.array([pid.split('|')[1] if '|' in pid else pid for pid in train_ids])
    test_ids = np.array([pid.split('|')[1] if '|' in pid else pid for pid in test_ids])

    return train_emb, train_ids, test_emb, test_ids


def load_train_terms():
    """Load training GO term annotations."""
    terms_path = DATA_DIR / 'Train' / 'train_terms.tsv'
    df = pd.read_csv(terms_path, sep='\t', skiprows=1, header=None, names=['protein', 'go_term', 'aspect'])
    return df


def get_top_terms_by_aspect(train_terms: pd.DataFrame, config: dict) -> list:
    """Get top N terms for each aspect."""
    all_terms = []
    aspect_names = {'P': 'Biological Process', 'F': 'Molecular Function', 'C': 'Cellular Component'}

    for aspect, n_terms in config.items():
        aspect_terms = train_terms[train_terms['aspect'] == aspect]
        counts = aspect_terms['go_term'].value_counts()
        actual_n = min(n_terms, len(counts))
        top_n = counts.head(actual_n).index.tolist()
        all_terms.extend(top_n)
        if len(top_n) > 0:
            min_count = counts.iloc[len(top_n) - 1]
            print(f"  {aspect_names.get(aspect, aspect)}: {len(top_n)} terms (min count: {min_count:,})")

    return all_terms


def train_and_predict_batch(
    batch_terms: list,
    batch_start_idx: int,
    X_train: np.ndarray,
    X_val: np.ndarray,
    Y_train: np.ndarray,
    Y_val: np.ndarray,
    X_test_pca: np.ndarray,
    test_ids: np.ndarray,
    Y_full: np.ndarray,
    lgb_params: dict,
    output_file,
) -> tuple[list, int]:
    """Train models for a batch of terms and generate predictions."""

    batch_aucs = []
    row_count = 0
    n_batch = len(batch_terms)

    print(f"\n  --- Batch {batch_start_idx}-{batch_start_idx + n_batch - 1} ---")

    # Train all models in this batch
    models = []
    for i, term in enumerate(batch_terms):
        term_idx = batch_start_idx + i

        if i % 100 == 0:
            print(f"    Training term {term_idx + 1}...", flush=True)

        y_train = Y_train[:, term_idx]
        y_val = Y_val[:, term_idx]

        if y_train.sum() < 5:
            models.append(None)
            batch_aucs.append(0)
            continue

        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=100,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )

        models.append(model)

        from sklearn.metrics import roc_auc_score
        try:
            y_pred = model.predict(X_val)
            auc = roc_auc_score(y_val, y_pred)
        except:
            auc = 0
        batch_aucs.append(auc)

    mean_auc = np.mean([a for a in batch_aucs if a > 0]) if batch_aucs else 0
    print(f"    Batch AUC: {mean_auc:.4f}")

    # Generate predictions for all test proteins - VECTORIZED (much faster)
    print(f"    Generating predictions for {len(test_ids):,} proteins...", flush=True)

    for j, (term, model) in enumerate(zip(batch_terms, models)):
        term_idx = batch_start_idx + j

        if j % 100 == 0:
            print(f"      Term {j+1}/{len(batch_terms)}...", flush=True)

        if model is None:
            scores = np.full(len(test_ids), Y_full[:, term_idx].mean())
        else:
            scores = model.predict(X_test_pca)  # Predict all proteins at once

        # Output only scores >= 0.01
        mask = scores >= 0.01
        for i in np.where(mask)[0]:
            output_file.write(f"{test_ids[i]}\t{term}\t{scores[i]:.6f}\n")
            row_count += 1

    # Free memory
    del models
    gc.collect()

    print(f"    Wrote {row_count:,} rows")

    return batch_aucs, row_count


def main():
    parser = argparse.ArgumentParser(description='ProtBoost 4500 terms (Batch Processing)')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--embedding', type=str, default='esm2_650M',
                        help='Embedding name')
    parser.add_argument('--resume', type=int, default=0,
                        help='Resume from term index (e.g., 1000 to resume from term 1000)')
    parser.add_argument('--batch-size', type=int, default=500,
                        help='Number of terms per batch')
    args = parser.parse_args()

    print("=== ProtBoost 4500 (Batch Processing) ===\n")
    print(f"Batch size: {args.batch_size} terms")

    # Load embeddings
    print(f"\nLoading {args.embedding} embeddings...")
    try:
        train_emb, train_ids, test_emb, test_ids = load_embeddings(args.embedding)
        print(f"  Train: {train_emb.shape}, Test: {test_emb.shape}")
    except Exception as e:
        print(f"  Error loading embeddings: {e}")
        return

    # Load training labels
    print("\nLoading training labels...")
    train_terms = load_train_terms()
    print(f"  Total annotations: {len(train_terms):,}")

    # Get top terms by aspect
    print("\nSelecting top terms by aspect:")
    top_terms = get_top_terms_by_aspect(train_terms, ASPECT_CONFIG)
    n_terms = len(top_terms)
    print(f"\nTotal: {n_terms} terms")

    # Create protein-to-index mapping
    train_id_to_idx = {pid: i for i, pid in enumerate(train_ids)}

    # Create label matrix
    print("\nCreating label matrix...")
    proteins_in_train = set(train_id_to_idx.keys())
    train_terms_filtered = train_terms[
        (train_terms['protein'].isin(proteins_in_train)) &
        (train_terms['go_term'].isin(top_terms))
    ]

    train_proteins = train_terms_filtered['protein'].unique()
    protein_to_row = {p: i for i, p in enumerate(train_proteins)}
    term_to_col = {t: i for i, t in enumerate(top_terms)}

    n_proteins = len(train_proteins)
    print(f"  Proteins: {n_proteins:,}, Terms: {n_terms}")

    Y = np.zeros((n_proteins, n_terms), dtype=np.float32)
    for _, row in train_terms_filtered.iterrows():
        p_idx = protein_to_row[row['protein']]
        t_idx = term_to_col[row['go_term']]
        Y[p_idx, t_idx] = 1.0

    print(f"  Label matrix: {Y.shape}, density: {Y.mean():.4f}")

    # Prepare embeddings
    print("\nPreparing training embeddings...")
    X = np.array([train_emb[train_id_to_idx[p]] for p in train_proteins])
    print(f"  X shape: {X.shape}")

    # Apply PCA
    print(f"\nApplying PCA (dim={PCA_DIM})...")
    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    X_pca = pca.fit_transform(X)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    X_test_pca = pca.transform(test_emb)
    print(f"  X_test_pca shape: {X_test_pca.shape}")

    # Free original embeddings
    del train_emb, test_emb
    gc.collect()

    # Split for validation
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_pca, Y, test_size=0.1, random_state=SEED
    )
    print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}")

    # Configure LightGBM
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': SEED,
        'n_jobs': -1,
    }

    if args.gpu:
        lgb_params['device'] = 'gpu'
        lgb_params['gpu_use_dp'] = False
        print("\nUsing GPU for training")

    # Setup output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f'protboost_4500_{args.embedding}.tsv'
    checkpoint_path = MODELS_DIR / 'checkpoints' / 'protboost_batch_progress.txt'
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # Check resume
    start_batch = args.resume
    if start_batch > 0:
        print(f"\nResuming from term {start_batch}")
        mode = 'a'
    else:
        mode = 'w'

    # Process in batches
    print(f"\n{'='*60}")
    print(f"Processing {n_terms} terms in batches of {args.batch_size}")
    print(f"{'='*60}")

    all_aucs = []
    total_rows = 0

    with open(output_path, mode) as f:
        for batch_start in range(start_batch, n_terms, args.batch_size):
            batch_end = min(batch_start + args.batch_size, n_terms)
            batch_terms = top_terms[batch_start:batch_end]

            batch_aucs, row_count = train_and_predict_batch(
                batch_terms=batch_terms,
                batch_start_idx=batch_start,
                X_train=X_train,
                X_val=X_val,
                Y_train=Y_train,
                Y_val=Y_val,
                X_test_pca=X_test_pca,
                test_ids=test_ids,
                Y_full=Y,
                lgb_params=lgb_params,
                output_file=f,
            )

            all_aucs.extend(batch_aucs)
            total_rows += row_count

            # Save checkpoint
            with open(checkpoint_path, 'w') as cp:
                cp.write(str(batch_end))
            print(f"  ✓ Checkpoint: {batch_end}/{n_terms} terms complete")

            # Force flush
            f.flush()

    # Summary
    mean_auc = np.mean([a for a in all_aucs if a > 0])
    print(f"\n{'='*60}")
    print(f"=== Done ===")
    print(f"{'='*60}")
    print(f"Mean validation AUC: {mean_auc:.4f}")
    print(f"Total rows: {total_rows:,}")
    print(f"Output: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Cleanup checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()


if __name__ == '__main__':
    main()

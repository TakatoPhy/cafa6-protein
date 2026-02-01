"""
ProtBoost with Multiple Embeddings.

Combines ESM2-650M + ProtT5-xl embeddings for better representation.
Based on CAFA5 2nd place approach.

Usage:
    python scripts/protboost_multi_emb.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import lightgbm as lgb
import warnings
import argparse
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
EMBEDDINGS_DIR = DATA_DIR / 'embeddings' / 'multi'
OUTPUT_DIR = BASE_DIR / 'submissions'

# Configuration
N_TOP_TERMS = 2000
PCA_DIM = 512  # Higher dim for combined embeddings
SEED = 42


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
    df = pd.read_csv(terms_path, sep='\t', header=None, names=['protein', 'go_term', 'aspect'])
    return df


def main():
    parser = argparse.ArgumentParser(description='ProtBoost Multi-Embedding')
    parser.add_argument('--embeddings', type=str, default='esm2_650M,protT5_xl',
                        help='Comma-separated embedding names')
    parser.add_argument('--n-terms', type=int, default=N_TOP_TERMS,
                        help='Number of top GO terms')
    args = parser.parse_args()

    embedding_names = [e.strip() for e in args.embeddings.split(',')]
    print(f"=== ProtBoost Multi-Embedding ===\n")
    print(f"Embeddings: {embedding_names}")

    # Load multiple embeddings
    all_train_emb = []
    all_test_emb = []
    train_ids = None
    test_ids = None

    for emb_name in embedding_names:
        print(f"\nLoading {emb_name}...")
        try:
            t_emb, t_ids, te_emb, te_ids = load_embeddings(emb_name)
            print(f"  Train: {t_emb.shape}, Test: {te_emb.shape}")

            all_train_emb.append(t_emb)
            all_test_emb.append(te_emb)

            if train_ids is None:
                train_ids = t_ids
                test_ids = te_ids
            else:
                # Verify IDs match
                assert np.array_equal(train_ids, t_ids), f"Train IDs mismatch for {emb_name}"
                assert np.array_equal(test_ids, te_ids), f"Test IDs mismatch for {emb_name}"

        except Exception as e:
            print(f"  Error: {e}")
            return

    # Concatenate embeddings
    print(f"\nConcatenating {len(all_train_emb)} embeddings...")
    train_emb = np.concatenate(all_train_emb, axis=1)
    test_emb = np.concatenate(all_test_emb, axis=1)
    print(f"  Combined: Train {train_emb.shape}, Test {test_emb.shape}")

    # Load training labels
    print("\nLoading training labels...")
    train_terms = load_train_terms()
    print(f"  Total annotations: {len(train_terms):,}")

    # Get top terms
    term_counts = train_terms['go_term'].value_counts()
    top_terms = term_counts.head(args.n_terms).index.tolist()
    print(f"  Top {args.n_terms} GO terms")

    # Create mappings
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
    n_terms = len(top_terms)
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
    pca = PCA(n_components=min(PCA_DIM, X.shape[1], X.shape[0] - 1), random_state=SEED)
    X_pca = pca.fit_transform(X)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    X_test_pca = pca.transform(test_emb)
    print(f"  X_test_pca shape: {X_test_pca.shape}")

    # Split for validation
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_pca, Y, test_size=0.1, random_state=SEED
    )
    print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}")

    # Train LightGBM
    print(f"\nTraining LightGBM for {n_terms} terms...")

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

    models = []
    val_aucs = []

    for i, term in enumerate(top_terms):
        if i % 200 == 0:
            print(f"  Training term {i+1}/{n_terms}...", flush=True)

        y_train = Y_train[:, i]
        y_val = Y_val[:, i]

        if y_train.sum() < 5:
            models.append(None)
            val_aucs.append(0)
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

        y_pred = model.predict(X_val)
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y_val, y_pred)
        except:
            auc = 0
        val_aucs.append(auc)

    mean_auc = np.mean([a for a in val_aucs if a > 0])
    print(f"\nMean validation AUC: {mean_auc:.4f}")

    # Generate predictions
    print("\nGenerating test predictions...")
    test_predictions = []

    for i, (term, model) in enumerate(zip(top_terms, models)):
        if i % 500 == 0:
            print(f"  Predicting term {i+1}/{n_terms}...", flush=True)

        if model is None:
            preds = np.full(len(test_ids), Y[:, i].mean())
        else:
            preds = model.predict(X_test_pca)

        test_predictions.append(preds)

    test_predictions = np.array(test_predictions).T
    print(f"  Test predictions shape: {test_predictions.shape}")

    # Save predictions
    print("\nSaving predictions...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    emb_suffix = '_'.join(embedding_names)
    output_path = OUTPUT_DIR / f'protboost_multi_{emb_suffix}.tsv'
    row_count = 0

    with open(output_path, 'w') as f:
        for i, pid in enumerate(test_ids):
            if i % 50000 == 0:
                print(f"  Writing protein {i+1}/{len(test_ids)}...", flush=True)

            for j, term in enumerate(top_terms):
                score = test_predictions[i, j]
                if score >= 0.01:
                    f.write(f"{pid}\t{term}\t{score:.6f}\n")
                    row_count += 1

    print(f"\nSaved: {output_path}")
    print(f"Rows: {row_count:,}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("\n=== Done ===")


if __name__ == '__main__':
    main()

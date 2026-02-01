"""
ProtBoost with 4500 GO terms, stratified by aspect.

Extends protboost_1000 to cover more terms:
- Biological Process (BP): 3000 terms
- Molecular Function (MF): 1000 terms
- Cellular Component (CC): 500 terms

This gives better coverage of the GO hierarchy while focusing
compute on the most important aspects.

Usage:
    python scripts/protboost_4500.py [--gpu]
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
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
EMBEDDINGS_DIR = DATA_DIR / 'embeddings' / 'multi'
OUTPUT_DIR = BASE_DIR / 'submissions'
MODELS_DIR = BASE_DIR / 'models'

# Aspect configuration (BP is most important in CAFA)
# P=Biological Process, F=Molecular Function, C=Cellular Component
ASPECT_CONFIG = {
    'P': 3000,  # Biological Process
    'F': 1000,  # Molecular Function
    'C': 500,   # Cellular Component
}

# Model configuration
PCA_DIM = 256
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
    """Load training GO term annotations with aspect mapping."""
    terms_path = DATA_DIR / 'Train' / 'train_terms.tsv'
    # Skip header row
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
        else:
            print(f"  {aspect_names.get(aspect, aspect)}: 0 terms (no data)")

    return all_terms


def main():
    parser = argparse.ArgumentParser(description='ProtBoost 4500 terms')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--embedding', type=str, default='esm2_650M',
                        help='Embedding name (esm2_650M, protT5_xl, ankh_large)')
    parser.add_argument('--save-models', action='store_true', default=True, help='Save trained models (default: True)')
    args = parser.parse_args()

    print("=== ProtBoost 4500 ===\n")

    # Load embeddings
    print(f"Loading {args.embedding} embeddings...")
    try:
        train_emb, train_ids, test_emb, test_ids = load_embeddings(args.embedding)
        print(f"  Train: {train_emb.shape}, Test: {test_emb.shape}")
    except Exception as e:
        print(f"  Error loading embeddings: {e}")
        print("  Please ensure embeddings are available at:")
        print(f"  {EMBEDDINGS_DIR / args.embedding}")
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

    # Train models
    print(f"\nTraining LightGBM for {n_terms} terms...")
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

    # Save models if requested
    if args.save_models:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model_path = MODELS_DIR / f'protboost_4500_{args.embedding}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'models': models,
                'top_terms': top_terms,
                'pca': pca,
                'val_aucs': val_aucs,
            }, f)
        print(f"\nSaved models: {model_path}")

    # Generate predictions - MEMORY EFFICIENT VERSION
    # Write directly to file instead of accumulating in array
    print("\nGenerating test predictions (memory-efficient)...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f'protboost_4500_{args.embedding}.tsv'

    row_count = 0
    with open(output_path, 'w') as f:
        for i, pid in enumerate(test_ids):
            if i % 10000 == 0:
                print(f"  Predicting protein {i+1}/{len(test_ids)}...", flush=True)

            # Get embedding for this protein
            x_protein = X_test_pca[i:i+1]

            for j, (term, model) in enumerate(zip(top_terms, models)):
                if model is None:
                    score = Y[:, j].mean()
                else:
                    score = model.predict(x_protein)[0]

                if score >= 0.01:
                    f.write(f"{pid}\t{term}\t{score:.6f}\n")
                    row_count += 1

    print(f"\nSaved: {output_path}")
    print(f"Rows: {row_count:,}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("\n=== Done ===")


if __name__ == '__main__':
    main()

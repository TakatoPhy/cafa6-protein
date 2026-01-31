"""
ProtBoost with 1000 GO terms (enhanced version).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
EMBEDDINGS_DIR = DATA_DIR / 'embeddings' / 'multi'
OUTPUT_DIR = BASE_DIR / 'submissions'

# Configuration - Enhanced
N_TOP_TERMS = 1000  # Increased from 500
PCA_DIM = 256
SEED = 42


def load_embeddings(name: str):
    """Load embeddings and IDs."""
    emb_dir = EMBEDDINGS_DIR / name
    train_emb = np.load(emb_dir / 'train_sequences_emb.npy')
    train_ids = np.load(emb_dir / 'train_sequences_ids.npy', allow_pickle=True)
    test_emb = np.load(emb_dir / 'testsuperset_emb.npy')
    test_ids = np.load(emb_dir / 'testsuperset_ids.npy', allow_pickle=True)
    train_ids = np.array([pid.split('|')[1] if '|' in pid else pid for pid in train_ids])
    test_ids = np.array([pid.split('|')[1] if '|' in pid else pid for pid in test_ids])
    return train_emb, train_ids, test_emb, test_ids


def load_train_terms():
    """Load training GO term annotations."""
    terms_path = DATA_DIR / 'Train' / 'train_terms.tsv'
    df = pd.read_csv(terms_path, sep='\t', header=None, names=['protein', 'go_term', 'aspect'])
    return df


def main():
    print("=== ProtBoost 1000 ===\n")

    print("Loading ESM2-650M embeddings...")
    train_emb, train_ids, test_emb, test_ids = load_embeddings('esm2_650M')
    print(f"  Train: {train_emb.shape}, Test: {test_emb.shape}")

    print("\nLoading training labels...")
    train_terms = load_train_terms()
    print(f"  Total annotations: {len(train_terms):,}")

    term_counts = train_terms['go_term'].value_counts()
    top_terms = term_counts.head(N_TOP_TERMS).index.tolist()
    print(f"\n  Top {N_TOP_TERMS} GO terms (min count: {term_counts.iloc[N_TOP_TERMS-1]:,})")

    train_id_to_idx = {pid: i for i, pid in enumerate(train_ids)}

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

    print("\nPreparing training embeddings...")
    X = np.array([train_emb[train_id_to_idx[p]] for p in train_proteins])
    print(f"  X shape: {X.shape}")

    print(f"\nApplying PCA (dim={PCA_DIM})...")
    pca = PCA(n_components=PCA_DIM, random_state=SEED)
    X_pca = pca.fit_transform(X)
    print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.2%}")

    X_test_pca = pca.transform(test_emb)
    print(f"  X_test_pca shape: {X_test_pca.shape}")

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_pca, Y, test_size=0.1, random_state=SEED
    )
    print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}")

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
    }

    models = []
    val_aucs = []

    for i, term in enumerate(top_terms):
        if i % 100 == 0:
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

    print("\nGenerating test predictions...")
    test_predictions = []

    for i, (term, model) in enumerate(zip(top_terms, models)):
        if i % 100 == 0:
            print(f"  Predicting term {i+1}/{n_terms}...", flush=True)

        if model is None:
            preds = np.full(len(test_ids), Y[:, i].mean())
        else:
            preds = model.predict(X_test_pca)

        test_predictions.append(preds)

    test_predictions = np.array(test_predictions).T
    print(f"  Test predictions shape: {test_predictions.shape}")

    print("\nSaving predictions...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / 'protboost_1000.tsv'
    with open(output_path, 'w') as f:
        for i, pid in enumerate(test_ids):
            if i % 50000 == 0:
                print(f"  Writing protein {i+1}/{len(test_ids)}...", flush=True)

            for j, term in enumerate(top_terms):
                score = test_predictions[i, j]
                if score >= 0.01:
                    f.write(f"{pid}\t{term}\t{score:.6f}\n")

    print(f"\nSaved: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("\n=== Done ===")


if __name__ == '__main__':
    main()

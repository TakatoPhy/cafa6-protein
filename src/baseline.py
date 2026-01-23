"""
CAFA6 Baseline: ESM Embeddings + LightGBM
"""
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path(__file__).parent.parent / 'data'
EMB_DIR = DATA_DIR / 'embeddings'

def extract_id(full_id: str) -> str:
    """Extract protein ID from embedding format."""
    if '|' in full_id:
        return full_id.split('|')[1]
    return full_id

def load_data(min_term_count: int = 500):
    """Load embeddings and labels."""
    print("Loading embeddings...")
    train_emb = np.load(EMB_DIR / 'protein_embeddings_train.npy')
    test_emb = np.load(EMB_DIR / 'protein_embeddings_test.npy')
    train_ids_raw = np.load(EMB_DIR / 'train_ids.npy', allow_pickle=True)
    test_ids = np.load(EMB_DIR / 'test_ids.npy', allow_pickle=True)

    # Clean train IDs
    train_ids = np.array([extract_id(x) for x in train_ids_raw])

    print("Loading labels...")
    train_terms = pd.read_csv(DATA_DIR / 'Train' / 'train_terms.tsv', sep='\t')

    # Filter to frequent terms
    term_counts = train_terms['term'].value_counts()
    frequent_terms = term_counts[term_counts >= min_term_count].index.tolist()
    print(f"Using {len(frequent_terms)} terms (>= {min_term_count} occurrences)")

    train_terms_filtered = train_terms[train_terms['term'].isin(frequent_terms)]

    # Create protein -> terms mapping
    protein_terms = train_terms_filtered.groupby('EntryID')['term'].apply(list).to_dict()

    # Create ID -> embedding index mapping
    train_id_to_idx = {id_: idx for idx, id_ in enumerate(train_ids)}

    # Build X, y for training
    valid_ids = [id_ for id_ in train_ids if id_ in protein_terms]
    X_train = np.array([train_emb[train_id_to_idx[id_]] for id_ in valid_ids])
    y_labels = [protein_terms[id_] for id_ in valid_ids]

    # Binarize labels
    mlb = MultiLabelBinarizer(classes=frequent_terms)
    y_train = mlb.fit_transform(y_labels)

    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shape: {test_emb.shape}")

    return X_train, y_train, test_emb, test_ids, mlb, frequent_terms

def train_models(X_train, y_train, frequent_terms):
    """Train LightGBM for each GO term."""
    print(f"\nTraining {len(frequent_terms)} models...")

    models = []
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'verbose': -1,
        'n_jobs': -1,
    }

    for i, term in enumerate(tqdm(frequent_terms)):
        y_col = y_train[:, i]

        # Skip if no positive samples
        if y_col.sum() == 0:
            models.append(None)
            continue

        train_data = lgb.Dataset(X_train, label=y_col)
        model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
        )
        models.append(model)

    return models

def predict(models, X_test, frequent_terms):
    """Generate predictions for test set."""
    print("\nGenerating predictions...")
    predictions = np.zeros((len(X_test), len(frequent_terms)))

    for i, model in enumerate(tqdm(models)):
        if model is not None:
            predictions[:, i] = model.predict(X_test)

    return predictions

def create_submission(predictions, test_ids, frequent_terms, threshold=0.1):
    """Create submission file."""
    print("\nCreating submission...")
    rows = []

    for i, protein_id in enumerate(tqdm(test_ids)):
        probs = predictions[i]
        for j, term in enumerate(frequent_terms):
            if probs[j] >= threshold:
                rows.append({
                    'EntryID': protein_id,
                    'term': term,
                    'confidence': probs[j]
                })

    submission = pd.DataFrame(rows)
    submission = submission.sort_values(['EntryID', 'confidence'], ascending=[True, False])

    return submission

def main():
    # Load data - 100件以上に拡大
    X_train, y_train, X_test, test_ids, mlb, frequent_terms = load_data(min_term_count=100)

    # Train models
    models = train_models(X_train, y_train, frequent_terms)

    # Predict
    predictions = predict(models, X_test, frequent_terms)

    # Create submission
    submission = create_submission(predictions, test_ids, frequent_terms, threshold=0.1)

    # Save
    output_path = DATA_DIR.parent / 'submission.tsv'
    submission.to_csv(output_path, sep='\t', index=False, header=False)
    print(f"\nSubmission saved to {output_path}")
    print(f"Total predictions: {len(submission)}")

    # Sample
    print("\nSample submission:")
    print(submission.head(10))

if __name__ == '__main__':
    main()

"""
CAFA6 Baseline v2: More terms + Lower threshold
"""
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import warnings
import joblib
from concurrent.futures import ProcessPoolExecutor
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path(__file__).parent.parent / 'data'
EMB_DIR = DATA_DIR / 'embeddings'

def extract_id(full_id: str) -> str:
    if '|' in full_id:
        return full_id.split('|')[1]
    return full_id

def load_data(min_term_count: int = 100):
    """Load embeddings and labels."""
    print("Loading embeddings...")
    train_emb = np.load(EMB_DIR / 'protein_embeddings_train.npy')
    test_emb = np.load(EMB_DIR / 'protein_embeddings_test.npy')
    train_ids_raw = np.load(EMB_DIR / 'train_ids.npy', allow_pickle=True)
    test_ids = np.load(EMB_DIR / 'test_ids.npy', allow_pickle=True)

    train_ids = np.array([extract_id(x) for x in train_ids_raw])

    print("Loading labels...")
    train_terms = pd.read_csv(DATA_DIR / 'Train' / 'train_terms.tsv', sep='\t')

    term_counts = train_terms['term'].value_counts()
    frequent_terms = term_counts[term_counts >= min_term_count].index.tolist()
    print(f"Using {len(frequent_terms)} terms (>= {min_term_count} occurrences)")

    train_terms_filtered = train_terms[train_terms['term'].isin(frequent_terms)]
    protein_terms = train_terms_filtered.groupby('EntryID')['term'].apply(list).to_dict()
    train_id_to_idx = {id_: idx for idx, id_ in enumerate(train_ids)}

    valid_ids = [id_ for id_ in train_ids if id_ in protein_terms]
    X_train = np.array([train_emb[train_id_to_idx[id_]] for id_ in valid_ids])
    y_labels = [protein_terms[id_] for id_ in valid_ids]

    mlb = MultiLabelBinarizer(classes=frequent_terms)
    y_train = mlb.fit_transform(y_labels)

    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shape: {test_emb.shape}")

    return X_train, y_train, test_emb, test_ids, mlb, frequent_terms

def train_single_model(args):
    """Train a single LightGBM model."""
    X_train, y_col, params = args
    if y_col.sum() == 0:
        return None
    train_data = lgb.Dataset(X_train, label=y_col, free_raw_data=False)
    model = lgb.train(params, train_data, num_boost_round=50)  # 減らして高速化
    return model

def train_models(X_train, y_train, frequent_terms):
    """Train LightGBM for each GO term."""
    print(f"\nTraining {len(frequent_terms)} models...")

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'verbose': -1,
        'n_jobs': 1,  # 個別モデルは1スレッド
    }

    models = []
    for i in tqdm(range(len(frequent_terms))):
        y_col = y_train[:, i]
        if y_col.sum() == 0:
            models.append(None)
            continue
        train_data = lgb.Dataset(X_train, label=y_col)
        model = lgb.train(params, train_data, num_boost_round=50)
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

def create_submission(predictions, test_ids, frequent_terms, threshold=0.05):
    """Create submission file with lower threshold."""
    print(f"\nCreating submission (threshold={threshold})...")
    rows = []

    for i, protein_id in enumerate(tqdm(test_ids)):
        probs = predictions[i]
        for j, term in enumerate(frequent_terms):
            if probs[j] >= threshold:
                rows.append({
                    'EntryID': protein_id,
                    'term': term,
                    'confidence': round(probs[j], 3)
                })

    submission = pd.DataFrame(rows)
    submission = submission.sort_values(['EntryID', 'confidence'], ascending=[True, False])

    return submission

def main():
    # Load data with 100+ terms
    X_train, y_train, X_test, test_ids, mlb, frequent_terms = load_data(min_term_count=100)

    # Train models
    models = train_models(X_train, y_train, frequent_terms)

    # Save models for reuse
    print("\nSaving models...")
    joblib.dump((models, frequent_terms), DATA_DIR.parent / 'models_v2.pkl')

    # Predict
    predictions = predict(models, X_test, frequent_terms)

    # Save predictions for threshold tuning
    np.save(DATA_DIR.parent / 'predictions_v2.npy', predictions)

    # Create submission with lower threshold
    submission = create_submission(predictions, test_ids, frequent_terms, threshold=0.05)

    # Save
    output_path = DATA_DIR.parent / 'submission.tsv'
    submission.to_csv(output_path, sep='\t', index=False, header=False)
    print(f"\nSubmission saved to {output_path}")
    print(f"Total predictions: {len(submission)}")

    print("\nSample submission:")
    print(submission.head(10))

if __name__ == '__main__':
    main()

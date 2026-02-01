"""
MLP Stacking Model for GO Term Prediction.

Combines predictions from multiple base models using a learned stacking approach.
This is a simpler alternative to GCN stacking that doesn't require graph operations.

Features per GO term (for each protein):
- Base model predictions (logits or probabilities)
- GO term frequency prior
- Aspect one-hot encoding (BP/MF/CC)

Usage:
    python scripts/stacking_mlp.py --train
    python scripts/stacking_mlp.py --predict
"""
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from parse_go import parse_go_obo

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'submissions'
MODELS_DIR = BASE_DIR / 'models'

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


class StackingMLP(nn.Module):
    """MLP for stacking predictions from base models."""

    def __init__(self, n_base_models: int, hidden_dim: int = 64):
        super().__init__()

        # Input: n_base_models predictions + 3 (aspect one-hot) + 1 (prior)
        input_dim = n_base_models + 4

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class StackingDataset(Dataset):
    """Dataset for stacking model training."""

    def __init__(self, features: np.ndarray, labels: np.ndarray = None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels) if labels is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]


def load_base_predictions(paths: list, names: list) -> dict:
    """Load predictions from multiple base models."""
    all_preds = {}

    for name, path in zip(names, paths):
        print(f"Loading {name}...")
        preds = defaultdict(dict)
        count = 0

        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 3:
                    protein, go_term, score = parts[0], parts[1], float(parts[2])
                    if not go_term.startswith('GO:'):
                        continue
                    preds[protein][go_term] = max(preds[protein].get(go_term, 0), score)
                    count += 1
                    if count % 10_000_000 == 0:
                        print(f"  {count:,} rows...", flush=True)

        print(f"  Loaded {count:,} rows, {len(preds):,} proteins")
        all_preds[name] = preds

    return all_preds


def load_go_info() -> tuple:
    """Load GO ontology information."""
    go_obo_path = DATA_DIR / 'Train' / 'go-basic.obo'
    term_to_parents, term_to_name, term_to_namespace = parse_go_obo(go_obo_path)

    # Map namespace to index
    namespace_to_idx = {
        'biological_process': 0,
        'molecular_function': 1,
        'cellular_component': 2,
    }

    return term_to_namespace, namespace_to_idx


def load_train_labels() -> dict:
    """Load training GO term annotations."""
    terms_path = DATA_DIR / 'Train' / 'train_terms.tsv'
    df = pd.read_csv(terms_path, sep='\t', header=None, names=['protein', 'go_term', 'aspect'])

    labels = defaultdict(set)
    for _, row in df.iterrows():
        labels[row['protein']].add(row['go_term'])

    return labels


def compute_priors(train_labels: dict, all_terms: set) -> dict:
    """Compute prior probabilities for each GO term."""
    n_proteins = len(train_labels)
    term_counts = defaultdict(int)

    for protein, terms in train_labels.items():
        for term in terms:
            term_counts[term] += 1

    priors = {}
    for term in all_terms:
        priors[term] = term_counts.get(term, 0) / n_proteins

    return priors


def prepare_features(base_preds: dict, proteins: list, terms: list,
                     term_to_namespace: dict, namespace_to_idx: dict,
                     priors: dict, labels: dict = None) -> tuple:
    """
    Prepare feature matrix for stacking.

    Returns:
        X: Feature matrix [n_samples, n_features]
        y: Label vector [n_samples] (if labels provided)
        sample_info: List of (protein, term) pairs
    """
    model_names = list(base_preds.keys())
    n_models = len(model_names)

    X_list = []
    y_list = []
    sample_info = []

    for protein in proteins:
        # Get all terms predicted by any model for this protein
        protein_terms = set()
        for model_preds in base_preds.values():
            if protein in model_preds:
                protein_terms.update(model_preds[protein].keys())

        # Filter to terms we care about
        protein_terms = protein_terms.intersection(terms)

        for term in protein_terms:
            # Base model predictions
            model_scores = []
            for model_name in model_names:
                score = base_preds[model_name].get(protein, {}).get(term, 0)
                model_scores.append(score)

            # Aspect one-hot
            namespace = term_to_namespace.get(term, 'biological_process')
            aspect_idx = namespace_to_idx.get(namespace, 0)
            aspect_onehot = [0, 0, 0]
            aspect_onehot[aspect_idx] = 1

            # Prior
            prior = priors.get(term, 0)

            # Combine features
            features = model_scores + aspect_onehot + [prior]
            X_list.append(features)

            # Label
            if labels is not None:
                label = 1.0 if term in labels.get(protein, set()) else 0.0
                y_list.append(label)

            sample_info.append((protein, term))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32) if y_list else None

    return X, y, sample_info


def train_stacker(X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  n_models: int, epochs: int = 20, batch_size: int = 4096):
    """Train the stacking MLP."""
    print(f"\nTraining stacker on {len(X_train):,} samples...")

    train_dataset = StackingDataset(X_train, y_train)
    val_dataset = StackingDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = StackingMLP(n_models).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                val_loss += criterion(preds, y_batch).item()

        print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss/len(train_loader):.4f}, Val: {val_loss/len(val_loader):.4f}")

    return model


def main():
    parser = argparse.ArgumentParser(description='MLP Stacking')
    parser.add_argument('--train', action='store_true', help='Train the stacker')
    parser.add_argument('--predict', action='store_true', help='Generate predictions')
    args = parser.parse_args()

    print("=== MLP Stacking ===\n")

    # Define base models
    # NOTE: cafa_opt/goaはスコア分布が異なる（1.0に集中）ので使わない
    # 0.364事件の教訓: 同じスコア分布のモデルのみ使う
    base_paths = [
        BASE_DIR / 'submission.tsv',
        DATA_DIR / 'processed' / 'notebooks' / 'external' / 'nb153' / 'submission.tsv',
    ]
    base_names = ['baseline', 'nb153']

    # ProtBoost完了後に追加（パスを変更）
    protboost_path = OUTPUT_DIR / 'protboost_4500.tsv'
    if protboost_path.exists():
        base_paths.append(protboost_path)
        base_names.append('protboost')

    # Filter to existing files
    existing = [(n, p) for n, p in zip(base_names, base_paths) if p.exists()]
    if len(existing) < 2:
        print("Error: Need at least 2 base model predictions")
        return

    base_names = [n for n, _ in existing]
    base_paths = [p for _, p in existing]
    print(f"Using {len(base_names)} base models: {', '.join(base_names)}")

    # Load base predictions
    base_preds = load_base_predictions(base_paths, base_names)

    # Load GO info
    print("\nLoading GO ontology...")
    term_to_namespace, namespace_to_idx = load_go_info()

    # Get all GO terms from predictions
    all_terms = set()
    for model_preds in base_preds.values():
        for protein_preds in model_preds.values():
            all_terms.update(protein_preds.keys())
    print(f"Total GO terms: {len(all_terms):,}")

    if args.train:
        # Load training labels
        print("\nLoading training labels...")
        train_labels = load_train_labels()

        # Get training proteins that appear in predictions
        train_proteins = list(set(train_labels.keys()).intersection(
            set.union(*[set(preds.keys()) for preds in base_preds.values()])
        ))
        print(f"Training proteins: {len(train_proteins):,}")

        # Compute priors
        priors = compute_priors(train_labels, all_terms)

        # Prepare features
        print("\nPreparing features...")
        X, y, sample_info = prepare_features(
            base_preds, train_proteins[:10000], all_terms,  # Limit for memory
            term_to_namespace, namespace_to_idx, priors, train_labels
        )
        print(f"Feature matrix: {X.shape}")
        print(f"Positive rate: {y.mean():.4f}")

        # Split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train
        model = train_stacker(X_train, y_train, X_val, y_val, len(base_names))

        # Save model
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODELS_DIR / 'stacking_mlp.pt')
        print(f"\nModel saved: {MODELS_DIR / 'stacking_mlp.pt'}")

    if args.predict:
        # Load model
        model = StackingMLP(len(base_names)).to(device)
        model.load_state_dict(torch.load(MODELS_DIR / 'stacking_mlp.pt'))
        model.eval()

        # Get test proteins
        test_proteins = set()
        for model_preds in base_preds.values():
            test_proteins.update(model_preds.keys())
        test_proteins = sorted(test_proteins)
        print(f"\nTest proteins: {len(test_proteins):,}")

        # Compute priors from predictions (approximate)
        priors = {term: 0.01 for term in all_terms}  # Default prior

        # Generate predictions
        print("\nGenerating predictions...")
        output_path = OUTPUT_DIR / 'stacking_mlp.tsv'
        row_count = 0

        with open(output_path, 'w') as f:
            for i, protein in enumerate(test_proteins):
                if i % 10000 == 0:
                    print(f"  Protein {i:,}/{len(test_proteins):,}...", flush=True)

                # Prepare features for this protein
                X, _, sample_info = prepare_features(
                    base_preds, [protein], all_terms,
                    term_to_namespace, namespace_to_idx, priors
                )

                if len(X) == 0:
                    continue

                # Predict
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X).to(device)
                    preds = model(X_tensor).cpu().numpy()

                # Write
                for (_, term), score in zip(sample_info, preds):
                    if score >= 0.01:
                        f.write(f"{protein}\t{term}\t{score:.6f}\n")
                        row_count += 1

        print(f"\nSaved: {output_path}")
        print(f"Rows: {row_count:,}")


if __name__ == '__main__':
    main()

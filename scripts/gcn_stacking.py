"""
GCN Stacking Model for GO Term Prediction.

Uses Graph Convolutional Networks to leverage the GO hierarchy for prediction.
Based on CAFA5 2nd place (ProtBoost) approach.

The GO hierarchy is used as a graph where:
- Nodes are GO terms
- Edges connect child terms to parent terms

For each protein, we create a feature vector for each GO term and use
GCN to propagate information through the graph.

Features per GO term:
- Base model predictions (5 models × 4 features = 20)
  - Logit value
  - Forward propagation score
  - Backward propagation score
  - Prior flag
- IEA flag (1 feature)
- Learnable GO term embedding (8 features)

Total: 29 features per GO term

Usage:
    python scripts/gcn_stacking.py --train
    python scripts/gcn_stacking.py --predict
"""
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import argparse
import warnings
warnings.filterwarnings('ignore')

# Add parent to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from parse_go import parse_go_obo, get_all_ancestors

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'submissions'
MODELS_DIR = BASE_DIR / 'models'

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


def build_go_graph(term_to_parents: dict, terms: list) -> tuple:
    """
    Build edge index for GO graph.

    Returns:
        edge_index: [2, num_edges] tensor of (source, target) pairs
        term_to_idx: mapping from GO term to node index
    """
    term_to_idx = {t: i for i, t in enumerate(terms)}

    edges = []
    for child in terms:
        if child in term_to_parents:
            for parent in term_to_parents[child]:
                if parent in term_to_idx:
                    # Edge from child to parent (for propagation)
                    edges.append([term_to_idx[child], term_to_idx[parent]])
                    # Bidirectional for message passing
                    edges.append([term_to_idx[parent], term_to_idx[child]])

    if edges:
        edge_index = torch.tensor(edges).T.long()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return edge_index, term_to_idx


class GOGraphConv(nn.Module):
    """Simple graph convolution for GO hierarchy."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        x: [num_nodes, in_channels]
        edge_index: [2, num_edges]
        """
        if edge_index.shape[1] == 0:
            return self.lin(x)

        # Aggregate neighbor features
        row, col = edge_index
        num_nodes = x.size(0)

        # Mean aggregation
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        deg = deg.clamp(min=1)

        agg = torch.zeros_like(x)
        agg.scatter_add_(0, row.unsqueeze(-1).expand(-1, x.size(-1)), x[col])
        agg = agg / deg.unsqueeze(-1)

        # Combine self and neighbor
        out = self.lin(x + agg)
        return out


class GCNStacker(nn.Module):
    """GCN-based stacking model for GO prediction."""

    def __init__(self, n_base_models: int, n_terms: int,
                 hidden_dim: int = 64, go_emb_dim: int = 8):
        super().__init__()

        # Features: base_models + aspect_onehot + prior
        n_features = n_base_models + 4

        # Learnable GO term embeddings
        self.go_embeddings = nn.Embedding(n_terms, go_emb_dim)

        # Input dimension: features + go_embedding
        input_dim = n_features + go_emb_dim

        # GCN layers
        self.conv1 = GOGraphConv(input_dim, hidden_dim)
        self.conv2 = GOGraphConv(hidden_dim, hidden_dim)
        self.conv3 = GOGraphConv(hidden_dim, hidden_dim // 2)

        # Output layer
        self.output = nn.Linear(hidden_dim // 2, 1)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                term_indices: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, num_terms, n_features]
        edge_index: [2, num_edges]
        term_indices: [num_terms] - indices for GO embeddings
        """
        batch_size = x.size(0)
        num_terms = x.size(1)

        # Add GO embeddings
        go_emb = self.go_embeddings(term_indices)  # [num_terms, go_emb_dim]
        go_emb = go_emb.unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.cat([x, go_emb], dim=-1)

        # Process each sample in batch
        outputs = []
        for i in range(batch_size):
            xi = x[i]  # [num_terms, features]

            # GCN layers
            h = F.relu(self.conv1(xi, edge_index))
            h = self.dropout(h)
            h = F.relu(self.conv2(h, edge_index))
            h = self.dropout(h)
            h = F.relu(self.conv3(h, edge_index))

            # Output
            out = torch.sigmoid(self.output(h)).squeeze(-1)
            outputs.append(out)

        return torch.stack(outputs)  # [batch_size, num_terms]


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


def prepare_protein_features(base_preds: dict, protein: str, terms: list,
                             term_to_namespace: dict, namespace_to_idx: dict,
                             priors: dict) -> np.ndarray:
    """
    Prepare feature matrix for a single protein.

    Returns:
        X: [num_terms, n_features]
    """
    model_names = list(base_preds.keys())
    n_terms = len(terms)

    X = np.zeros((n_terms, len(model_names) + 4), dtype=np.float32)

    for i, term in enumerate(terms):
        # Base model scores
        for j, model_name in enumerate(model_names):
            X[i, j] = base_preds[model_name].get(protein, {}).get(term, 0)

        # Aspect one-hot
        namespace = term_to_namespace.get(term, 'biological_process')
        aspect_idx = namespace_to_idx.get(namespace, 0)
        X[i, len(model_names) + aspect_idx] = 1.0

        # Prior
        X[i, len(model_names) + 3] = priors.get(term, 0)

    return X


def train_gcn(base_preds: dict, train_proteins: list, terms: list,
              term_to_parents: dict, term_to_namespace: dict,
              namespace_to_idx: dict, priors: dict, labels: dict,
              epochs: int = 10, batch_size: int = 32):
    """Train the GCN stacker."""
    n_models = len(base_preds)
    n_terms = len(terms)

    # Build GO graph
    print("Building GO graph...")
    edge_index, term_to_idx = build_go_graph(term_to_parents, terms)
    edge_index = edge_index.to(device)
    term_indices = torch.arange(n_terms).to(device)
    print(f"  Nodes: {n_terms}, Edges: {edge_index.shape[1]}")

    # Prepare training data
    print("Preparing training data...")
    X_list = []
    y_list = []

    for protein in train_proteins[:5000]:  # Limit for memory
        X = prepare_protein_features(
            base_preds, protein, terms,
            term_to_namespace, namespace_to_idx, priors
        )
        y = np.array([1.0 if term in labels.get(protein, set()) else 0.0 for term in terms])

        X_list.append(X)
        y_list.append(y)

    X_all = np.array(X_list)
    y_all = np.array(y_list)
    print(f"  Shape: X={X_all.shape}, y={y_all.shape}")

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42
    )

    # Model
    model = GCNStacker(n_models, n_terms).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"\nTraining GCN ({epochs} epochs)...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        n_batches = 0

        for i in range(0, len(X_train), batch_size):
            X_batch = torch.FloatTensor(X_train[i:i+batch_size]).to(device)
            y_batch = torch.FloatTensor(y_train[i:i+batch_size]).to(device)

            optimizer.zero_grad()
            preds = model(X_batch, edge_index, term_indices)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            X_val_t = torch.FloatTensor(X_val).to(device)
            y_val_t = torch.FloatTensor(y_val).to(device)
            val_preds = model(X_val_t, edge_index, term_indices)
            val_loss = criterion(val_preds, y_val_t).item()

        print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss/n_batches:.4f}, Val: {val_loss:.4f}")

    return model, edge_index, term_indices


def main():
    parser = argparse.ArgumentParser(description='GCN Stacking')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--predict', action='store_true', help='Generate predictions')
    parser.add_argument('--top-k', type=int, default=5000, help='Number of top GO terms to use')
    args = parser.parse_args()

    print("=== GCN Stacking ===\n")

    # Define base models
    # NOTE: cafa_opt/goaはスコア分布が異なる（1.0に集中）ので使わない
    # 0.364事件の教訓: 同じスコア分布のモデルのみ使う
    base_paths = [
        BASE_DIR / 'submission.tsv',
        DATA_DIR / 'processed' / 'notebooks' / 'external' / 'nb153' / 'submission.tsv',
    ]
    base_names = ['baseline', 'nb153']

    # ProtBoost完了後に追加
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

    # Load GO ontology
    print("\nLoading GO ontology...")
    go_obo_path = DATA_DIR / 'Train' / 'go-basic.obo'
    term_to_parents, term_to_name, term_to_namespace = parse_go_obo(go_obo_path)
    namespace_to_idx = {'biological_process': 0, 'molecular_function': 1, 'cellular_component': 2}

    # Get top-k most frequent GO terms
    print("\nIdentifying top GO terms...")
    term_counts = defaultdict(int)
    for model_preds in base_preds.values():
        for protein_preds in model_preds.values():
            for term in protein_preds.keys():
                term_counts[term] += 1

    terms = sorted(term_counts.keys(), key=lambda t: -term_counts[t])[:args.top_k]
    print(f"Using top {len(terms)} GO terms")

    # Compute priors
    priors = {term: count / sum(term_counts.values()) for term, count in term_counts.items()}

    if args.train:
        # Load training labels
        print("\nLoading training labels...")
        terms_path = DATA_DIR / 'Train' / 'train_terms.tsv'
        df = pd.read_csv(terms_path, sep='\t', header=None, names=['protein', 'go_term', 'aspect'])
        labels = defaultdict(set)
        for _, row in df.iterrows():
            labels[row['protein']].add(row['go_term'])

        # Get training proteins
        train_proteins = list(set(labels.keys()).intersection(
            set.union(*[set(preds.keys()) for preds in base_preds.values()])
        ))
        print(f"Training proteins: {len(train_proteins):,}")

        # Train
        model, edge_index, term_indices = train_gcn(
            base_preds, train_proteins, terms,
            term_to_parents, term_to_namespace, namespace_to_idx,
            priors, labels
        )

        # Save
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': model.state_dict(),
            'terms': terms,
            'n_models': len(base_names),
        }, MODELS_DIR / 'gcn_stacker.pt')
        print(f"\nModel saved: {MODELS_DIR / 'gcn_stacker.pt'}")

    if args.predict:
        # Load model
        checkpoint = torch.load(MODELS_DIR / 'gcn_stacker.pt')
        terms = checkpoint['terms']
        n_models = checkpoint['n_models']

        model = GCNStacker(n_models, len(terms)).to(device)
        model.load_state_dict(checkpoint['model_state'])
        model.eval()

        # Build graph
        edge_index, _ = build_go_graph(term_to_parents, terms)
        edge_index = edge_index.to(device)
        term_indices = torch.arange(len(terms)).to(device)

        # Get test proteins
        test_proteins = set()
        for model_preds in base_preds.values():
            test_proteins.update(model_preds.keys())
        test_proteins = sorted(test_proteins)
        print(f"\nTest proteins: {len(test_proteins):,}")

        # Generate predictions
        print("\nGenerating predictions...")
        output_path = OUTPUT_DIR / 'gcn_stacking.tsv'
        row_count = 0

        with open(output_path, 'w') as f:
            for i, protein in enumerate(test_proteins):
                if i % 5000 == 0:
                    print(f"  Protein {i:,}/{len(test_proteins):,}...", flush=True)

                X = prepare_protein_features(
                    base_preds, protein, terms,
                    term_to_namespace, namespace_to_idx, priors
                )

                with torch.no_grad():
                    X_t = torch.FloatTensor(X).unsqueeze(0).to(device)
                    preds = model(X_t, edge_index, term_indices).squeeze(0).cpu().numpy()

                for j, (term, score) in enumerate(zip(terms, preds)):
                    if score >= 0.01:
                        f.write(f"{protein}\t{term}\t{score:.6f}\n")
                        row_count += 1

        print(f"\nSaved: {output_path}")
        print(f"Rows: {row_count:,}")


if __name__ == '__main__':
    main()

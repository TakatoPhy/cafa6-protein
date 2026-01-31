"""
Dual-Tower Architecture for Protein Function Prediction.

Tower 1: Protein encoder (MLP on ESM2 embeddings)
Tower 2: GO term encoder (learnable embeddings)
Prediction: Dot product similarity

Uses GPU for training.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
EMBEDDINGS_DIR = DATA_DIR / 'embeddings' / 'multi'
OUTPUT_DIR = BASE_DIR / 'submissions'

# Config
EMBEDDING_DIM = 256  # Output dimension for both towers
HIDDEN_DIM = 512
N_TOP_TERMS = 2000  # More terms than ProtBoost
BATCH_SIZE = 4096
EPOCHS = 10
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42


class ProteinEncoder(nn.Module):
    """Encodes protein embeddings to shared space."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class GOTermEncoder(nn.Module):
    """Learnable GO term embeddings."""
    def __init__(self, n_terms: int, output_dim: int):
        super().__init__()
        self.embeddings = nn.Embedding(n_terms, output_dim)
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, term_ids):
        return F.normalize(self.embeddings(term_ids), dim=-1)


class DualTowerModel(nn.Module):
    """Dual-tower model combining protein and GO term encoders."""
    def __init__(self, protein_dim: int, n_terms: int, hidden_dim: int, embed_dim: int):
        super().__init__()
        self.protein_encoder = ProteinEncoder(protein_dim, hidden_dim, embed_dim)
        self.go_encoder = GOTermEncoder(n_terms, embed_dim)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, protein_emb, term_ids):
        protein_vec = self.protein_encoder(protein_emb)  # (batch, embed_dim)
        go_vec = self.go_encoder(term_ids)  # (batch, embed_dim)
        # Scaled dot product
        logits = (protein_vec * go_vec).sum(dim=-1) * self.temperature
        return logits

    def predict_all_terms(self, protein_emb, n_terms: int):
        """Predict scores for all GO terms for given proteins."""
        protein_vec = self.protein_encoder(protein_emb)  # (batch, embed_dim)
        all_term_ids = torch.arange(n_terms, device=protein_emb.device)
        go_vec = self.go_encoder(all_term_ids)  # (n_terms, embed_dim)
        # Compute all pairwise similarities
        logits = torch.matmul(protein_vec, go_vec.T) * self.temperature  # (batch, n_terms)
        return torch.sigmoid(logits)


class ProteinGODataset(Dataset):
    """Dataset for (protein, GO term, label) triplets."""
    def __init__(self, protein_embs, labels, term_to_idx):
        self.protein_embs = protein_embs
        self.labels = labels  # DataFrame with protein, go_term columns
        self.term_to_idx = term_to_idx

        # Create positive samples
        self.samples = []
        for _, row in labels.iterrows():
            if row['go_term'] in term_to_idx:
                self.samples.append((row['protein_idx'], term_to_idx[row['go_term']], 1.0))

        # Add negative samples (random sampling)
        n_terms = len(term_to_idx)
        n_proteins = len(protein_embs)
        n_negatives = len(self.samples)  # 1:1 ratio

        np.random.seed(SEED)
        for _ in range(n_negatives):
            p_idx = np.random.randint(n_proteins)
            t_idx = np.random.randint(n_terms)
            self.samples.append((p_idx, t_idx, 0.0))

        print(f"Dataset: {len(self.samples)} samples ({len(self.samples)//2} pos, {len(self.samples)//2} neg)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p_idx, t_idx, label = self.samples[idx]
        return (
            torch.tensor(self.protein_embs[p_idx], dtype=torch.float32),
            torch.tensor(t_idx, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )


def load_data():
    """Load embeddings and labels."""
    print("Loading ESM2-650M embeddings...")
    emb_dir = EMBEDDINGS_DIR / 'esm2_650M'

    train_emb = np.load(emb_dir / 'train_sequences_emb.npy')
    train_ids = np.load(emb_dir / 'train_sequences_ids.npy', allow_pickle=True)
    test_emb = np.load(emb_dir / 'testsuperset_emb.npy')
    test_ids = np.load(emb_dir / 'testsuperset_ids.npy', allow_pickle=True)

    # Extract UniProt IDs
    train_ids = np.array([pid.split('|')[1] if '|' in pid else pid for pid in train_ids])
    test_ids = np.array([pid.split('|')[1] if '|' in pid else pid for pid in test_ids])

    print(f"  Train: {train_emb.shape}, Test: {test_emb.shape}")

    # Load labels
    print("Loading training labels...")
    terms_path = DATA_DIR / 'Train' / 'train_terms.tsv'
    train_terms = pd.read_csv(terms_path, sep='\t', header=None, names=['protein', 'go_term', 'aspect'])
    print(f"  Total annotations: {len(train_terms):,}")

    return train_emb, train_ids, test_emb, test_ids, train_terms


def train_model(model, train_loader, val_loader, epochs: int):
    """Train the dual-tower model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for protein_emb, term_ids, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            protein_emb = protein_emb.to(DEVICE)
            term_ids = term_ids.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(protein_emb, term_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for protein_emb, term_ids, labels in val_loader:
                protein_emb = protein_emb.to(DEVICE)
                term_ids = term_ids.to(DEVICE)
                labels = labels.to(DEVICE)

                logits = model(protein_emb, term_ids)
                loss = criterion(logits, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step()

        print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), OUTPUT_DIR / 'dual_tower_best.pt')

    # Load best model
    model.load_state_dict(torch.load(OUTPUT_DIR / 'dual_tower_best.pt'))
    return model


def main():
    print(f"=== Dual-Tower Architecture ===")
    print(f"Device: {DEVICE}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load data
    train_emb, train_ids, test_emb, test_ids, train_terms = load_data()

    # Get top GO terms
    term_counts = train_terms['go_term'].value_counts()
    top_terms = term_counts.head(N_TOP_TERMS).index.tolist()
    term_to_idx = {t: i for i, t in enumerate(top_terms)}
    print(f"\nTop {N_TOP_TERMS} GO terms (min count: {term_counts.iloc[N_TOP_TERMS-1]:,})")

    # Create protein index mapping
    train_id_to_idx = {pid: i for i, pid in enumerate(train_ids)}

    # Filter labels
    train_terms_filtered = train_terms[
        (train_terms['protein'].isin(train_id_to_idx)) &
        (train_terms['go_term'].isin(top_terms))
    ].copy()
    train_terms_filtered['protein_idx'] = train_terms_filtered['protein'].map(train_id_to_idx)

    # Split
    train_labels, val_labels = train_test_split(
        train_terms_filtered, test_size=0.1, random_state=SEED
    )

    # Create datasets
    print("\nCreating datasets...")
    train_dataset = ProteinGODataset(train_emb, train_labels, term_to_idx)
    val_dataset = ProteinGODataset(train_emb, val_labels, term_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Create model
    print("\nCreating model...")
    model = DualTowerModel(
        protein_dim=train_emb.shape[1],
        n_terms=N_TOP_TERMS,
        hidden_dim=HIDDEN_DIM,
        embed_dim=EMBEDDING_DIM,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Train
    print("\nTraining...")
    model = train_model(model, train_loader, val_loader, EPOCHS)

    # Generate predictions
    print("\nGenerating test predictions...")
    model.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / 'dual_tower.tsv'

    with torch.no_grad(), open(output_path, 'w') as f:
        test_tensor = torch.tensor(test_emb, dtype=torch.float32)

        for i in tqdm(range(0, len(test_ids), 1000), desc="Predicting"):
            batch = test_tensor[i:i+1000].to(DEVICE)
            scores = model.predict_all_terms(batch, N_TOP_TERMS)  # (batch, n_terms)
            scores = scores.cpu().numpy()

            for j, pid in enumerate(test_ids[i:i+1000]):
                for k, term in enumerate(top_terms):
                    score = scores[j, k]
                    if score >= 0.01:
                        f.write(f"{pid}\t{term}\t{score:.6f}\n")

    print(f"\nSaved: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("\n=== Done ===")


if __name__ == '__main__':
    main()

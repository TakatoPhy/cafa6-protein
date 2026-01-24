"""
CAFA6 Neural Network with ESM2-650M embeddings (1280 dim)
"""
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = Path(__file__).parent.parent / 'data'
ESM2_DIR = DATA_DIR / 'embeddings' / 'esm2'

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

def extract_id(full_id: str) -> str:
    if '|' in str(full_id):
        return full_id.split('|')[1]
    return str(full_id)

class MultiLabelMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x):
        return self.net(x)

def load_data(min_term_count: int = 50):
    print("Loading ESM2-650M embeddings...")
    esm2_emb = np.load(ESM2_DIR / 'protein_embeddings.npy')
    esm2_ids = pd.read_csv(ESM2_DIR / 'protein_ids.csv')['protein_id'].values

    # Create ID to index mapping for ESM2
    esm2_id_to_idx = {id_: idx for idx, id_ in enumerate(esm2_ids)}

    print(f"ESM2 embeddings: {esm2_emb.shape}")

    # Load train/test IDs from original embeddings
    train_ids_raw = np.load(DATA_DIR / 'embeddings' / 'train_ids.npy', allow_pickle=True)
    test_ids = np.load(DATA_DIR / 'embeddings' / 'test_ids.npy', allow_pickle=True)

    train_ids = np.array([extract_id(x) for x in train_ids_raw])

    print("Loading labels...")
    train_terms = pd.read_csv(DATA_DIR / 'Train' / 'train_terms.tsv', sep='\t')

    term_counts = train_terms['term'].value_counts()
    frequent_terms = term_counts[term_counts >= min_term_count].index.tolist()
    print(f"Using {len(frequent_terms)} terms (>= {min_term_count} occurrences)")

    train_terms_filtered = train_terms[train_terms['term'].isin(frequent_terms)]
    protein_terms = train_terms_filtered.groupby('EntryID')['term'].apply(list).to_dict()

    # Get train embeddings from ESM2
    valid_train_ids = []
    X_train_list = []
    y_labels = []

    for id_ in train_ids:
        if id_ in protein_terms and id_ in esm2_id_to_idx:
            valid_train_ids.append(id_)
            X_train_list.append(esm2_emb[esm2_id_to_idx[id_]])
            y_labels.append(protein_terms[id_])

    X_train = np.array(X_train_list)

    # Get test embeddings from ESM2
    X_test_list = []
    valid_test_ids = []

    for id_ in test_ids:
        if id_ in esm2_id_to_idx:
            valid_test_ids.append(id_)
            X_test_list.append(esm2_emb[esm2_id_to_idx[id_]])

    X_test = np.array(X_test_list)
    valid_test_ids = np.array(valid_test_ids)

    mlb = MultiLabelBinarizer(classes=frequent_terms)
    y_train = mlb.fit_transform(y_labels)

    print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test shape: {X_test.shape}")

    return X_train, y_train, X_test, valid_test_ids, mlb, frequent_terms

def train_model(X_train, y_train, num_labels, epochs=30, batch_size=256, lr=1e-3):
    print(f"\nTraining NN model ({epochs} epochs)...")

    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Larger model for 1280-dim input
    model = MultiLabelMLP(
        input_dim=X_train.shape[1],
        hidden_dim=1024,  # Increased from 512
        output_dim=num_labels,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    return model

def predict(model, X_test, batch_size=512):
    print("\nGenerating predictions...")
    model.eval()

    X_tensor = torch.FloatTensor(X_test)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for (X_batch,) in tqdm(dataloader):
            X_batch = X_batch.to(device)
            outputs = torch.sigmoid(model(X_batch))
            all_preds.append(outputs.cpu().numpy())

    predictions = np.vstack(all_preds)
    return predictions

def create_submission(predictions, test_ids, frequent_terms, threshold=0.05):
    print(f"\nCreating submission (threshold={threshold})...")
    rows = []

    for i, protein_id in enumerate(tqdm(test_ids)):
        probs = predictions[i]
        for j, term in enumerate(frequent_terms):
            if probs[j] >= threshold:
                rows.append({
                    'EntryID': protein_id,
                    'term': term,
                    'confidence': round(float(probs[j]), 3)
                })

    submission = pd.DataFrame(rows)
    submission = submission.sort_values(['EntryID', 'confidence'], ascending=[True, False])
    return submission

def main():
    # Load data with more terms (ESM2 can handle more)
    X_train, y_train, X_test, test_ids, mlb, frequent_terms = load_data(min_term_count=50)

    # Train model
    model = train_model(X_train, y_train, len(frequent_terms), epochs=30, batch_size=256)

    # Save model
    torch.save(model.state_dict(), DATA_DIR.parent / 'nn_esm2_model.pt')

    # Predict
    predictions = predict(model, X_test)
    np.save(DATA_DIR.parent / 'predictions_esm2.npy', predictions)

    # Create submission
    submission = create_submission(predictions, test_ids, frequent_terms, threshold=0.05)

    # Save
    output_path = DATA_DIR.parent / 'submission.tsv'
    submission.to_csv(output_path, sep='\t', index=False, header=False)
    print(f"\nSubmission saved to {output_path}")
    print(f"Total predictions: {len(submission)}")

    print("\nSample:")
    print(submission.head(10))

if __name__ == '__main__':
    main()

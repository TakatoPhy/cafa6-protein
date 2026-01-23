"""
CAFA6 Neural Network Baseline: All labels at once
Much faster than training separate models
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
EMB_DIR = DATA_DIR / 'embeddings'

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

def extract_id(full_id: str) -> str:
    if '|' in full_id:
        return full_id.split('|')[1]
    return full_id

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
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

def load_data(min_term_count: int = 100):
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

def train_model(X_train, y_train, num_labels, epochs=20, batch_size=256, lr=1e-3):
    print(f"\nTraining NN model ({epochs} epochs)...")

    # Convert to tensors
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = MultiLabelMLP(
        input_dim=X_train.shape[1],
        hidden_dim=512,
        output_dim=num_labels,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
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
    # Load data
    X_train, y_train, X_test, test_ids, mlb, frequent_terms = load_data(min_term_count=100)

    # Train model
    model = train_model(X_train, y_train, len(frequent_terms), epochs=30, batch_size=512)

    # Save model
    torch.save(model.state_dict(), DATA_DIR.parent / 'nn_model.pt')

    # Predict
    predictions = predict(model, X_test)
    np.save(DATA_DIR.parent / 'predictions_nn.npy', predictions)

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

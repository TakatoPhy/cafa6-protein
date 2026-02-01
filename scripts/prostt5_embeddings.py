#!/usr/bin/env python3
"""
ProstT5を使って配列から構造を考慮した埋め込みを生成

ProstT5は配列から3Di（構造トークン）を予測できるため、
AlphaFold構造のダウンロードが不要。

Usage:
    python scripts/prostt5_embeddings.py --split train --batch-size 8
    python scripts/prostt5_embeddings.py --split test --batch-size 8

Output:
    data/embeddings/prostt5/{split}_embeddings.npy
    data/embeddings/prostt5/{split}_ids.npy
"""
import argparse
import numpy as np
import torch
from pathlib import Path
from transformers import T5Tokenizer, T5EncoderModel
import re
from tqdm import tqdm

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = DATA_DIR / 'embeddings' / 'prostt5'


def load_fasta(fasta_path: Path) -> list[tuple[str, str]]:
    """Load FASTA file and return list of (id, sequence) tuples."""
    sequences = []
    current_id = None
    current_seq = []

    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences.append((current_id, ''.join(current_seq)))
                # Extract protein ID (first word after >)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)

        if current_id is not None:
            sequences.append((current_id, ''.join(current_seq)))

    return sequences


def prepare_sequence(seq: str) -> str:
    """Prepare sequence for ProstT5."""
    # Replace rare amino acids with X
    seq = re.sub(r"[UZOB]", "X", seq.upper())
    # Add spaces between amino acids
    seq = " ".join(list(seq))
    # Add AA2fold prefix for sequence -> embedding
    return f"<AA2fold> {seq}"


def generate_embeddings(
    sequences: list[tuple[str, str]],
    model,
    tokenizer,
    device,
    batch_size: int = 8,
    max_length: int = 1024,
) -> tuple[np.ndarray, list[str]]:
    """Generate embeddings for sequences."""

    embeddings = []
    ids = []

    # Process in batches
    for i in tqdm(range(0, len(sequences), batch_size), desc="Generating embeddings"):
        batch = sequences[i:i + batch_size]
        batch_ids = [seq_id for seq_id, _ in batch]
        batch_seqs = [prepare_sequence(seq) for _, seq in batch]

        # Tokenize
        inputs = tokenizer.batch_encode_plus(
            batch_seqs,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(device)

        # Generate embeddings
        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask
            )

        # Extract per-protein embeddings (mean pooling)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, 1024)
        attention_mask = inputs.attention_mask.unsqueeze(-1)  # (batch, seq_len, 1)

        # Mean pooling over non-padding tokens
        masked_hidden = last_hidden * attention_mask
        summed = masked_hidden.sum(dim=1)  # (batch, 1024)
        counts = attention_mask.sum(dim=1)  # (batch, 1)
        mean_pooled = summed / counts  # (batch, 1024)

        embeddings.append(mean_pooled.cpu().numpy())
        ids.extend(batch_ids)

    return np.vstack(embeddings), ids


def main():
    parser = argparse.ArgumentParser(description='Generate ProstT5 embeddings')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'test'],
                        help='Which split to process')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for processing')
    parser.add_argument('--max-length', type=int, default=1024,
                        help='Maximum sequence length')
    parser.add_argument('--checkpoint-interval', type=int, default=1000,
                        help='Save checkpoint every N batches')
    args = parser.parse_args()

    print("=== ProstT5 Embedding Generation ===\n")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    print("\nLoading ProstT5 model...")
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/ProstT5', do_lower_case=False)
    model = T5EncoderModel.from_pretrained("Rostlab/ProstT5")
    model = model.to(device)

    # Use half precision on GPU
    if device.type == 'cuda':
        model = model.half()
        print("  Using half precision (fp16)")

    model.eval()
    print("  Model loaded!")

    # Load sequences
    if args.split == 'train':
        fasta_path = DATA_DIR / 'Train' / 'train_sequences.fasta'
    else:
        fasta_path = DATA_DIR / 'Test' / 'testsuperset.fasta'

    print(f"\nLoading sequences from {fasta_path}...")
    sequences = load_fasta(fasta_path)
    print(f"  Loaded {len(sequences):,} sequences")

    # Check for checkpoint
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / f'{args.split}_checkpoint.npz'

    start_idx = 0
    all_embeddings = []
    all_ids = []

    if checkpoint_path.exists():
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        checkpoint = np.load(checkpoint_path, allow_pickle=True)
        all_embeddings = list(checkpoint['embeddings'])
        all_ids = list(checkpoint['ids'])
        start_idx = len(all_ids)
        print(f"  Loaded {start_idx} embeddings, resuming...")

    # Process remaining sequences
    remaining = sequences[start_idx:]
    if len(remaining) == 0:
        print("All sequences already processed!")
    else:
        print(f"\nProcessing {len(remaining):,} sequences...")

        # Process in chunks for checkpointing
        chunk_size = args.checkpoint_interval * args.batch_size

        for chunk_start in range(0, len(remaining), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(remaining))
            chunk = remaining[chunk_start:chunk_end]

            chunk_embeddings, chunk_ids = generate_embeddings(
                chunk, model, tokenizer, device,
                batch_size=args.batch_size,
                max_length=args.max_length
            )

            all_embeddings.append(chunk_embeddings)
            all_ids.extend(chunk_ids)

            # Save checkpoint
            np.savez(
                checkpoint_path,
                embeddings=np.vstack(all_embeddings),
                ids=np.array(all_ids)
            )
            print(f"  Checkpoint saved: {len(all_ids):,} embeddings")

    # Save final output
    final_embeddings = np.vstack(all_embeddings) if all_embeddings else np.array([])
    final_ids = np.array(all_ids)

    emb_path = OUTPUT_DIR / f'{args.split}_embeddings.npy'
    ids_path = OUTPUT_DIR / f'{args.split}_ids.npy'

    np.save(emb_path, final_embeddings)
    np.save(ids_path, final_ids)

    print(f"\n=== Done ===")
    print(f"Embeddings: {emb_path} ({final_embeddings.shape})")
    print(f"IDs: {ids_path} ({len(final_ids):,})")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Checkpoint cleaned up")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Balanced split of GNN datasets: train, val, test per path length (equal pos/neg for each path length).

Outputs:
  gnn_datasets/balanced/train_path_length_{N}.pt
  gnn_datasets/balanced/val_path_length_{N}.pt
  gnn_datasets/balanced/test_path_length_{N}.pt
  (plus corresponding *_info.json files)
"""

import os
import json
import random
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import sys

def save_dataset_info(dataset, out_prefix: str, *, path_length=None):
    pos = sum(1 for d in dataset if d.y.item() == 1)
    neg = len(dataset) - pos
    info = {
        "num_samples": len(dataset),
        "positive_samples": pos,
        "negative_samples": neg,
        "class_ratio": round(pos / (pos + neg), 3) if (pos + neg) else None,
    }
    if path_length is not None:
        info["path_length"] = path_length
    with open(f"{out_prefix}_info.json", "w") as f:
        json.dump(info, f, indent=4)

def print_dataset_statistics(dataset, name="Dataset"):
    print(f"\n{name} Statistics:")
    print(f"  Total samples: {len(dataset)}")
    buckets = {}
    for d in dataset:
        length = d.edge_index.shape[1] + 1
        buckets.setdefault(length, {"pos": 0, "neg": 0})
        if d.y.item() == 1:
            buckets[length]["pos"] += 1
        else:
            buckets[length]["neg"] += 1
    for length, c in sorted(buckets.items()):
        total = c["pos"] + c["neg"]
        ratio = c["pos"] / total if total else 0
        print(f"  path_length={length}:  pos={c['pos']}  neg={c['neg']}  ratio={ratio:.2f}")

def balance_dataset(dataset):
    """Down-sample to equal pos/neg counts for this path length."""
    pos = [d for d in dataset if d.y.item() == 1]
    neg = [d for d in dataset if d.y.item() == 0]
    if not pos or not neg:
        print("WARNING: Missing class, skipping balancing.")
        return []
    n = min(len(pos), len(neg))
    if n < 100:
        print(f"WARNING: Very few samples after balancing: {n}")
    balanced = random.sample(pos, n) + random.sample(neg, n)
    random.shuffle(balanced)
    return balanced

def main():
    # Set up logging to file and console
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w", encoding="utf-8")
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    sys.stdout = Logger("stat.txt")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    os.makedirs("gnn_datasets/balanced", exist_ok=True)

    for length in range(3, 7):
        pt_path = f"gnn_datasets/path_length_{length}.pt"
        if not os.path.exists(pt_path):
            print(f"File not found: {pt_path}, skipping.")
            continue
        data_list = torch.load(pt_path, weights_only=False)
        print(f"Loaded {len(data_list)} samples from {pt_path}")
        print_dataset_statistics(data_list, f"Original (path_length={length})")

        # Balance the data
        pos = [d for d in data_list if d.y.item() == 1]
        neg = [d for d in data_list if d.y.item() == 0]
        if not pos or not neg:
            print("WARNING: Missing class, skipping balancing.")
            continue
        n = min(len(pos), len(neg))
        if n < 5:
            print(f"Not enough balanced samples for path_length={length}, skipping split.")
            continue
        pos = random.sample(pos, n) if len(pos) > n else pos
        neg = random.sample(neg, n)
        balanced_data = pos + neg
        random.shuffle(balanced_data)
        print_dataset_statistics(balanced_data, f"Balanced (path_length={length})")

        # Split positives and negatives separately
        train_pos, test_pos = train_test_split(pos, test_size=0.2, random_state=42)
        train_pos, val_pos = train_test_split(train_pos, test_size=0.2, random_state=42)

        train_neg, test_neg = train_test_split(neg, test_size=0.2, random_state=42)
        train_neg, val_neg = train_test_split(train_neg, test_size=0.2, random_state=42)

        train = train_pos + train_neg
        val = val_pos + val_neg
        test = test_pos + test_neg
        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)

        # Print statistics for each split
        print_dataset_statistics(train, f"Train (path_length={length})")
        print_dataset_statistics(val, f"Val (path_length={length})")
        print_dataset_statistics(test, f"Test (path_length={length})")

        # Save splits
        torch.save(train, f"gnn_datasets/balanced/train_path_length_{length}.pt")
        save_dataset_info(train, out_prefix=f"gnn_datasets/balanced/train_path_length_{length}", path_length=length)

        torch.save(val, f"gnn_datasets/balanced/val_path_length_{length}.pt")
        save_dataset_info(val, out_prefix=f"gnn_datasets/balanced/val_path_length_{length}", path_length=length)

        torch.save(test, f"gnn_datasets/balanced/test_path_length_{length}.pt")
        save_dataset_info(test, out_prefix=f"gnn_datasets/balanced/test_path_length_{length}", path_length=length)

        print(f"  Saved balanced splits for path_length={length}: train={len(train)}, val={len(val)}, test={len(test)}")

    # Combine all path lengths into single train, val, test files
    all_train, all_val, all_test = [], [], []
    for length in range(3, 7):
        train_path = f"gnn_datasets/balanced/train_path_length_{length}.pt"
        val_path = f"gnn_datasets/balanced/val_path_length_{length}.pt"
        test_path = f"gnn_datasets/balanced/test_path_length_{length}.pt"
        if os.path.exists(train_path):
            all_train.extend(torch.load(train_path, weights_only=False))
        if os.path.exists(val_path):
            all_val.extend(torch.load(val_path, weights_only=False))
        if os.path.exists(test_path):
            all_test.extend(torch.load(test_path, weights_only=False))
    # Shuffle combined splits
    random.shuffle(all_train)
    random.shuffle(all_val)
    random.shuffle(all_test)
    # Save combined splits
    torch.save(all_train, "gnn_datasets/balanced/train.pt")
    save_dataset_info(all_train, out_prefix="gnn_datasets/balanced/train")
    torch.save(all_val, "gnn_datasets/balanced/val.pt")
    save_dataset_info(all_val, out_prefix="gnn_datasets/balanced/val")
    torch.save(all_test, "gnn_datasets/balanced/test.pt")
    save_dataset_info(all_test, out_prefix="gnn_datasets/balanced/test")
    print("\nSaved combined balanced train.pt, val.pt, test.pt in gnn_datasets/balanced/")

    # Print summary statistics for each path length
    print("\n===== Per-Path-Length Balanced Dataset Statistics =====")
    for length in range(3, 7):
        stats = {}
        for split in ['train', 'val', 'test']:
            pt_path = f"gnn_datasets/balanced/{split}_path_length_{length}.pt"
            if os.path.exists(pt_path):
                data_list = torch.load(pt_path, weights_only=False)
                pos = sum(1 for d in data_list if d.y.item() == 1)
                neg = sum(1 for d in data_list if d.y.item() == 0)
                total = len(data_list)
                ratio = round(pos / total, 3) if total else None
                stats[split] = (total, pos, neg, ratio)
            else:
                stats[split] = (0, 0, 0, None)
        print(f"\nPath length {length}:")
        for split in ['train', 'val', 'test']:
            total, pos, neg, ratio = stats[split]
            print(f"  {split.capitalize()}: total={total}, pos={pos}, neg={neg}, class_ratio={ratio}")
    print("\n===== End of Statistics =====\n")

if __name__ == "__main__":
    main() 
import os
import json
import random
import sys
from typing import List

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

# =====================================================
# Utility functions
# =====================================================

def save_dataset_info(dataset: List[Data], out_prefix: str, *, path_length=None):
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


def print_dataset_statistics(dataset: List[Data], name="Dataset"):
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


def main():
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w", encoding="utf-8")

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    sys.stdout = Logger("stat_balanced_split.txt")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    base_dir = "gnn_datasets_split"
    out_dir = os.path.join(base_dir, "balanced")
    os.makedirs(out_dir, exist_ok=True)

    # We balance both TRAIN and TEST splits. Validation is created from the balanced TRAIN set.
    train_dir = os.path.join(base_dir, "train")

    if not os.path.exists(train_dir):
        print(f"Train directory not found: {train_dir}")
        return

    for length in range(3, 7):
        pt_path = os.path.join(train_dir, f"path_length_{length}.pt")
        if not os.path.exists(pt_path):
            print(f"File not found: {pt_path}, skipping.")
            continue
        data_list: List[Data] = torch.load(pt_path, weights_only=False)
        print(f"Loaded {len(data_list)} samples from {pt_path}")
        print_dataset_statistics(data_list, f"Original TRAIN (path_length={length})")

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
        print_dataset_statistics(balanced_data, f"Balanced TRAIN (path_length={length})")

        # Split positives and negatives separately for balanced TRAIN set
        train_pos, val_pos = train_test_split(pos, test_size=0.2, random_state=42)
        train_neg, val_neg = train_test_split(neg, test_size=0.2, random_state=42)

        train_split = train_pos + train_neg
        val_split = val_pos + val_neg
        random.shuffle(train_split)
        random.shuffle(val_split)

        # Prepare output directory per path length
        split_out_dir = os.path.join(out_dir, f"path_length_{length}")
        os.makedirs(split_out_dir, exist_ok=True)

        # ---------- Save TRAIN & VAL ----------
        torch.save(train_split, os.path.join(split_out_dir, "train.pt"))
        save_dataset_info(train_split, out_prefix=os.path.join(split_out_dir, "train"), path_length=length)

        torch.save(val_split, os.path.join(split_out_dir, "val.pt"))
        save_dataset_info(val_split, out_prefix=os.path.join(split_out_dir, "val"), path_length=length)

        # ---------- Balance and Save TEST ----------
        test_pt = os.path.join(base_dir, "test", f"path_length_{length}.pt")
        if os.path.exists(test_pt):
            test_data: List[Data] = torch.load(test_pt, weights_only=False)
            print_dataset_statistics(test_data, f"Original TEST (path_length={length})")

            test_pos = [d for d in test_data if d.y.item() == 1]
            test_neg = [d for d in test_data if d.y.item() == 0]
            if test_pos and test_neg:
                n_test = min(len(test_pos), len(test_neg))
                test_pos = random.sample(test_pos, n_test) if len(test_pos) > n_test else test_pos
                test_neg = random.sample(test_neg, n_test) if len(test_neg) > n_test else test_neg
                balanced_test = test_pos + test_neg
                random.shuffle(balanced_test)

                torch.save(balanced_test, os.path.join(split_out_dir, "test.pt"))
                save_dataset_info(balanced_test, out_prefix=os.path.join(split_out_dir, "test"), path_length=length)
                print_dataset_statistics(balanced_test, f"Balanced TEST (path_length={length})")
            else:
                print("WARNING: Missing class in TEST set, saving unbalanced copy.")
                torch.save(test_data, os.path.join(split_out_dir, "test.pt"))
                save_dataset_info(test_data, out_prefix=os.path.join(split_out_dir, "test"), path_length=length)
        else:
            print(f"File not found: {test_pt}, skipping TEST balancing.")

        print(f"  Saved balanced TRAIN/VAL/TEST for path_length={length}")

    print("\nBalancing completed. Output under gnn_datasets_split/balanced/")


if __name__ == "__main__":
    main()
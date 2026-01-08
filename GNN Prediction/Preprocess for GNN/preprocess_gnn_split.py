import os
import json
import random
import sys
from typing import List, Tuple, Dict, Set

import torch
from torch_geometric.data import Data
import argparse

# ==================== Helper Functions ====================

def read_triples(path: str) -> List[Tuple[str, str, int]]:
    """Read processed_triples.txt and return a list of (head, tail, relation_id)."""
    with open(path, "r", encoding="utf-8") as f:
        total = int(f.readline())
        triples: List[Tuple[str, str, int]] = []
        for _ in range(total):
            h, t, r = f.readline().strip().split()
            triples.append((h, t, int(r)))
    return triples


def load_mappings() -> Tuple[Dict[int, str], Dict[int, str]]:
    """Load relation2id and entity2id mappings.

    The function searches for the mapping files in several likely locations so that
    users can run this script from different directories without having to
    manually tweak paths. The first matching pair of files will be loaded.
    """

    # Potential base directories that might contain the mapping txt files.
    # Ordered from most-specific to most-generic.
    candidate_dirs = [
        "MT40KG",  # original expected location (kept for backward compatibility)
        os.path.join(os.path.dirname(__file__), "..", "MT40K"),  # preprocess/MT40K
        "preprocess/MT40K",  # workspace root relative
        os.path.join(os.path.dirname(__file__), "MT40K"),  # GNN/MT40K alongside this script
        os.path.join(os.path.dirname(__file__), "..", "..", "MT40K"),  # ../../MT40K
        os.path.join(os.path.dirname(__file__), "..", "..", "MT40KG"),
        # Other dataset locations (e.g. MalKG-1) can be added here if needed
    ]

    relation_path = entity_path = None
    for base in candidate_dirs:
        rel_p = os.path.join(base, "relation2id.txt")
        ent_p = os.path.join(base, "entity2id.txt")
        if os.path.exists(rel_p) and os.path.exists(ent_p):
            relation_path, entity_path = rel_p, ent_p
            break

    if relation_path is None or entity_path is None:
        raise FileNotFoundError(
            "Could not locate relation2id.txt and entity2id.txt in expected locations. "
            "Please update `candidate_dirs` in load_mappings() or specify paths explicitly."
        )

    relation2id: Dict[int, str] = {}
    with open(relation_path, "r", encoding="utf-8") as f:
        f.readline()  # first line count
        for line in f:
            rel, rel_id = line.strip().split("\t")
            relation2id[int(rel_id)] = rel

    entity2id: Dict[int, str] = {}
    with open(entity_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            name, eid = line.strip().split("\t")
            entity2id[int(eid)] = name

    return relation2id, entity2id


def build_graph(triples: List[Tuple[str, str, int]]) -> Tuple[Dict[str, Dict[str, int]], Set[str]]:
    """Build adjacency dict graph from triples."""
    graph: Dict[str, Dict[str, int]] = {}
    nodes: Set[str] = set()
    for h, t, r in triples:
        nodes.add(h)
        nodes.add(t)
        if h not in graph:
            graph[h] = {}
        graph[h][t] = r
    return graph, nodes


def generate_all_paths(graph: Dict[str, Dict[str, int]], max_length: int):
    """Generate all simple paths in directed graph up to max_length."""

    def dfs(current: str, path: List[str], length: int):
        if length > max_length:
            return
        if 3 <= length <= max_length:
            paths[length].append(path.copy())
        for nbr in graph.get(current, {}):
            if nbr not in path:
                path.append(nbr)
                dfs(nbr, path, length + 1)
                path.pop()

    paths = {i: [] for i in range(2, max_length + 1)}
    for node in graph:
        dfs(node, [node], 1)
    return paths


def create_gnn_dataset_for_length(paths_dict, graph, length, node_list):
    path_list = paths_dict[length]
    dataset = []

    num_nodes = len(node_list)
    node_features = torch.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        node_features[i, i] = 1.0

    for path in path_list:
        edge_index = []
        edge_attr = []
        for i in range(len(path) - 1):
            h = int(path[i])
            t = int(path[i + 1])
            edge_index.append([h, t])
            edge_attr.append(graph[path[i]][path[i + 1]])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)

        first_node, last_node = path[0], path[-1]
        if last_node in graph.get(first_node, {}):
            label = 1
            direct_relation = graph[first_node][last_node]
        else:
            label = 0
            direct_relation = -1

        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(label, dtype=torch.float),
            num_nodes=num_nodes,
        )
        data.direct_relation = direct_relation
        dataset.append(data)
    return dataset


def save_dataset_info(dataset, out_path: str, length: int, num_nodes: int, num_relations: int):
    info = {
        "num_samples": len(dataset),
        "num_nodes": num_nodes,
        "num_relations": num_relations,
        "path_length": length,
        "positive_samples": sum(1 for d in dataset if d.y.item() == 1),
        "negative_samples": sum(1 for d in dataset if d.y.item() == 0),
    }
    with open(out_path, "w") as f:
        json.dump(info, f, indent=4)


# ==================== Main ====================

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

    sys.stdout = Logger("preprocess_split_log.txt")

    parser = argparse.ArgumentParser(description="Split triples into train/test and generate GNN datasets.")
    parser.add_argument("--triples", type=str, default=None, help="Path to processed_triples.txt")
    args = parser.parse_args()

    # Determine triples path
    if args.triples and os.path.exists(args.triples):
        triples_path = args.triples
    else:
        # Try common relative locations
        candidate_paths = [
            "MT40KG/processed_triples.txt",
            "preprocess/MT40K/processed_triples.txt",  # common location when running from repo root
            os.path.join(os.path.dirname(__file__), "..", "MT40K", "processed_triples.txt"),
            os.path.join(os.path.dirname(__file__), "MT40KG", "processed_triples.txt"),
            os.path.join(os.path.dirname(__file__), "..", "MT40KG", "processed_triples.txt"),
        ]
        triples_path = next((p for p in candidate_paths if os.path.exists(p)), None)
        if triples_path is None:
            raise FileNotFoundError("Could not locate processed_triples.txt. Pass --triples PATH explicitly.")

    print(f"Reading triples from {triples_path}…")
    triples = read_triples(triples_path)
    print(f"Total triples: {len(triples)}")

    # Shuffle and split
    random.shuffle(triples)
    split_idx = int(0.8 * len(triples))
    train_triples = triples[:split_idx]
    test_triples = triples[split_idx:]
    print(f"Train triples: {len(train_triples)}  |  Test triples: {len(test_triples)}")

    # Build graphs
    train_graph, train_nodes = build_graph(train_triples)
    test_graph, test_nodes = build_graph(test_triples)
    all_nodes = sorted(train_nodes.union(test_nodes))
    num_nodes = len(all_nodes)

    # Load mappings
    relation2id, entity2id = load_mappings()
    num_relations = len(relation2id)

    out_root = "gnn_datasets_split"
    os.makedirs(out_root, exist_ok=True)
    max_len = 6

    for split_name, graph in [("train", train_graph), ("test", test_graph)]:
        print(f"\n=== Processing {split_name} split ===")
        split_dir = os.path.join(out_root, split_name)
        os.makedirs(split_dir, exist_ok=True)

        paths_dict = generate_all_paths(graph, max_len)
        split_datasets = {}
        total_samples = 0
        for length in range(3, max_len + 1):
            print(f"  Generating datasets for path length {length}…")
            ds = create_gnn_dataset_for_length(paths_dict, graph, length, all_nodes)
            split_datasets[length] = ds
            total_samples += len(ds)

            pt_path = os.path.join(split_dir, f"path_length_{length}.pt")
            torch.save(ds, pt_path)
            save_dataset_info(ds, os.path.join(split_dir, f"path_length_{length}_info.json"), length, num_nodes, num_relations)
            print(f"    Saved {len(ds)} samples → {pt_path}")

        # Save combined dataset for split
        combined = [d for lst in split_datasets.values() for d in lst]
        torch.save(combined, os.path.join(split_dir, "all_paths.pt"))
        print(f"  Total samples for {split_name}: {total_samples}")

    print("\nDataset generation (train/test) completed!")


if __name__ == "__main__":
    main()
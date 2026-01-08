import pandas as pd
import json
from collections import deque
import csv
import os
import torch
import numpy as np
from torch_geometric.data import Data
import sys

def load_graph_data():
    """Load graph data from files"""
    input_file = open(r"MT40KG/processed_triples.txt", "r")
    number = int(input_file.readline())

    nodes = set()
    graph = {}

    for i in range(number):
        content = input_file.readline()
        node1, node2, relation = content.strip().split()
        nodes.add(node1)
        relation = int(relation)

        if node1 not in graph:
            graph[node1] = {}
        graph[node1][node2] = relation

    node_list = list(nodes)
    
    # Load relation mappings
    relation2id = {}
    with open(r"MT40KG/relation2id.txt", "r", encoding="utf-8") as file:
        relations = int(file.readline())
        for line in file:
            relation, relation_id = line.strip().split("\t")
            relation2id[int(relation_id)] = relation

    # Load entity mappings
    entity2id = {}
    with open(r"MT40KG/entity2id.txt", "r", encoding="utf-8") as file:
        file.readline()  # Skip the first line
        for line in file:
            entity_name, entity_id = line.strip().split("\t")
            entity2id[int(entity_id)] = entity_name

    return graph, node_list, relation2id, entity2id

def generate_all_paths(graph, max_length):
    """Generate all possible paths up to max_length"""
    def dfs(current_node, path, length):
        if length > max_length:
            return
        if 3 <= length <= max_length:
            paths[length].append(path.copy())
        for neighbor in graph.get(current_node, {}):
            if neighbor not in path:
                path.append(neighbor)
                dfs(neighbor, path, length + 1)
                path.pop()

    paths = {i: [] for i in range(2, max_length + 1)}
    for node in graph:
        dfs(node, [node], 1)
    return paths

def create_gnn_dataset_for_length(paths, graph, length, node_list):
    """Create GNN dataset for a specific path length"""
    path_list = paths[length]
    dataset = []
    
    # Create node features (one-hot encoding)
    num_nodes = len(node_list)
    node_features = torch.zeros((num_nodes, num_nodes))
    for i, node in enumerate(sorted(node_list)):
        node_features[i][i] = 1
    
    # Process paths of the specified length
    for path in path_list:
        # Create edge indices and attributes for this path
        edge_index = []
        edge_attr = []
        
        # Add edges between consecutive nodes in the path
        for i in range(len(path) - 1):
            node1 = int(path[i])
            node2 = int(path[i + 1])
            edge_index.append([node1, node2])
            edge_attr.append(graph[path[i]][path[i + 1]])
        
        # Convert to tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        
        # Create label (1 if there's a direct connection between first and last node)
        first_node = path[0]
        last_node = path[-1]
        if last_node in graph.get(first_node, {}):
            label = 1
            direct_relation = graph[first_node][last_node]
        else:
            label = 0
            direct_relation = -1
        
        # Create PyTorch Geometric Data object
        data = Data(
            x=node_features,  # Node features
            edge_index=edge_index,  # Graph connectivity for this path
            edge_attr=edge_attr,  # Edge features for this path
            y=torch.tensor(label, dtype=torch.float),  # Label
            num_nodes=num_nodes
        )
        data.direct_relation = direct_relation
        dataset.append(data)
    
    return dataset

def save_dataset_info(datasets, node_list, relation2id):
    """Save information about the created datasets"""
    overall_info = {
        'num_nodes': len(node_list),
        'num_relations': len(relation2id),
        'datasets': {
            length: {
                'num_samples': len(dataset),
                'positive_samples': sum(1 for data in dataset if data.y.item() == 1),
                'negative_samples': sum(1 for data in dataset if data.y.item() == 0)
            }
            for length, dataset in datasets.items()
        }
    }
    
    with open('gnn_datasets/overall_info.json', 'w') as f:
        json.dump(overall_info, f, indent=4)

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
    sys.stdout = Logger("preprocess_log.txt")

    # Create output directory
    os.makedirs('gnn_datasets', exist_ok=True)
    
    # Load graph data
    print("Loading graph data...")
    graph, node_list, relation2id, entity2id = load_graph_data()
    
    # Generate paths
    print("Generating paths...")
    max_path_length = 6
    paths = generate_all_paths(graph, max_path_length)
    
    # Create and save datasets for each path length
    datasets = {}
    for length in range(3, max_path_length + 1):
        print(f"\nProcessing paths of length {length}...")
        dataset = create_gnn_dataset_for_length(paths, graph, length, node_list)
        datasets[length] = dataset
        
        # Save dataset
        torch.save(dataset, f'gnn_datasets/path_length_{length}.pt')
        
        # Save dataset info
        dataset_info = {
            'num_samples': len(dataset),
            'num_nodes': len(node_list),
            'num_relations': len(relation2id),
            'path_length': length,
            'positive_samples': sum(1 for data in dataset if data.y.item() == 1),
            'negative_samples': sum(1 for data in dataset if data.y.item() == 0)
        }
        
        with open(f'gnn_datasets/path_length_{length}_info.json', 'w') as f:
            json.dump(dataset_info, f, indent=4)
        
        print(f"Created dataset with {len(dataset)} samples")
        print(f"Positive samples: {sum(1 for data in dataset if data.y.item() == 1)}")
        print(f"Negative samples: {sum(1 for data in dataset if data.y.item() == 0)}")
    
    # Save overall dataset info
    save_dataset_info(datasets, node_list, relation2id)

    # Save all paths in a single .pt file
    all_data = []
    for length in range(3, max_path_length + 1):
        all_data.extend(datasets[length])
    torch.save(all_data, 'gnn_datasets/all_paths.pt')
    print(f"\nSaved all paths (all lengths) to gnn_datasets/all_paths.pt. Total samples: {len(all_data)}")

    print("\nDataset creation completed!")
    print(f"Total number of nodes: {len(node_list)}")
    print(f"Total number of relations: {len(relation2id)}")
    print("\nSamples per path length:")
    for length, dataset in datasets.items():
        print(f"Length {length}: {len(dataset)} samples")

if __name__ == "__main__":
    main() 
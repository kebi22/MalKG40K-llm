import torch
import csv
import os
from torch_geometric.data import Data

def pt_to_csv(pt_path, csv_path):
    if not os.path.exists(pt_path):
        print(f"File not found: {pt_path}")
        return
    data_list = torch.load(pt_path, weights_only=False)
    print(f"Loaded {len(data_list)} samples from {pt_path}")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['path_nodes', 'label', 'path_length'])
        for data in data_list:
            edge_index = data.edge_index.cpu().numpy()
            edge_attr = data.edge_attr.cpu().numpy() if hasattr(data, 'edge_attr') else []
            path_nodes = [edge_index[0][0]]
            for i in range(edge_index.shape[1]):
                path_nodes.append(edge_index[1][i])
            path_nodes_str = '-'.join(str(n) for n in path_nodes)
            edge_indices_str = ';'.join(f"({edge_index[0][i]},{edge_index[1][i]})" for i in range(edge_index.shape[1]))
            # Extract relations from edge attributes if available
            relations = edge_attr if edge_attr is not None and len(edge_attr) > 0 else ['None'] * edge_index.shape[1]
            # Determine label based on positive or negative
            if hasattr(data, 'y') and data.y.item() > 0:
                # Use the relation ID between the first and last node for positive
                label = relations[-1] if len(relations) > 0 else 'None'
            else:
                # Use 0 for negative labels
                label = 0
            path_length = len(path_nodes)
            # Extract relations from edge attributes if available
            # Check if edge_attr is not None and has elements
            if edge_attr is not None and len(edge_attr) > 0:
                relations = edge_attr
            else:
                relations = ['None'] * edge_index.shape[1]
            relations_str = ';'.join(str(r) for r in relations)
            # Create path_nodes with inline relations
            path_nodes_with_relations = []
            for i in range(edge_index.shape[1]):
                path_nodes_with_relations.append(str(edge_index[0][i]))
                if i < len(relations):
                    path_nodes_with_relations.append(str(relations[i]))
            path_nodes_with_relations.append(str(edge_index[1][-1]))
            path_nodes_str = '-'.join(path_nodes_with_relations)
            # Write the row with path_nodes, label, and path length
            writer.writerow([path_nodes_str, label, path_length])
    print(f"Saved CSV to {csv_path}")

def main():
    base = '/Users/kbahlibi/Library/CloudStorage/GoogleDrive-kbahlibi@angelo.edu/My Drive/Malware KG/More_Detailed_path_information/unbalanced/'
    for split in ['train', 'val', 'test']:
        pt_path = os.path.join(base, f'{split}.pt')
        csv_path = os.path.join(base, f'{split}.csv')
        pt_to_csv(pt_path, csv_path)

if __name__ == "__main__":
    main() 
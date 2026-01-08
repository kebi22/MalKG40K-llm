import torch
import csv
import os
from torch_geometric.data import Data

def pt_to_triples_csv(pt_path, csv_path):
    if not os.path.exists(pt_path):
        print(f"File not found: {pt_path}")
        return
    data_list = torch.load(pt_path, weights_only=False)
    print(f"Loaded {len(data_list)} samples from {pt_path}")
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['source', 'relation', 'target', 'path_length', 'label'])
        for data in data_list:
            edge_index = data.edge_index.cpu().numpy()
            edge_attr = data.edge_attr.cpu().numpy() if hasattr(data, 'edge_attr') else []
            path_length = edge_index.shape[1] + 1
            label = int(data.y.item())
            for i in range(edge_index.shape[1]):
                src = edge_index[0][i]
                dst = edge_index[1][i]
                rel = edge_attr[i] if len(edge_attr) > i else ''
                # Only last triple gets the label
                row_label = label if i == edge_index.shape[1] - 1 else ''
                writer.writerow([src, rel, dst, path_length, row_label])
            # For positive samples, add the direct triple row
            if label == 1 and hasattr(data, 'direct_relation') and data.direct_relation != -1:
                first_node = edge_index[0][0]
                last_node = edge_index[1][-1]
                direct_rel = data.direct_relation
                writer.writerow([first_node, direct_rel, last_node, path_length, 1])
            # For negative samples, add the direct triple row with label 0
            if label == 0:
                first_node = edge_index[0][0]
                last_node = edge_index[1][-1]
                writer.writerow([first_node, '', last_node, path_length, 0])
    print(f"Saved triples CSV to {csv_path}")

def main():
    base = 'gnn_datasets/balanced/'
    for split in ['train', 'val', 'test']:
        pt_path = os.path.join(base, f'{split}.pt')
        csv_path = os.path.join(base, f'{split}_triples.csv')
        pt_to_triples_csv(pt_path, csv_path)
    
if __name__ == "__main__":

    main() 
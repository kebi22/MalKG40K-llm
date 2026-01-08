#!/usr/bin/env python
"""
Per-Hop Evaluation Script for Trained GNN Model
==============================================

This script loads a trained GNN model and evaluates its performance
separately for each path length (3, 4, 5, 6) without retraining.
"""

import argparse
import logging
import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, recall_score
import sys
from pathlib import Path

# ───────────────────────────── Logging ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)5s | %(message)s",
    handlers=[logging.FileHandler("per_hop_evaluation.log"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ─────────────────────────── GNN Models ─────────────────────────────
class GNNLinkPredictor(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_layers=2, model_type='sage'):
        super().__init__()
        self.num_layers = num_layers
        self.model_type = model_type
        
        # GNN layers
        if model_type == 'sage':
            self.convs = torch.nn.ModuleList()
            self.convs.append(SAGEConv(num_node_features, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        elif model_type == 'gat':
            self.convs = torch.nn.ModuleList()
            self.convs.append(GATConv(num_node_features, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(GATConv(hidden_channels, hidden_channels))
        elif model_type == 'gcn':
            self.convs = torch.nn.ModuleList()
            self.convs.append(GCNConv(num_node_features, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Link prediction head
        self.link_predictor = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * 2, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_channels, hidden_channels // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_channels // 2, 1)
        )
        
    def encode(self, x, edge_index, edge_attr=None):
        for i, conv in enumerate(self.convs):
            # GraphSAGE and GCN don't support edge attributes, so we ignore them
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.2, training=self.training)
        return x
    
    def decode(self, z, edge_label_index):
        # Get node embeddings for the edge endpoints
        row, col = edge_label_index
        z_row = z[row]
        z_col = z[col]
        
        # Concatenate and predict
        z_cat = torch.cat([z_row, z_col], dim=1)
        return self.link_predictor(z_cat).squeeze()
    
    def forward(self, x, edge_index, edge_label_index, edge_attr=None):
        z = self.encode(x, edge_index, edge_attr)
        return self.decode(z, edge_label_index)
    
    def forward_path_prediction(self, x, edge_index, edge_attr=None):
        """Forward pass for path-based link prediction with local node remapping."""
        # Get unique node indices in this path
        unique_nodes, edge_index_local = torch.unique(edge_index, return_inverse=True)
        edge_index_local = edge_index_local.view(edge_index.size())
        # Select node features for just these nodes
        x_local = x[unique_nodes]
        # Encode the subgraph
        z = self.encode(x_local, edge_index_local)
        # The first and last node in the path (in local indices)
        first_node_global = edge_index[0, 0].item()
        last_node_global = edge_index[1, -1].item()
        first_node_local = (unique_nodes == first_node_global).nonzero(as_tuple=True)[0].item()
        last_node_local = (unique_nodes == last_node_global).nonzero(as_tuple=True)[0].item()
        z_first = z[first_node_local]
        z_last = z[last_node_local]
        z_cat = torch.cat([z_first, z_last], dim=0)
        return self.link_predictor(z_cat.unsqueeze(0)).view(-1)

# ─────────────────────────── Data Loading ───────────────────────────
def load_test_datasets(data_dir, path_lengths=None):
    """Load test datasets for evaluation"""
    if path_lengths is None:
        path_lengths = [3, 4, 5, 6]
    
    test_data = []
    
    # Load path-specific test datasets
    log.info("Loading path-specific test datasets")
    for length in path_lengths:
        test_path = os.path.join(data_dir, f"test_path_length_{length}.pt")
        if os.path.exists(test_path):
            test_data.extend(torch.load(test_path, weights_only=False))
            log.info(f"Loaded test data for path length {length}: {len(torch.load(test_path, weights_only=False))} samples")
    
    # Shuffle test data
    random.shuffle(test_data)
    
    return test_data

def print_dataset_stats(dataset, name):
    """Print statistics about a dataset"""
    if not dataset:
        log.info(f"{name}: Empty dataset")
        return
    
    pos_count = sum(1 for data in dataset if data.y.item() == 1)
    neg_count = len(dataset) - pos_count
    pos_ratio = pos_count / len(dataset) if len(dataset) > 0 else 0
    
    log.info(f"{name}: {len(dataset)} samples, {pos_count} pos, {neg_count} neg, {pos_ratio:.3f} pos ratio")
    
    # Path length distribution
    length_counts = {}
    for data in dataset:
        length = data.edge_index.shape[1] + 1
        length_counts[length] = length_counts.get(length, 0) + 1
    
    log.info(f"{name} path lengths: {dict(sorted(length_counts.items()))}")

# ─────────────────────────── Evaluation ─────────────────────────────
def evaluate_by_path_length(model, loader, device, name="test"):
    """Evaluate separately for each path length"""
    path_length_results = {}
    
    # Group data by path length
    path_length_data = {}
    for batch in loader:
        # Process each sample in the batch individually
        batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
        
        for i in range(batch_size):
            # Get individual sample
            if batch_size == 1:
                sample = batch
            else:
                sample = batch[i]
            
            path_length = sample.edge_index.shape[1] + 1
            if path_length not in path_length_data:
                path_length_data[path_length] = []
            path_length_data[path_length].append(sample)
    
    # Evaluate each path length separately
    for length, samples in path_length_data.items():
        log.info(f"Evaluating {name} path length {length}...")
        
        all_preds = []
        all_labels = []
        all_pred_classes = []
        
        with torch.no_grad():
            for sample in samples:
                sample = sample.to(device)
                
                # Forward pass for path-based prediction
                out = model.forward_path_prediction(sample.x, sample.edge_index, sample.edge_attr)
                
                # Check for NaN in output and handle
                if torch.isnan(out).any():
                    log.warning(f"NaN detected in model output during per-hop evaluation, skipping sample")
                    continue
                
                # Predictions
                preds = torch.sigmoid(out)
                pred_classes = (preds >= 0.5).float()
                
                # Handle scalar y values
                if sample.y.dim() == 0:
                    all_preds.append(preds.cpu().item())
                    all_labels.append(sample.y.cpu().item())
                    all_pred_classes.append(pred_classes.cpu().item())
                else:
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(sample.y.cpu().numpy())
                    all_pred_classes.extend(pred_classes.cpu().numpy())
        
        if len(all_preds) > 0:
            # Filter out NaN values
            valid_indices = ~np.isnan(all_preds)
            if np.sum(valid_indices) > 0:
                valid_preds = np.array(all_preds)[valid_indices]
                valid_labels = np.array(all_labels)[valid_indices]
                valid_pred_classes = np.array(all_pred_classes)[valid_indices]
                
                try:
                    auc = roc_auc_score(valid_labels, valid_preds)
                except ValueError:
                    auc = 0.5  # Default for edge cases
                
                try:
                    ap = average_precision_score(valid_labels, valid_preds)
                except ValueError:
                    ap = 0.5
                
                accuracy = accuracy_score(valid_labels, valid_pred_classes)
                f1 = f1_score(valid_labels, valid_pred_classes)
                precision = precision_score(valid_labels, valid_pred_classes)
                recall = recall_score(valid_labels, valid_pred_classes)
                
                path_length_results[length] = {
                    'auc': float(auc),
                    'ap': float(ap),
                    'accuracy': float(accuracy),
                    'f1': float(f1),
                    'precision': float(precision),
                    'recall': float(recall),
                    'num_samples': int(len(valid_preds)),
                    'positive_samples': int(sum(valid_labels)),
                    'negative_samples': int(len(valid_labels) - sum(valid_labels))
                }
                
                log.info(f"{name} path length {length}: AUC={auc:.4f}, AP={ap:.4f}, Acc={accuracy:.4f}, F1={f1:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, samples={len(valid_preds)}")
            else:
                log.warning(f"No valid predictions for path length {length}")
        else:
            log.warning(f"No predictions for path length {length}")
    
    return path_length_results

# ────────────────────────────── CLI ────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Per-Hop GNN Evaluation')
    parser.add_argument('--data_dir', default='More_Detailed_path_information/balanced',
                       help='Directory containing test datasets')
    parser.add_argument('--model_path', default='batch_test/best_model.pt',
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_type', default='sage', choices=['sage', 'gat', 'gcn'],
                       help='Type of GNN model')
    parser.add_argument('--hidden_channels', type=int, default=128,
                       help='Number of hidden channels')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of GNN layers')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--path_lengths', nargs='+', type=int, default=[3, 4, 5, 6],
                       help='Path lengths to evaluate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set up device for MacBook
    if torch.backends.mps.is_available():
        device = torch.device('mps')  # Apple Silicon GPU
        log.info("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')  # CPU fallback
        log.info("Using CPU")
    
    log.info(f"Device: {device}")
    
    # Load test datasets
    log.info("Loading test datasets...")
    test_data = load_test_datasets(args.data_dir, args.path_lengths)
    
    print_dataset_stats(test_data, "Test")
    
    if not test_data:
        log.error("Empty test dataset! Check data directory and path lengths.")
        return
    
    # Create test loader
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    # Get number of node features from first sample
    num_node_features = test_data[0].x.shape[1]
    log.info(f"Number of node features: {num_node_features}")
    
    # Create model
    model = GNNLinkPredictor(
        num_node_features=num_node_features,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        model_type=args.model_type
    )
    
    # Load trained model
    if os.path.exists(args.model_path):
        log.info(f"Loading model from: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        log.info(f"Model loaded successfully! Best validation AUC: {checkpoint.get('val_auc', 'N/A')}")
    else:
        log.error(f"Model file not found: {args.model_path}")
        return
    
    model = model.to(device)
    model.eval()
    
    # Evaluate by path length
    log.info("Starting per-hop evaluation...")
    path_length_results = evaluate_by_path_length(model, test_loader, device, "test")
    
    # Save results
    results_file = "per_hop_results.json"
    with open(results_file, 'w') as f:
        json.dump(path_length_results, f, indent=4)
    
    # Save results to resultsfinal.txt (table format)
    resultsfinal_file = "resultsfinal.txt"
    with open(resultsfinal_file, 'w') as f:
        f.write("PER-HOP EVALUATION RESULTS\n")
        f.write("============================\n\n")
        
        # Table header
        f.write("Path Length | Accuracy | F1      | Recall  | Precision | Samples\n")
        f.write("------------|----------|---------|---------|-----------|--------\n")
        
        # Table rows
        for length in sorted(path_length_results.keys()):
            metrics = path_length_results[length]
            f.write(f"{length:11d} | {metrics['accuracy']:8.4f} | {metrics['f1']:7.4f} | {metrics['recall']:7.4f} | {metrics['precision']:9.4f} | {metrics['num_samples']:7d}\n")
        
        f.write("\n")
    
    log.info(f"Per-hop evaluation completed!")
    log.info(f"Results saved to: {results_file}")
    log.info(f"Simple results saved to: {resultsfinal_file}")
    
    # Print summary
    log.info("\n" + "="*50)
    log.info("PER-HOP EVALUATION SUMMARY")
    log.info("="*50)
    for length, metrics in path_length_results.items():
        log.info(f"Path Length {length}:")
        log.info(f"  AUC: {metrics['auc']:.4f}")
        log.info(f"  AP: {metrics['ap']:.4f}")
        log.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        log.info(f"  F1: {metrics['f1']:.4f}")
        log.info(f"  Precision: {metrics['precision']:.4f}")
        log.info(f"  Recall: {metrics['recall']:.4f}")
        log.info(f"  Samples: {metrics['num_samples']}")
        log.info("")

if __name__ == "__main__":
    main() 
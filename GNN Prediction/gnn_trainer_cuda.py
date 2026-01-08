#!/usr/bin/env python
"""
Knowledge-Graph Link-Prediction GNN Trainer (CUDA Version)
=========================================================

* Graph Neural Network for link prediction optimized for CUDA servers
* Balanced multi-hop (3-6) paths from preprocessed datasets
* PyTorch Geometric with GraphSAGE/GAT architecture
* CUDA-optimized with improved memory management
* Ready to resume training from existing checkpoints
* Comprehensive logging and evaluation
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
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, recall_score
import sys
from pathlib import Path
import gc

# ───────────────────────────── Logging ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)5s | %(message)s",
    handlers=[logging.FileHandler("gnn_training_cuda.log"), logging.StreamHandler()],
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
def load_balanced_datasets(data_dir, path_lengths=None):
    """Load balanced train/val/test datasets"""
    if path_lengths is None:
        path_lengths = [3, 4, 5, 6]
    
    train_data, val_data, test_data = [], [], []
    
    # Load combined train and val datasets
    train_combined_path = os.path.join(data_dir, "train.pt")
    val_combined_path = os.path.join(data_dir, "val.pt")
    
    if os.path.exists(train_combined_path) and os.path.exists(val_combined_path):
        log.info("Loading combined train and val datasets (train.pt, val.pt)")
        train_data = torch.load(train_combined_path, weights_only=False)
        val_data = torch.load(val_combined_path, weights_only=False)
    else:
        # Fall back to path-specific train and val datasets
        log.info("Loading path-specific train and val datasets")
        for length in path_lengths:
            train_path = os.path.join(data_dir, f"train_path_length_{length}.pt")
            val_path = os.path.join(data_dir, f"val_path_length_{length}.pt")
            
            if os.path.exists(train_path):
                train_data.extend(torch.load(train_path, weights_only=False))
            if os.path.exists(val_path):
                val_data.extend(torch.load(val_path, weights_only=False))
    
    # Always load path-specific test datasets
    log.info("Loading path-specific test datasets")
    for length in path_lengths:
        test_path = os.path.join(data_dir, f"test_path_length_{length}.pt")
        if os.path.exists(test_path):
            test_data.extend(torch.load(test_path, weights_only=False))
            log.info(f"Loaded test data for path length {length}: {len(torch.load(test_path, weights_only=False))} samples")
    
    # Shuffle datasets
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data

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

# ─────────────────────────── Training ───────────────────────────────
class GNNTrainer:
    def __init__(self, model, train_loader, val_loader, test_loader, 
                 device, learning_rate=0.001, weight_decay=1e-5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        self.best_val_loss = float('inf')
        self.best_val_auc = 0.0
        self.patience_counter = 0
        self.patience = 20
        
        # CUDA memory management
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            log.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            log.info(f"CUDA memory cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Process each sample in the batch individually
            batch_loss = 0
            batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
            
            for i in range(batch_size):
                # Get individual sample
                if batch_size == 1:
                    sample = batch
                else:
                    sample = batch[i]
                
                # Forward pass for path-based prediction
                out = self.model.forward_path_prediction(sample.x, sample.edge_index, sample.edge_attr)
                
                # Binary cross entropy loss
                sample_loss = F.binary_cross_entropy_with_logits(out, sample.y.float())
                batch_loss += sample_loss
            
            # Average loss over batch
            avg_batch_loss = batch_loss / batch_size
            
            # Backward pass
            avg_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += avg_batch_loss.item()
            num_batches += 1
            
            # CUDA memory management - clear cache periodically
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def evaluate(self, loader, name="val"):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_pred_classes = []
        num_batches = 0
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                
                # Process each sample in the batch individually
                batch_loss = 0
                batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 1
                
                for i in range(batch_size):
                    # Get individual sample
                    if batch_size == 1:
                        sample = batch
                    else:
                        sample = batch[i]
                    
                    # Forward pass for path-based prediction
                    out = self.model.forward_path_prediction(sample.x, sample.edge_index, sample.edge_attr)
                    
                    # Loss
                    sample_loss = F.binary_cross_entropy_with_logits(out, sample.y.float())
                    batch_loss += sample_loss
                    
                    # Predictions
                    preds = torch.sigmoid(out)
                    pred_classes = (preds >= 0.5).float()  # Binary classification threshold
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(sample.y.cpu().numpy())
                    all_pred_classes.extend(pred_classes.cpu().numpy())
                
                total_loss += batch_loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Calculate metrics
        if len(all_preds) > 0:
            auc = roc_auc_score(all_labels, all_preds)
            ap = average_precision_score(all_labels, all_preds)
            accuracy = accuracy_score(all_labels, all_pred_classes)
            f1 = f1_score(all_labels, all_pred_classes)
            precision = precision_score(all_labels, all_pred_classes)
            recall = recall_score(all_labels, all_pred_classes)
        else:
            auc = ap = accuracy = f1 = precision = recall = 0.0
        
        log.info(f"{name} - Loss: {avg_loss:.4f}, AUC: {auc:.4f}, AP: {ap:.4f}")
        log.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        return avg_loss, auc, ap, accuracy, f1, precision, recall
    
    def evaluate_by_path_length(self, loader, name="test"):
        """Evaluate separately for each path length"""
        path_length_results = {}
        
        # Group data by path length
        path_length_data = {}
        for batch in loader:
            path_length = batch.edge_index.shape[1] + 1
            if path_length not in path_length_data:
                path_length_data[path_length] = []
            path_length_data[path_length].append(batch)
        
        # Evaluate each path length separately
        for length, batches in path_length_data.items():
            log.info(f"Evaluating {name} path length {length}...")
            
            # Create a new loader for this path length
            length_loader = DataLoader(batches, batch_size=32, shuffle=False)
            
            all_preds = []
            all_labels = []
            all_pred_classes = []
            
            with torch.no_grad():
                for batch in length_loader:
                    batch = batch.to(self.device)
                    
                    # Forward pass for path-based prediction
                    out = self.model.forward_path_prediction(batch.x, batch.edge_index, batch.edge_attr)
                    
                    # Predictions
                    preds = torch.sigmoid(out)
                    pred_classes = (preds >= 0.5).float()
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(batch.y.cpu().numpy())
                    all_pred_classes.extend(pred_classes.cpu().numpy())
            
            if len(all_preds) > 0:
                auc = roc_auc_score(all_labels, all_preds)
                ap = average_precision_score(all_labels, all_preds)
                accuracy = accuracy_score(all_labels, all_pred_classes)
                f1 = f1_score(all_labels, all_pred_classes)
                precision = precision_score(all_labels, all_pred_classes)
                recall = recall_score(all_labels, all_pred_classes)
                
                path_length_results[length] = {
                    'auc': auc,
                    'ap': ap,
                    'accuracy': accuracy,
                    'f1': f1,
                    'precision': precision,
                    'recall': recall,
                    'num_samples': len(all_preds),
                    'positive_samples': sum(all_labels),
                    'negative_samples': len(all_labels) - sum(all_labels)
                }
                
                log.info(f"{name} path length {length}: AUC={auc:.4f}, AP={ap:.4f}, Acc={accuracy:.4f}, F1={f1:.4f}, Prec={precision:.4f}, Rec={recall:.4f}, samples={len(all_preds)}")
        
        return path_length_results
    
    def train(self, num_epochs, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
        log.info(f"Starting training for {num_epochs} epochs")
        log.info(f"Model: {self.model.model_type}, Device: {self.device}")
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch()
            
            # Validation
            val_loss, val_auc, val_ap, val_acc, val_f1, val_prec, val_rec = self.evaluate(self.val_loader, "val")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Logging
            if epoch % 10 == 0:
                log.info(f"Epoch {epoch:3d} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
                
                # CUDA memory logging
                if torch.cuda.is_available():
                    log.info(f"CUDA memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB allocated, {torch.cuda.memory_reserved() / 1024**3:.2f} GB cached")
            
            # Save best model
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_auc': val_auc,
                    'val_ap': val_ap,
                    'val_accuracy': val_acc,
                    'val_f1': val_f1,
                    'val_precision': val_prec,
                    'val_recall': val_rec,
                }, os.path.join(save_dir, 'best_model.pt'))
                
                log.info(f"New best model saved! Val AUC: {val_auc:.4f}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.patience:
                log.info(f"Early stopping after {epoch + 1} epochs")
                break
        
        # Load best model and evaluate on test set
        checkpoint = torch.load(os.path.join(save_dir, 'best_model.pt'), map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_loss, test_auc, test_ap, test_acc, test_f1, test_prec, test_rec = self.evaluate(self.test_loader, "test")
        
        # Evaluate test performance by path length
        test_by_path_length = self.evaluate_by_path_length(self.test_loader, "test")
        
        # Save final results
        results = {
            'best_epoch': checkpoint['epoch'],
            'best_val_loss': checkpoint['val_loss'],
            'best_val_auc': checkpoint['val_auc'],
            'best_val_ap': checkpoint['val_ap'],
            'best_val_accuracy': checkpoint.get('val_accuracy', 0.0),
            'best_val_f1': checkpoint.get('val_f1', 0.0),
            'best_val_precision': checkpoint.get('val_precision', 0.0),
            'best_val_recall': checkpoint.get('val_recall', 0.0),
            'test_loss': test_loss,
            'test_auc': test_auc,
            'test_ap': test_ap,
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'test_precision': test_prec,
            'test_recall': test_rec,
            'test_by_path_length': test_by_path_length,
        }
        
        with open(os.path.join(save_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        log.info(f"Training completed!")
        log.info(f"Test Metrics - AUC: {test_auc:.4f}, AP: {test_ap:.4f}")
        log.info(f"Test Metrics - Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}, Precision: {test_prec:.4f}, Recall: {test_rec:.4f}")
        return results

# ────────────────────────────── CLI ────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='GNN Link Prediction Training (CUDA Version)')
    parser.add_argument('--data_dir', default='More_Detailed_path_information/balanced',
                       help='Directory containing balanced datasets')
    parser.add_argument('--output_dir', default='gnn_checkpoint_cuda',
                       help='Output directory for checkpoints and results')
    parser.add_argument('--model_type', default='sage', choices=['sage', 'gat', 'gcn'],
                       help='Type of GNN model')
    parser.add_argument('--hidden_channels', type=int, default=128,
                       help='Number of hidden channels')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of GNN layers')
    parser.add_argument('--batch_size', type=int, default=64,  # Increased for CUDA
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--resume', default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--path_lengths', nargs='+', type=int, default=[3, 4, 5, 6],
                       help='Path lengths to include in training')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use (default: 0)')
    
    args = parser.parse_args()
    
    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Set up device for CUDA server
    if torch.cuda.is_available():
        # Set specific GPU if specified
        if args.gpu_id >= 0:
            torch.cuda.set_device(args.gpu_id)
            device = torch.device(f'cuda:{args.gpu_id}')
        else:
            device = torch.device('cuda')
        
        # CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        log.info(f"Using CUDA GPU: {torch.cuda.get_device_name(device)}")
        log.info(f"CUDA memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB total")
        log.info(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    else:
        device = torch.device('cpu')
        log.info("CUDA not available, using CPU")
    
    log.info(f"Device: {device}")
    
    # Load datasets
    log.info("Loading balanced datasets...")
    train_data, val_data, test_data = load_balanced_datasets(args.data_dir, args.path_lengths)
    
    print_dataset_stats(train_data, "Train")
    print_dataset_stats(val_data, "Validation")
    print_dataset_stats(test_data, "Test")
    
    if not train_data or not val_data or not test_data:
        log.error("Empty datasets! Check data directory and path lengths.")
        return
    
    # Create data loaders with CUDA-optimized batch size
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    
    # Get number of node features from first sample
    num_node_features = train_data[0].x.shape[1]
    log.info(f"Number of node features: {num_node_features}")
    
    # Create model
    model = GNNLinkPredictor(
        num_node_features=num_node_features,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        model_type=args.model_type
    )
    
    # Create trainer
    trainer = GNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        log.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.best_val_auc = checkpoint['val_auc']
        trainer.best_val_loss = checkpoint['val_loss']
    
    # Train
    results = trainer.train(args.num_epochs, args.output_dir)
    
    log.info("Training completed successfully!")
    log.info(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 
#!/bin/bash

# Quick GNN Architecture Comparison for Fair LLM Comparison
# ========================================================

echo "=========================================="
echo "GNN Architecture Comparison for LLM Study"
echo "=========================================="

DATA_DIR="../More_Detailed_path_information/balanced"
OUTPUT_BASE="comparison_results"

# Create output directory
mkdir -p $OUTPUT_BASE

# Models to compare
MODELS=("sage" "gat" "gcn" "transformer")
DESCRIPTIONS=(
    "GraphSAGE - Neighborhood aggregation"
    "GAT - Attention-based message passing"
    "GCN - Graph convolution"
    "Transformer - Self-attention + GNN (LLM-like)"
)

echo "Comparing ${#MODELS[@]} GNN architectures..."
echo "Data directory: $DATA_DIR"
echo "Output base: $OUTPUT_BASE"
echo ""

# Function to run training and extract results
run_model() {
    local model_type=$1
    local description=$2
    local output_dir="$OUTPUT_BASE/${model_type}_checkpoint"
    
    echo "=========================================="
    echo "Training: $description"
    echo "Model: $model_type"
    echo "Output: $output_dir"
    echo "=========================================="
    
    # Run training
    python3 gnn_trainer.py \
        --model_type $model_type \
        --output_dir $output_dir \
        --data_dir $DATA_DIR \
        --num_epochs 10 \
        --batch_size 32 \
        --hidden_channels 128 \
        --num_layers 2 \
        --learning_rate 0.001 \
        --path_lengths 3 4 5 6
    
    # Check if training was successful
    if [ $? -eq 0 ]; then
        echo "✅ $model_type training completed successfully"
        
        # Extract key metrics from results.json
        if [ -f "$output_dir/results.json" ]; then
            echo "📊 Results for $model_type:"
            python3 -c "
import json
import sys
try:
    with open('$output_dir/results.json', 'r') as f:
        results = json.load(f)
    print(f'  Test AUC: {results.get(\"test_auc\", \"N/A\"):.4f}')
    print(f'  Test AP: {results.get(\"test_ap\", \"N/A\"):.4f}')
    print(f'  Test Accuracy: {results.get(\"test_accuracy\", \"N/A\"):.4f}')
    print(f'  Test F1: {results.get(\"test_f1\", \"N/A\"):.4f}')
except Exception as e:
    print(f'  Error reading results: {e}')
"
        else
            echo "❌ No results.json found for $model_type"
        fi
    else
        echo "❌ $model_type training failed"
    fi
    
    echo ""
}

# Run all models
for i in "${!MODELS[@]}"; do
    run_model "${MODELS[$i]}" "${DESCRIPTIONS[$i]}"
done

echo "=========================================="
echo "Comparison Complete!"
echo "=========================================="

# Create summary
echo "📋 Summary of all models:"
for model in "${MODELS[@]}"; do
    results_file="$OUTPUT_BASE/${model}_checkpoint/results.json"
    if [ -f "$results_file" ]; then
        echo "  $model:"
        python3 -c "
import json
try:
    with open('$results_file', 'r') as f:
        results = json.load(f)
    print(f'    AUC: {results.get(\"test_auc\", \"N/A\"):.4f}')
    print(f'    F1: {results.get(\"test_f1\", \"N/A\"):.4f}')
except:
    print('    Results not available')
"
    fi
done

echo ""
echo "🎯 For fair LLM comparison, use the best performing GNN as baseline"
echo "📁 Check individual results in: $OUTPUT_BASE/"
echo "" 
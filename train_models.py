# -*- coding: utf-8 -*-
"""
train_models.py

This script trains GCN and GAT models on specified datasets (Cora, CiteSeer, AmazonPhoto)
and saves the trained model state_dict and initialization arguments to .pkl files
in the 'models/' directory. These saved models can then be loaded by the Dash application.

Run this script after 'preprocess_datasets.py' for faster dataset loading.
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit # For creating splits for Amazon dataset
import os
import pickle
import time
import numpy as np

# --- Configuration ---
MODEL_DIR = 'models'  # Directory to save trained models
DATA_DIR = 'data'     # Root directory for raw datasets
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed') # Subdirectory for pre-processed .pt files

# Datasets and models to train
# Ensure these dataset names match those used in preprocess_datasets.py and app.py
DATASETS_TO_TRAIN = ['Cora', 'CiteSeer', 'AmazonPhoto'] # Updated to AmazonPhoto
MODELS_TO_TRAIN = ['GCN', 'GAT'] 

# Training Hyperparameters (can be adjusted)
LEARNING_RATE = 0.005 
WEIGHT_DECAY = 5e-4
EPOCHS = 300 
EARLY_STOPPING_PATIENCE = 20 

# Model-specific Hyperparameters (kept from previous AmazonComputers attempt, suitable for AmazonPhoto)
GCN_HIDDEN_CHANNELS = 64 
GAT_HIDDEN_CHANNELS = 32 
GAT_HEADS = 8
GAT_OUTPUT_HEADS = 1 

# --- GPU Setup ---
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f"CUDA is available. Training on GPU: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device('cpu')
    print("CUDA not available. Training on CPU.")

script_dir = os.path.dirname(__file__) if '__file__' in locals() else '.'
model_abs_dir = os.path.join(script_dir, MODEL_DIR)
data_abs_dir = os.path.join(script_dir, DATA_DIR)
processed_data_abs_dir = os.path.join(script_dir, PROCESSED_DATA_DIR)

os.makedirs(model_abs_dir, exist_ok=True)
os.makedirs(data_abs_dir, exist_ok=True)
os.makedirs(processed_data_abs_dir, exist_ok=True)


# --- Model Definitions (Should match app.py) ---
class GCNNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.init_args = {'hidden_channels': hidden_channels} 

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training) 
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, output_heads=1):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=output_heads, concat=False, dropout=0.6)
        self.init_args = {'hidden_channels': hidden_channels, 'heads': heads} 

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training) 
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training) 
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# --- Dataset Wrapper for loading preprocessed data ---
class PreprocessedDatasetWrapper:
    def __init__(self, loaded_obj):
        self.data = loaded_obj['data']
        self.num_node_features = loaded_obj['num_node_features']
        self.num_classes = loaded_obj['num_classes']
        self.name = loaded_obj.get('name', 'Unknown Preprocessed')

    def __getitem__(self, idx):
        if idx == 0: return self.data
        raise IndexError(f"{self.name} index out of range")
    def __len__(self): return 1


# --- Dataset Loading Function ---
def load_training_dataset(name, root_dir, processed_dir):
    """
    Loads a dataset for training. Checks for pre-processed .pt file first.
    For AmazonPhoto, it will create random splits if not present in the loaded Data object.
    """
    print(f"  Attempting to load dataset '{name}' for training...")
    dataset_loader_obj = None 

    processed_file_path = os.path.join(processed_dir, f"{name}_processed.pt")
    data_loaded_from_pt = False
    if os.path.exists(processed_file_path):
        try:
            print(f"    Found pre-processed file. Loading '{name}' from .pt ...")
            loaded_obj = torch.load(processed_file_path, map_location=torch.device('cpu'), weights_only=False)
            if isinstance(loaded_obj, dict) and all(k in loaded_obj for k in ['data', 'num_node_features', 'num_classes']):
                dataset_loader_obj = PreprocessedDatasetWrapper(loaded_obj)
                data_loaded_from_pt = True
                print(f"    Dataset '{name}' loaded successfully from pre-processed file.")
            else:
                print(f"    Warning: Pre-processed file '{processed_file_path}' has unexpected format. Falling back.")
        except Exception as e:
            print(f"    Warning: Error loading pre-processed file '{processed_file_path}': {e}. Falling back.")
    
    if dataset_loader_obj is None: 
        try:
            if name in ['Cora', 'CiteSeer']:
                print(f"    Loading '{name}' using Planetoid (root: {root_dir})...")
                original_loader = Planetoid(root=root_dir, name=name)
            elif name == 'AmazonPhoto': # Changed from AmazonComputers
                print(f"    Loading '{name}' using Amazon Photo loader (root: {root_dir})...")
                original_loader = Amazon(root=root_dir, name='Photo') # Changed to 'Photo'
            elif name == 'AmazonComputers': # Kept for reference if user wants to switch back easily
                print(f"    Loading '{name}' using Amazon Computers loader (root: {root_dir})...")
                original_loader = Amazon(root=root_dir, name='Computers')
            else:
                print(f"    Dataset '{name}' not recognized by standard loader in training script.")
                return None
            
            data_obj_raw = original_loader[0]
            dataset_loader_obj = PreprocessedDatasetWrapper({
                'data': data_obj_raw,
                'num_node_features': original_loader.num_node_features,
                'num_classes': original_loader.num_classes,
                'name': name
            })
            print(f"    Dataset '{name}' loaded successfully (standard method).")
        except Exception as e:
            print(f"    Error loading dataset {name} using standard method: {e}")
            import traceback; traceback.print_exc()
            return None
    
    # Create splits for Amazon datasets if they don't exist
    if name in ['AmazonPhoto', 'AmazonComputers']: # Apply to both Amazon datasets
        data = dataset_loader_obj.data
        if not (hasattr(data, 'train_mask') and hasattr(data, 'val_mask') and hasattr(data, 'test_mask') and \
                data.train_mask is not None and data.val_mask is not None and data.test_mask is not None and \
                data.train_mask.sum() > 0) : 
            
            print(f"    '{name}' dataset does not have predefined splits. Creating random splits...")
            if dataset_loader_obj.num_classes > 0:
                # Using percentages for train/val/test splits
                # For AmazonPhoto (~7.6k nodes):
                # Train: ~2.5% (approx 190 nodes)
                # Val:   ~4%   (approx 300 nodes)
                # Test:  Remaining
                # These are still relatively small for demonstration purposes.
                num_train_nodes = int(data.num_nodes * 0.025) 
                num_val_nodes = int(data.num_nodes * 0.04)   
                num_test_nodes = data.num_nodes - num_train_nodes - num_val_nodes

                if num_train_nodes > 0 and num_val_nodes > 0 and num_test_nodes > 0:
                    # Using RandomNodeSplit to create train, val, and test masks
                    # It ensures that num_val and num_test are met, and the rest go to train.
                    # However, to be more explicit about train size first:
                    # We might need to adjust if we want a very specific train count first.
                    # Let's use num_val and num_test, and the rest will be train.
                    # This means train will be data.num_nodes - num_val_nodes - num_test_nodes
                    transform = RandomNodeSplit(split='train_rest', num_val=num_val_nodes, num_test=num_test_nodes)
                    
                    dataset_loader_obj.data = transform(dataset_loader_obj.data) 
                    print(f"    Created splits for '{name}': Train={dataset_loader_obj.data.train_mask.sum().item()}, Val={dataset_loader_obj.data.val_mask.sum().item()}, Test={dataset_loader_obj.data.test_mask.sum().item()}")
                else:
                    print(f"    Could not create valid splits for {name} with current settings (num_nodes={data.num_nodes}, train_ratio=0.025, val_ratio=0.04). Dataset might be too small or ratios too low.")
                    return None 
            else:
                print(f"    Cannot create class-aware splits for {name} as num_classes is 0 or unknown.")
                return None
        elif data_loaded_from_pt: 
             print(f"    '{name}' loaded from .pt already has splits (or this check was passed).")

    return dataset_loader_obj


# --- Training and Evaluation Functions ---
def train_one_epoch(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    train_mask = data.train_mask.to(device, dtype=torch.bool)
    loss = criterion(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data, mask):
    model.eval()
    out = model(data.x, data.edge_index)
    eval_mask = mask.to(device, dtype=torch.bool)
    if eval_mask.sum() == 0: # Avoid division by zero if mask is empty
        return 0.0
    pred = out[eval_mask].argmax(dim=1)
    correct = (pred == data.y[eval_mask]).sum()
    acc = int(correct) / int(eval_mask.sum()) 
    return acc

# --- Main Training Loop ---
if __name__ == '__main__':
    print("--- Starting Model Training Script ---")

    for dataset_name in DATASETS_TO_TRAIN:
        print(f"\n--- Training for Dataset: {dataset_name} ---")
        
        dataset = load_training_dataset(dataset_name, root_dir=data_abs_dir, processed_dir=processed_data_abs_dir)
        if dataset is None or dataset.data is None: 
            print(f"Skipping {dataset_name} due to loading error or empty data.")
            continue
        
        data = dataset.data.to(device) 
        num_features = dataset.num_node_features
        num_classes = dataset.num_classes

        if not (hasattr(data, 'train_mask') and hasattr(data, 'val_mask') and hasattr(data, 'test_mask') and \
                data.train_mask is not None and data.val_mask is not None and data.test_mask is not None and \
                data.train_mask.sum() > 0 and data.val_mask.sum() > 0): 
            print(f"  *** ERROR: Dataset '{dataset_name}' is missing valid train/validation masks after loading/splitting. Skipping.")
            continue

        print(f"  Dataset '{dataset_name}' on {device}. Nodes: {data.num_nodes}, Features: {num_features}, Classes: {num_classes}")
        print(f"  Train samples: {data.train_mask.sum().item()}, Val samples: {data.val_mask.sum().item()}, Test samples: {data.test_mask.sum().item()}")


        for model_type in MODELS_TO_TRAIN:
            print(f"\n  -- Training Model Type: {model_type} --")
            start_time_model_train = time.time()
            model = None
            model_init_args = {}

            if model_type == 'GCN':
                model_init_args = {'hidden_channels': GCN_HIDDEN_CHANNELS}
                model = GCNNet(num_features, GCN_HIDDEN_CHANNELS, num_classes)
            elif model_type == 'GAT':
                model_init_args = {'hidden_channels': GAT_HIDDEN_CHANNELS, 'heads': GAT_HEADS}
                model = GATNet(num_features, GAT_HIDDEN_CHANNELS, num_classes, heads=GAT_HEADS, output_heads=GAT_OUTPUT_HEADS)
            
            if model is None:
                print(f"    Unknown model type: {model_type}. Skipping.")
                continue
            
            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            criterion = torch.nn.NLLLoss() 

            best_val_acc = 0
            epochs_no_improve = 0
            best_model_state = None

            for epoch in range(1, EPOCHS + 1):
                train_loss = train_one_epoch(model, data, optimizer, criterion)
                val_acc = evaluate(model, data, data.val_mask)
                
                if epoch % 10 == 0 or epoch == 1: 
                    print(f"    Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Val Acc: {val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    epochs_no_improve = 0
                    best_model_state = model.state_dict().copy() 
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    print(f"    Early stopping triggered at epoch {epoch} due to no improvement in validation accuracy for {EARLY_STOPPING_PATIENCE} epochs.")
                    break
            
            print(f"  Training finished for {model_type} on {dataset_name}.")
            
            if best_model_state:
                model.load_state_dict(best_model_state)
                print(f"    Loaded best model state with Val Acc: {best_val_acc:.4f}")
            else:
                print("    Warning: No best model state found. Using last state.")

            test_acc = evaluate(model, data, data.test_mask)
            print(f"    Final Test Accuracy for {model_type} on {dataset_name}: {test_acc:.4f}")

            model_filename = f"{model_type}_{dataset_name}.pkl"
            model_save_path = os.path.join(model_abs_dir, model_filename)
            
            model.to('cpu') 
            save_package = (model.state_dict(), model.init_args) 

            try:
                with open(model_save_path, 'wb') as f:
                    pickle.dump(save_package, f)
                print(f"    Model saved to: {model_save_path}")
            except Exception as e:
                print(f"    Error saving model {model_save_path}: {e}")

            end_time_model_train = time.time()
            print(f"    Total training and saving time for {model_type} on {dataset_name}: {end_time_model_train - start_time_model_train:.2f} seconds.")

    print("\n--- Model Training Script Finished ---")

# -*- coding: utf-8 -*-
"""
train_models.py

This script trains GCN and GAT models on specified datasets (Cora, CiteSeer).

This script should be run after 'preprocess_datasets.py'.
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle
import time

# --- Configuration ---
MODEL_DIR = 'models'
DATA_DIR = 'data'
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

DATASETS_TO_TRAIN = ['Cora', 'CiteSeer']
MODELS_TO_TRAIN = ['GCN', 'GAT']

# --- General Training Hyperparameters ---
# Increased epochs to give the model ample time to converge with early stopping.
EPOCHS = 500
# Patience for early stopping is set to a reasonable value to avoid premature termination.
EARLY_STOPPING_PATIENCE = 50

# --- Model-Specific Hyperparameters ---
# These have been carefully tuned for each dataset and model to boost performance.
# GCN: Using smaller hidden layers to prevent overfitting and improve generalization.
GCN_CONFIG = {
    'Cora': {'lr': 0.01, 'weight_decay': 5e-4, 'hidden_channels': 32, 'dropout_rate': 0.5},
    'CiteSeer': {'lr': 0.01, 'weight_decay': 0.01, 'hidden_channels': 32, 'dropout_rate': 0.5}
}

# GAT: Using the standard, high-performing configuration for both datasets.
# The combination of 8 heads and 8 hidden channels is known to work well.
GAT_CONFIG = {
    'Cora': {'lr': 0.005, 'weight_decay': 5e-4, 'hidden_channels': 8, 'heads': 8, 'dropout_rate': 0.6},
    'CiteSeer': {'lr': 0.005, 'weight_decay': 5e-4, 'hidden_channels': 8, 'heads': 8, 'dropout_rate': 0.6}
}

# --- GPU/CPU Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# --- Path Management ---
# Resolving absolute paths for robust execution.
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
model_abs_dir = os.path.join(script_dir, MODEL_DIR)
data_abs_dir = os.path.join(script_dir, DATA_DIR)
processed_data_abs_dir = os.path.join(script_dir, PROCESSED_DATA_DIR)

os.makedirs(model_abs_dir, exist_ok=True)
os.makedirs(data_abs_dir, exist_ok=True)
os.makedirs(processed_data_abs_dir, exist_ok=True)

# --- Model Definitions (Consistent Two-Layer Architecture) ---

class GCNNet(torch.nn.Module):
    """A standard two-layer GCN model."""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.5):
        super().__init__()
        # Using cached=True for performance gain on transductive datasets.
        self.conv1 = GCNConv(in_channels, hidden_channels, cached=True)
        self.conv2 = GCNConv(hidden_channels, out_channels, cached=True)
        self.dropout_rate = dropout_rate
        # Store init args to allow for easy model reloading.
        self.init_args = {'hidden_channels': hidden_channels, 'dropout_rate': dropout_rate}

    def forward(self, x, edge_index):
        # **FIX**: Use F.relu() for applying the activation function.
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GATNet(torch.nn.Module):
    """A standard two-layer GAT model."""
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout_rate=0.6):
        super().__init__()
        # Dropout is applied at each GAT layer as per the original paper.
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout_rate)
        # The second layer uses a single head and no concatenation for the final output.
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout_rate)
        self.dropout_rate = dropout_rate
        # Store init args for model reloading.
        self.init_args = {'hidden_channels': hidden_channels, 'heads': heads, 'dropout_rate': dropout_rate}

    def forward(self, x, edge_index):
        # Dropout on input features is a common regularization technique for GATs.
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        # **FIX**: Use F.elu() for applying the activation function.
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# --- Dataset Wrapper ---
class PreprocessedDatasetWrapper:
    """A simple wrapper to standardize access to pre-processed dataset objects."""
    def __init__(self, loaded_obj):
        self.data = loaded_obj['data']
        self.num_node_features = loaded_obj['num_node_features']
        self.num_classes = loaded_obj['num_classes']
        self.name = loaded_obj.get('name', 'Unknown')
    def __getitem__(self, idx):
        if idx == 0: return self.data
        raise IndexError
    def __len__(self): return 1

# --- Dataset Loading Function ---
def load_training_dataset(name, root_dir, processed_dir):
    """Loads a dataset, prioritizing the pre-processed version if available."""
    print(f"  Attempting to load dataset '{name}'...")
    processed_file_path = os.path.join(processed_dir, f"{name}_processed.pt")
    if os.path.exists(processed_file_path):
        try:
            # Note: weights_only=False is necessary if the .pt file contains more than just tensors.
            loaded_obj = torch.load(processed_file_path, map_location='cpu', weights_only=False)
            print(f"    Loaded from pre-processed file: {processed_file_path}")
            return PreprocessedDatasetWrapper(loaded_obj)
        except Exception as e:
            print(f"    Warning: Could not load pre-processed file. Error: {e}. Falling back to Planetoid.")
    try:
        print(f"    Loading '{name}' using Planetoid...")
        dataset = Planetoid(root=root_dir, name=name)
        # Create a dictionary to match the structure of the pre-processed file.
        return PreprocessedDatasetWrapper({
            'data': dataset[0],
            'num_node_features': dataset.num_node_features,
            'num_classes': dataset.num_classes,
            'name': name
        })
    except Exception as e:
        print(f"    FATAL: Could not load dataset {name}. Error: {e}")
        return None

# --- Training & Evaluation Functions ---
def train_one_epoch(model, data, optimizer):
    """Performs a single training step and returns the loss."""
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate(model, data, mask):
    """Evaluates the model on a given data mask and returns accuracy."""
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out[mask].argmax(dim=1)
    correct = (pred == data.y[mask]).sum()
    return correct.item() / mask.sum().item()

# --- Main Training Loop ---
if __name__ == '__main__':
    print("--- Starting Hyperparameter-Tuned Model Training Script ---")
    for dataset_name in DATASETS_TO_TRAIN:
        print(f"\n--- Training on Dataset: {dataset_name} ---")
        dataset = load_training_dataset(dataset_name, data_abs_dir, processed_data_abs_dir)
        if not dataset:
            continue

        data = dataset.data.to(device)
        print(f"  Dataset '{dataset_name}' on {device}. Nodes: {data.num_nodes}, Features: {dataset.num_node_features}, Classes: {dataset.num_classes}")
        print(f"  Train: {data.train_mask.sum()}, Val: {data.val_mask.sum()}, Test: {data.test_mask.sum()}")

        for model_type in MODELS_TO_TRAIN:
            print(f"\n  -- Training Model: {model_type} --")
            start_time = time.time()
            
            config = (GCN_CONFIG if model_type == 'GCN' else GAT_CONFIG)[dataset_name]
            
            # Separate model hyperparams from optimizer hyperparams.
            # Create a new dict with only the parameters needed by the model's __init__ method.
            model_params = {k: v for k, v in config.items() if k not in ['lr', 'weight_decay']}

            if model_type == 'GCN':
                model = GCNNet(in_channels=dataset.num_node_features,
                               out_channels=dataset.num_classes,
                               **model_params) # Pass only the relevant params
            elif model_type == 'GAT':
                model = GATNet(in_channels=dataset.num_node_features,
                               out_channels=dataset.num_classes,
                               **model_params) # Pass only the relevant params
            else:
                print(f"    ERROR: Model type '{model_type}' not recognized. Skipping.")
                continue

            model = model.to(device)
            # Optimizer is still configured using the full config dict.
            optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=15, min_lr=1e-6)

            best_val_acc = 0.0
            epochs_no_improve = 0
            best_model_state = None

            for epoch in range(1, EPOCHS + 1):
                train_loss = train_one_epoch(model, data, optimizer)
                val_acc = evaluate(model, data, data.val_mask)
                
                # Learning rate scheduler step
                scheduler.step(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # Save the model state on CPU to avoid GPU memory issues.
                    best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                if epoch % 20 == 0:
                    test_acc = evaluate(model, data, data.test_mask)
                    print(f"    Epoch: {epoch:03d} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    print(f"    Early stopping triggered at epoch {epoch}. Best Val Acc: {best_val_acc:.4f}")
                    break
            
            print(f"  Training finished for {model_type}.")
            
            # Load the best performing model before final evaluation.
            if best_model_state:
                model.load_state_dict(best_model_state)

            final_test_acc = evaluate(model, data, data.test_mask)
            print(f"    > Best Validation Accuracy: {best_val_acc:.4f}")
            print(f"    > Final Test Accuracy:      {final_test_acc:.4f}")

            model_filename = f"{model_type}_{dataset_name}_tuned.pkl"
            model_save_path = os.path.join(model_abs_dir, model_filename)
            
            # Saving both state_dict and init_args for robust model loading later.
            save_package = (model.state_dict(), model.init_args)
            with open(model_save_path, 'wb') as f:
                pickle.dump(save_package, f) # Using pickle as requested.
            print(f"    Model saved to: {model_save_path}")
            
            end_time = time.time()
            print(f"    Total time: {end_time - start_time:.2f}s")
            
    print("\n--- Model Training Script Finished ---")

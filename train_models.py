import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle
import time
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap.umap_ as umap
import warnings
from link_prediction_utils import create_edge_splits, evaluate_link_prediction, generate_negative_samples
from gnnvis.models import build_model, model_capabilities

# --- Configuration ---
MODEL_DIR = 'models'
DATA_DIR = 'data'
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
DATASETS_TO_TRAIN = ['Cora', 'CiteSeer']
MODELS_TO_TRAIN = ['GCN', 'GAT']

# Suppress warnings from libraries like UMAP/TSNE for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# --- General Training Hyperparameters ---
EPOCHS = 500
EARLY_STOPPING_PATIENCE = 50

# --- Model-Specific Hyperparameters ---
GCN_CONFIG = {
    'Cora': {'lr': 0.01, 'weight_decay': 5e-4, 'hidden_channels': 32, 'dropout_rate': 0.5, 'enable_link_prediction': True, 'link_pred_weight': 0.3},
    'CiteSeer': {'lr': 0.01, 'weight_decay': 0.01, 'hidden_channels': 32, 'dropout_rate': 0.5, 'enable_link_prediction': True, 'link_pred_weight': 0.3}
}

GAT_CONFIG = {
    'Cora': {'lr': 0.005, 'weight_decay': 5e-4, 'hidden_channels': 8, 'heads': 8, 'dropout_rate': 0.6, 'enable_link_prediction': True, 'link_pred_weight': 0.3},
    'CiteSeer': {'lr': 0.005, 'weight_decay': 5e-4, 'hidden_channels': 8, 'heads': 8, 'dropout_rate': 0.6, 'enable_link_prediction': True, 'link_pred_weight': 0.3}
}

# Hyperparameters per registered model type (see gnnvis/models)
MODEL_CONFIGS = {'GCN': GCN_CONFIG, 'GAT': GAT_CONFIG}

# --- GPU Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"Device name: {torch.cuda.get_device_name(0)}")

# --- Path Management ---
script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
model_abs_dir = os.path.join(script_dir, MODEL_DIR)
data_abs_dir = os.path.join(script_dir, DATA_DIR)
processed_data_abs_dir = os.path.join(script_dir, PROCESSED_DATA_DIR)

os.makedirs(model_abs_dir, exist_ok=True)
os.makedirs(data_abs_dir, exist_ok=True)
os.makedirs(processed_data_abs_dir, exist_ok=True)

# --- Model Definitions ---
# GCNNet / GATNet live in gnnvis/models (shared with the app and the precompute
# scripts); build_model() constructs any registered model by name.

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
            # The file holds a pickled PyG Data object written by preprocess_dataset.py
            # (a trusted local file). PyTorch >= 2.6 defaults to weights_only=True,
            # which rejects it and silently fell back to Planetoid.
            loaded_obj = torch.load(processed_file_path, map_location='cpu', weights_only=False)
            print(f"    Loaded from pre-processed file: {processed_file_path}")
            return PreprocessedDatasetWrapper(loaded_obj)
        except Exception as e:
            print(f"    Warning: Could not load pre-processed file. Error: {e}. Falling back to Planetoid.")
    try:
        print(f"    Loading '{name}' using Planetoid...")
        dataset = Planetoid(root=root_dir, name=name)
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

def train_one_epoch_with_links(model, data, pos_edges, neg_edges, optimizer, config):
    """Training step with both node classification and link prediction."""
    model.train()
    optimizer.zero_grad()
    
    # Node classification loss
    node_output, embeddings = model(data.x, data.edge_index, return_embeddings=True)
    node_loss = F.nll_loss(node_output[data.train_mask], data.y[data.train_mask])
    
    total_loss = node_loss
    
    # Link prediction loss (if enabled)
    if model.enable_link_prediction and pos_edges.size(1) > 0 and neg_edges.size(1) > 0:
        # Positive edges
        pos_scores = model.predict_links(embeddings, pos_edges)
        pos_targets = torch.ones(pos_scores.size(0), device=pos_scores.device)
        
        # Negative edges
        neg_scores = model.predict_links(embeddings, neg_edges)
        neg_targets = torch.zeros(neg_scores.size(0), device=neg_scores.device)
        
        # Combine and compute link prediction loss
        all_scores = torch.cat([pos_scores, neg_scores])
        all_targets = torch.cat([pos_targets, neg_targets])
        link_loss = F.binary_cross_entropy_with_logits(all_scores, all_targets)
        
        # Combined loss
        alpha = config.get('link_pred_weight', 0.3)
        total_loss = (1 - alpha) * node_loss + alpha * link_loss
    
    total_loss.backward()
    optimizer.step()
    return total_loss.item()

def evaluate_link_prediction_model(model, data, pos_edges, neg_edges):
    """Evaluate link prediction performance."""
    model.eval()
    with torch.no_grad():
        _, embeddings = model(data.x, data.edge_index, return_embeddings=True)
        
        if pos_edges.size(1) > 0 and neg_edges.size(1) > 0:
            pos_scores = model.predict_links(embeddings, pos_edges)
            neg_scores = model.predict_links(embeddings, neg_edges)
            
            # Apply sigmoid to get probabilities
            pos_scores = torch.sigmoid(pos_scores)
            neg_scores = torch.sigmoid(neg_scores)
            
            return evaluate_link_prediction(pos_scores, neg_scores)
        else:
            return {'auc': 0.0, 'ap': 0.0, 'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

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
    print("--- Starting Model Training & Pre-computation Script ---")
    for dataset_name in DATASETS_TO_TRAIN:
        print(f"\n--- Processing Dataset: {dataset_name} ---")
        dataset = load_training_dataset(dataset_name, data_abs_dir, processed_data_abs_dir)
        if not dataset:
            continue

        data = dataset.data.to(device)
        print(f"  Dataset '{dataset_name}' on {device}. Nodes: {data.num_nodes}, Features: {dataset.num_node_features}, Classes: {dataset.num_classes}")
        print(f"  Train: {data.train_mask.sum()}, Val: {data.val_mask.sum()}, Test: {data.test_mask.sum()}")
        
        # Prepare link prediction data if any model uses it
        link_data = None
        if any(MODEL_CONFIGS[m][dataset_name].get('enable_link_prediction', False) for m in MODELS_TO_TRAIN):
            print(f"  Preparing link prediction data...")
            link_data = create_edge_splits(data, val_ratio=0.05, test_ratio=0.15)
            print(f"    Train edges: {link_data['num_train_edges']}, Val edges: {link_data['num_val_edges']}, Test edges: {link_data['num_test_edges']}")

        for model_type in MODELS_TO_TRAIN:
            print(f"\n  -- Training Model: {model_type} --")
            start_time = time.time()
            
            config = MODEL_CONFIGS[model_type][dataset_name]
            model_params = {k: v for k, v in config.items() if k not in ['lr', 'weight_decay', 'link_pred_weight']}

            try:
                model = build_model(model_type, dataset.num_node_features, dataset.num_classes, **model_params)
            except ValueError as e:
                print(f"    ERROR: {e}. Skipping.")
                continue

            model = model.to(device)
            optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.7, patience=15, min_lr=1e-6)

            best_val_acc = 0.0
            epochs_no_improve = 0
            best_model_state = None

            # Prepare training data based on whether link prediction is enabled
            if model.enable_link_prediction and link_data:
                train_data = link_data['train_data'].to(device)
                # Generate negative samples for training
                train_neg_edges = generate_negative_samples(
                    link_data['train_edge_index'], 
                    data.num_nodes, 
                    link_data['train_edge_index'].size(1) // 2  # Same number as positive edges
                ).to(device)
                # Use only a subset of positive edges for training to balance
                train_pos_edges = link_data['train_edge_index'][:, ::2].to(device)  # Every other edge
                print(f"    Link prediction enabled. Training with {train_pos_edges.size(1)} pos edges, {train_neg_edges.size(1)} neg edges")
            else:
                train_data = data
                train_pos_edges = torch.empty(2, 0, dtype=torch.long, device=device)
                train_neg_edges = torch.empty(2, 0, dtype=torch.long, device=device)

            for epoch in range(1, EPOCHS + 1):
                if model.enable_link_prediction and link_data:
                    train_loss = train_one_epoch_with_links(model, train_data, train_pos_edges, train_neg_edges, optimizer, config)
                else:
                    train_loss = train_one_epoch(model, data, optimizer)
                
                val_acc = evaluate(model, data, data.val_mask)
                
                # Evaluate link prediction if enabled
                if model.enable_link_prediction and link_data and epoch % 50 == 0:
                    link_metrics = evaluate_link_prediction_model(
                        model, train_data, 
                        link_data['val_pos_edges'].to(device), 
                        link_data['val_neg_edges'].to(device)
                    )
                    print(f"    Link Pred - AUC: {link_metrics['auc']:.4f}, AP: {link_metrics['ap']:.4f}")
                
                scheduler.step(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    # Store state dict on CPU to save GPU memory
                    best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                
                if epoch % 50 == 0:
                    test_acc = evaluate(model, data, data.test_mask)
                    print(f"    Epoch: {epoch:03d} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Test Acc: {test_acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    print(f"    Early stopping triggered at epoch {epoch}. Best Val Acc: {best_val_acc:.4f}")
                    break
            
            print(f"  Training finished for {model_type}.")
            
            if not best_model_state:
                print("  WARNING: No best model was saved. Skipping pre-computation.")
                continue

            # --- PRE-COMPUTATION STEP ---
            print(f"\n  -- Pre-computing artifacts for {model_type} on {dataset_name} --")
            # Load the best performing model state for inference
            model.load_state_dict(best_model_state)
            model.to(device)
            model.eval()

            # 1. Pre-compute Predictions and Embeddings
            print("    1. Computing predictions and high-dimensional embeddings...")
            has_attention = model_capabilities(model_type)['attention']
            if has_attention:
                log_probs, embeddings, attention_weights = model.inference(data, return_attention_weights=True)
            else:
                log_probs, embeddings = model.inference(data)
            
            predictions = log_probs.argmax(dim=1)
            embeddings_np = embeddings.cpu().numpy()

            # 2. Pre-compute 2D Embeddings for visualization
            print("    2. Computing 2D embeddings (PCA, t-SNE, UMAP)...")
            
            # PCA (fastest)
            pca_2d = PCA(n_components=2).fit_transform(embeddings_np)
            
            # t-SNE
            # Adjust perplexity for small datasets to avoid errors
            perplexity = min(30.0, max(5.0, float(embeddings_np.shape[0] - 1)))
            tsne_2d = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto').fit_transform(embeddings_np)
            
            # UMAP
            # Adjust n_neighbors for small datasets
            n_neighbors = min(15, embeddings_np.shape[0] - 1)
            umap_2d = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1, random_state=42).fit_transform(embeddings_np)

            # 3. Assemble the save package
            print("    3. Assembling final save package...")
            save_package = {
                'model_state_dict': best_model_state,
                'model_init_args': model.init_args,
                'predictions': predictions.cpu().tolist(),
                'confidences': log_probs.exp().max(dim=1).values.cpu().tolist(),
                'embeddings': embeddings.cpu().tolist(), # High-dimensional embeddings
                'embeddings_2d': {
                    'pca': pca_2d.tolist(),
                    'tsne': tsne_2d.tolist(),
                    'umap': umap_2d.tolist()
                }
            }
            
            # Add attention weights only for GAT models
            if has_attention:
                print("    4. Storing attention weights...")
                # Deconstruct tuple and store on CPU
                att1, att2 = attention_weights
                save_package['attention_weights'] = {
                    'conv1': (att1[0].cpu().tolist(), att1[1].cpu().tolist()),
                    'conv2': (att2[0].cpu().tolist(), att2[1].cpu().tolist())
                }
            
            # Add link prediction results if enabled
            if model.enable_link_prediction and link_data:
                print("    5. Computing link prediction results...")
                # Final evaluation on test set
                final_link_metrics = evaluate_link_prediction_model(
                    model, train_data if model.enable_link_prediction else data,
                    link_data['test_pos_edges'].to(device),
                    link_data['test_neg_edges'].to(device)
                )
                save_package['link_prediction'] = {
                    'enabled': True,
                    'test_metrics': final_link_metrics,
                    'edge_splits': {
                        'num_train_edges': link_data['num_train_edges'],
                        'num_val_edges': link_data['num_val_edges'],
                        'num_test_edges': link_data['num_test_edges']
                    }
                }
                print(f"    Final Link Prediction - AUC: {final_link_metrics['auc']:.4f}, AP: {final_link_metrics['ap']:.4f}")
            else:
                save_package['link_prediction'] = {'enabled': False}

            # 4. Save the comprehensive package to disk
            model_filename = f"{model_type}_{dataset_name}_tuned.pkl"
            model_save_path = os.path.join(model_abs_dir, model_filename)
            
            with open(model_save_path, 'wb') as f:
                pickle.dump(save_package, f)
            
            print(f"    Successfully pre-computed and saved artifacts to: {model_save_path}")
            
            final_test_acc = evaluate(model, data, data.test_mask)
            print(f"    > Best Validation Accuracy: {best_val_acc:.4f}")
            print(f"    > Final Test Accuracy:      {final_test_acc:.4f}")
            
            end_time = time.time()
            print(f"    Total time (train + pre-compute): {end_time - start_time:.2f}s")
            
    print("\n--- Model Training & Pre-computation Script Finished ---")

import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.datasets import Planetoid
import pickle
from cpa_iv_link_prediction import run_cpa_iv_link
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Define Paths
SCRIPT_DIR = os.getcwd()
MODEL_DIR = 'models'
DATA_DIR = 'data'
MODEL_ABS_DIR = os.path.join(SCRIPT_DIR, MODEL_DIR)
DATA_ABS_DIR = os.path.join(SCRIPT_DIR, DATA_DIR)
device = torch.device('cpu')

# --- Model Definitions (Copied from updated1_app.py) ---

class GCNNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout_rate=0.5, enable_link_prediction=False, **kwargs):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout_rate = dropout_rate
        self.enable_link_prediction = enable_link_prediction

        if enable_link_prediction:
            self.link_predictor = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels * 2, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(hidden_channels, 1)
            )

    def forward(self, x, edge_index, edge_weight=None, return_embeddings=False, **kwargs):
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        embeddings = x  
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        node_output = F.log_softmax(x, dim=1)
        if return_embeddings:
            return node_output, embeddings
        return node_output

    def predict_links(self, embeddings, edge_index):
        if not self.enable_link_prediction:
            raise ValueError("Link prediction not enabled for this model")
        source_emb = embeddings[edge_index[0]]
        target_emb = embeddings[edge_index[1]]
        edge_features = torch.cat([source_emb, target_emb], dim=1)
        link_scores = self.link_predictor(edge_features)
        return link_scores.squeeze()

class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout_rate=0.6, enable_link_prediction=False, **kwargs):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout_rate)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout_rate)
        self.dropout_rate = dropout_rate
        self.enable_link_prediction = enable_link_prediction
        self.init_args = {'hidden_channels': hidden_channels, 'heads': heads, 'dropout_rate': dropout_rate, 'enable_link_prediction': enable_link_prediction, **kwargs}

        if enable_link_prediction:
            self.link_predictor = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels * heads * 2, hidden_channels * heads),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(hidden_channels * heads, 1)
            )

    def forward(self, x, edge_index, return_embeddings=False):
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        embeddings = x 
        x = self.conv2(x, edge_index)
        node_output = F.log_softmax(x, dim=1)
        if return_embeddings:
            return node_output, embeddings
        return node_output

    def predict_links(self, embeddings, edge_index):
        if not self.enable_link_prediction:
            raise ValueError("Link prediction not enabled for this model")
        source_emb = embeddings[edge_index[0]]
        target_emb = embeddings[edge_index[1]]
        edge_features = torch.cat([source_emb, target_emb], dim=1)
        link_scores = self.link_predictor(edge_features)
        return link_scores.squeeze()

def load_precomputed_package(model_type, dataset_name):
    filename = f"{model_type}_{dataset_name}_tuned.pkl"
    path = os.path.join(MODEL_ABS_DIR, filename)
    if not os.path.exists(path):
        print(f"ERROR: Pre-computed file not found: {path}")
        return None
    try:
        with open(path, 'rb') as f:
            package = pickle.load(f)
        return package
    except Exception as e:
        print(f"ERROR: Failed to load pre-computed file '{path}'. Error: {e}")
        return None

def instantiate_model(package, model_type, num_features, num_classes):
    if not package: return None
    try:
        model_args = package.get('model_init_args', {})
        if model_type == 'GCN': 
            model = GCNNet(in_channels=num_features, out_channels=num_classes, **model_args)
        elif model_type == 'GAT':
            model = GATNet(in_channels=num_features, out_channels=num_classes, **model_args)
        
        state_dict = package.get('model_state_dict', {})
        state_dict = {key: torch.tensor(value) if isinstance(value, list) else value 
                     for key, value in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"ERROR: Could not instantiate model: {e}")
        return None

# --- Main Execution ---

def main():
    print("Initializing Causal Path Analysis...")
    
    # 1. Load Data and Model
    dataset_name = 'Cora'
    model_type = 'GCN'
    
    # Load Dataset
    dataset = Planetoid(root=DATA_ABS_DIR, name=dataset_name)
    data = dataset[0]
    
    # Load Model Package
    package = load_precomputed_package(model_type, dataset_name)
    if not package:
        print("Failed to load model package.")
        return

    # Instantiate Model
    model = instantiate_model(package, model_type, data.num_features, dataset.num_classes)
    if not model:
        print("Failed to instantiate model.")
        return

    # 2. Run Analysis
    source_node = 45
    target_node = 75
    
    print(f"Running CPA-IV for Link: {source_node} -> {target_node}")
    
    cpa_results = run_cpa_iv_link(model, data, source_node, target_node, top_k=5, max_path_len=3)
    
    if 'error' in cpa_results:
        print(f"Analysis Failed: {cpa_results['error']}")
        return

    # 3. Format Output
    print("\n" + "="*40)
    print("      CAUSAL LINK ANALYSIS REPORT      ")
    print("="*40)
    
    print(f"\nLink Evaluated: {source_node} -> {target_node}")
    print(f"Baseline Prediction Score: {cpa_results.get('baseline_score', 0):.4f}")
    
    print("\nTop Causal Paths (Effect on Link Score):")
    print("-" * 40)
    
    paths = cpa_results.get('paths', [])
    if not paths:
        print("No significant causal paths found.")
    else:
        for i, p in enumerate(paths, 1):
            path_str = " -> ".join(map(str, p['nodes']))
            score = p['score']
            print(f"{i}. Path: [{path_str}]")
            print(f"   Effect Size: {score:.4f}")
            print(f"   (Removing this path reduces link probability by {score:.4f})")
            print("-" * 20)

    # Verdict
    # Simple heuristic for verdict
    strong_paths = [p for p in paths if p['score'] > 0.05]
    if len(strong_paths) > 0:
        verdict = "Strong Causal Support"
    elif len(paths) > 0:
        verdict = "Weak Causal Support"
    else:
        verdict = "No Causal Evidence Found"
        
    print(f"\nFinal Causal Verdict: {verdict}")
    print("="*40)

if __name__ == "__main__":
    main()

import os
import torch
from torch_geometric.datasets import Planetoid
from gnnvis.models import load_package, load_model
from cpa_iv_link_prediction import run_cpa_iv_link
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Define Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # not cwd: imported by other scripts
MODEL_DIR = 'models'
DATA_DIR = 'data'
MODEL_ABS_DIR = os.path.join(SCRIPT_DIR, MODEL_DIR)
DATA_ABS_DIR = os.path.join(SCRIPT_DIR, DATA_DIR)
device = torch.device('cpu')

# Models: gnnvis/models (shared with the app and the training script)

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
    package = load_package(model_type, dataset_name, MODEL_ABS_DIR)
    if not package:
        print("Failed to load model package.")
        return

    # Instantiate Model
    model = load_model(package, model_type, data.num_features, dataset.num_classes, device=device)
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

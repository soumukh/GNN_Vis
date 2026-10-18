"""
precompute_stability.py
Run stability analysis for every dataset/model/sigma combination and save
to the stability_cache/ folder so the Dash app never has to recompute.

Usage:
    python precompute_stability.py
    python precompute_stability.py --datasets Cora --models GCN --sigmas 0.05
"""

import argparse
import math
import os
import pickle
import sys
import time

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.explain import Explainer, GNNExplainer

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR       = os.path.join(SCRIPT_DIR, 'models')
DATA_DIR        = os.path.join(SCRIPT_DIR, 'data')
CACHE_DIR       = os.path.join(SCRIPT_DIR, 'stability_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

device = torch.device('cpu')

# ── Model definitions (must mirror app.py exactly) ──────────────────

class GCNNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 dropout_rate=0.5, enable_link_prediction=False, **kwargs):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout_rate = dropout_rate
        self.enable_link_prediction = enable_link_prediction

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        return F.log_softmax(x, dim=1)


class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 heads=8, dropout_rate=0.6, enable_link_prediction=False, **kwargs):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout_rate)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=dropout_rate)
        self.dropout_rate = dropout_rate
        self.enable_link_prediction = enable_link_prediction

    def forward(self, x, edge_index, **kwargs):
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_package(model_type, dataset_name):
    path = os.path.join(MODEL_DIR, f"{model_type}_{dataset_name}_tuned.pkl")
    if not os.path.exists(path):
        print(f"  [SKIP] Package not found: {path}")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def build_model(package, model_type, num_features, num_classes):
    args = package.get('model_init_args', {})
    if model_type == 'GCN':
        model = GCNNet(in_channels=num_features, out_channels=num_classes, **args)
    else:
        model = GATNet(in_channels=num_features, out_channels=num_classes, **args)
    sd = package.get('model_state_dict', {})
    sd = {k: torch.tensor(v) if isinstance(v, list) else v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def get_edge_mask(explanation):
    for attr in ('edge_mask', '_edge_mask', 'edge_weight', '_edge_weight'):
        val = getattr(explanation, attr, None)
        if val is not None:
            return val.detach().cpu().numpy()
    if hasattr(explanation, 'to_dict'):
        d = explanation.to_dict()
        for k in ('edge_mask', 'edge_weight'):
            if k in d and d[k] is not None:
                return d[k].detach().cpu().numpy()
    return None


def cache_key(dataset_name, model_type, sigma):
    return f"{dataset_name.lower()}_{model_type.lower()}_{sigma}"


def cache_path(key):
    filename = f"stability_{key.replace('.', '_').replace(' ', '_')}.pkl"
    return os.path.join(CACHE_DIR, filename)


# ── Core stability computation (shared with the app) ─────────────────────────

def run_stability(model, data, predictions, sigma):
    """Delegate to app_stability.run_stability_analysis so cached results use
    exactly the same definitions (seeding, Lipschitz, fidelity) as live runs."""
    from app_stability import run_stability_analysis
    return run_stability_analysis(model, data, predictions, sigma, sample_size=0,
                                  progress_every=100)


# ── Sanitise ──────────────────────────────────────────────────────────────────

def sanitise(results):
    def _s(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    return [{k: _s(v) for k, v in r.items()} for r in results]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Precompute stability cache")
    parser.add_argument('--datasets', nargs='+', default=['Cora', 'CiteSeer'])
    parser.add_argument('--models',   nargs='+', default=['GCN', 'GAT'])
    parser.add_argument('--sigmas',   nargs='+', type=float, default=[0.01, 0.05, 0.1])
    parser.add_argument('--force',    action='store_true',
                        help='Recompute even if cache already exists')
    args = parser.parse_args()

    combos = [(d, m, s) for d in args.datasets
                         for m in args.models
                         for s in args.sigmas]

    print(f"\nPrecomputing stability for {len(combos)} combinations …\n")

    for dataset_name, model_type, sigma in combos:
        key  = cache_key(dataset_name, model_type, sigma)
        path = cache_path(key)

        print(f"-- {dataset_name} / {model_type} / sigma={sigma}  (key={key})")

        if os.path.exists(path) and not args.force:
            print(f"  Already cached — skipping. (use --force to recompute)\n")
            continue

        # Load package
        package = load_package(model_type, dataset_name)
        if package is None:
            print()
            continue

        # Load dataset
        try:
            ds   = Planetoid(root=DATA_DIR, name=dataset_name)
            data = ds[0]
        except Exception as e:
            print(f"  [SKIP] Failed to load dataset: {e}\n")
            continue

        # Build model
        try:
            model = build_model(package, model_type, data.num_features, ds.num_classes)
        except Exception as e:
            print(f"  [SKIP] Failed to build model: {e}\n")
            continue

        # Live predictions on the full graph (the saved package predictions can
        # come from the training graph), matching what the app now uses.
        with torch.no_grad():
            preds = model(data.x, data.edge_index).argmax(dim=1).tolist()
        predictions_data = {'preds': preds}

        data_obj = Data(x=data.x, edge_index=data.edge_index,
                        y=data.y, num_nodes=data.num_nodes)

        print(f"  nodes={data.num_nodes}  features={data.num_features}  "
              f"classes={ds.num_classes}  edges={data.edge_index.shape[1]}")
        print(f"  Computing from scratch …", flush=True)

        t0 = time.time()
        results = run_stability(model, data_obj, predictions_data, sigma)
        elapsed = time.time() - t0

        if not results:
            print(f"  [WARN] 0 results returned — not caching.\n")
            continue

        safe = sanitise(results)

        with open(path, 'wb') as f:
            pickle.dump(safe, f)

        avg_s = sum(r.get('stability', 0) or 0 for r in safe) / len(safe)
        n_ok  = sum(1 for r in safe if r.get('correct'))
        print(f"  Saved {len(safe)} nodes -> {path}")
        print(f"  avg_stability={avg_s:.3f}  correct={n_ok}/{len(safe)}  "
              f"time={elapsed:.1f}s\n")

    print("Done.")


if __name__ == '__main__':
    main()

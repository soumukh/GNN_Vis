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
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid

from gnnvis.models import available_models, load_package, load_model, model_capabilities

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR       = os.path.join(SCRIPT_DIR, 'models')
DATA_DIR        = os.path.join(SCRIPT_DIR, 'data')
CACHE_DIR       = os.path.join(SCRIPT_DIR, 'stability_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

device = torch.device('cpu')

# Models come from gnnvis/models (the single definition shared with the app);
# the stability computation itself lives in app_stability.py.


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
    parser.add_argument('--models',   nargs='+',
                        default=[m for m in available_models() if model_capabilities(m)['stability']],
                        help='Default: every registered model that supports stability analysis')
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

        if not model_capabilities(model_type)['stability']:
            print(f"  [SKIP] Stability analysis is not supported for {model_type}\n")
            continue

        if os.path.exists(path) and not args.force:
            print(f"  Already cached — skipping. (use --force to recompute)\n")
            continue

        # Load package
        package = load_package(model_type, dataset_name, MODEL_DIR)
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
            model = load_model(package, model_type, data.num_features, ds.num_classes, device=device)
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

"""
precompute_link_stability.py
────────────────────────────
Offline script: compute GNNExplainer Top-k Jaccard stability for every
edge in the graph, for all dataset / model / sigma combinations, and save
the results as parquet files in stability_cache/.

The Dash app loads these files at runtime — zero recomputation.

Usage
-----
    # Compute everything (first run or refresh)
    python precompute_link_stability.py

    # Targeted run
    python precompute_link_stability.py --datasets Cora --models GCN --sigmas 0.05

    # Force recompute even if parquet already exists
    python precompute_link_stability.py --force

Output
------
    stability_cache/stability_link_{dataset}_{model}_{sigma}.parquet

Columns: source, target, confidence, stability, degree_sum, common_neighbors, correct
"""

import argparse
import math
import os
import time

import pandas as pd
import torch
from torch_geometric.data import Data

from gnnvis import stability_cache
from gnnvis.models import available_models, load_package, load_model, model_capabilities

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(SCRIPT_DIR, 'models')
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')
CACHE_DIR  = stability_cache.CACHE_DIR


# Models come from gnnvis/models (the single definition shared with the app);
# the stability computation itself lives in app_stability.py.


def _safe(v):
    """Replace NaN/Inf floats with None for clean serialisation."""
    return None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v


# ── Core computation ──────────────────────────────────────────────────────────

def run_link_stability(model, data: Data, sigma: float) -> pd.DataFrame:
    """Full-graph link stability: one row per undirected edge.
    Delegates to app_stability.run_link_stability_analysis so precomputed and
    live results share one implementation (seeding, explainer config)."""
    from app_stability import run_link_stability_analysis
    rows = run_link_stability_analysis(model, data, sigma, sample_size=0, progress_every=50)
    return pd.DataFrame([{k: _safe(v) for k, v in r.items()} for r in rows])


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute link-stability cache (parquet)")
    parser.add_argument('--datasets', nargs='+', default=['Cora', 'CiteSeer'],
                        help='Dataset names matching models/*_tuned.pkl filenames')
    parser.add_argument('--models',   nargs='+',
                        default=[m for m in available_models() if model_capabilities(m)['stability']],
                        help='Default: every registered model that supports stability analysis')
    parser.add_argument('--sigmas',   nargs='+', type=float, default=[0.01, 0.05, 0.1])
    parser.add_argument('--force',    action='store_true',
                        help='Recompute even if a valid parquet file exists')
    args = parser.parse_args()

    combos = [(d, m, s) for d in args.datasets
                         for m in args.models
                         for s in args.sigmas]
    print(f"\n{'='*60}")
    print(f"Link stability precompute: {len(combos)} combinations")
    print(f"Output dir: {CACHE_DIR}")
    print(f"{'='*60}\n")

    from torch_geometric.datasets import Planetoid

    for dataset_name, model_type, sigma in combos:
        print(f"── {dataset_name} / {model_type} / σ={sigma}")

        if not model_capabilities(model_type)['stability']:
            print(f"  [SKIP] Stability analysis is not supported for {model_type}\n")
            continue

        # Load the model package the app uses (models/*_tuned.pkl)
        pkg = load_package(model_type, dataset_name, MODEL_DIR)
        if pkg is None:
            print()
            continue

        # Load dataset
        try:
            ds   = Planetoid(root=DATA_DIR, name=dataset_name)
            data = ds[0]
        except Exception as e:
            print(f"  [SKIP] Dataset load failed: {e}\n")
            continue

        try:
            model = load_model(pkg, model_type, data.num_features, ds.num_classes, device='cpu')
        except Exception as e:
            print(f"  [SKIP] Model build failed: {e}\n")
            continue

        # Skip only if the existing file was produced by this model
        fingerprint = stability_cache.model_fingerprint(pkg)
        if not args.force:
            cached, state, _ = stability_cache.load(
                stability_cache.LINK, dataset_name, model_type, sigma, fingerprint,
                verify=lambda r: stability_cache.link_results_match(model, data, r))
            if state in (stability_cache.HIT, stability_cache.STAMPED):
                avg = pd.DataFrame(cached)['stability'].dropna().mean()
                print(f"   Already cached and up to date — {len(cached)} edges, "
                      f"avg S={avg:.3f}  (--force to recompute)\n")
                continue
            if state == stability_cache.STALE:
                print("   Cached file is out of date (model changed) — recomputing.")

        print(f"  nodes={data.num_nodes}  features={data.num_features}  "
              f"classes={ds.num_classes}  edges={data.edge_index.shape[1]}")
        print(f"  Starting computation …\n", flush=True)

        t0 = time.time()
        try:
            df = run_link_stability(model, data, sigma)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  [ERROR] Computation failed: {e}\n")
            continue
        elapsed = time.time() - t0

        if df.empty:
            print(f"  [WARN] Empty DataFrame — nothing saved.\n")
            continue

        out = stability_cache.save(stability_cache.LINK, dataset_name, model_type, sigma, fingerprint,
                                   df.to_dict('records'))

        avg_s = float(df['stability'].dropna().mean())
        n_ok  = int(df['correct'].sum())
        print(f"\n  ✓ Saved {len(df)} edges → {out}")
        print(f"    avg_stability={avg_s:.3f}  correct={n_ok}/{len(df)}"
              f"  time={elapsed:.0f}s\n")

    print("All done.")


if __name__ == '__main__':
    main()

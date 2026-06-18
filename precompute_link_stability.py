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
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(SCRIPT_DIR, 'models')
LP_DIR     = os.path.join(MODEL_DIR, 'link_prediction')
DATA_DIR   = os.path.join(SCRIPT_DIR, 'data')
CACHE_DIR  = os.path.join(SCRIPT_DIR, 'stability_cache')
os.makedirs(CACHE_DIR, exist_ok=True)


# ── Model definitions — must mirror updated1_app.py exactly ──────────────────

class GCNNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 dropout_rate=0.5, enable_link_prediction=False, **kwargs):
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
                torch.nn.Linear(hidden_channels, 1),
            )

    def forward(self, x, edge_index, edge_weight=None, return_embeddings=False, **kwargs):
        h = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        out = self.conv2(h, edge_index, edge_weight=edge_weight)
        log_probs = F.log_softmax(out, dim=1)
        return (log_probs, h) if return_embeddings else log_probs

    def predict_links(self, emb, edge_index):
        if not self.enable_link_prediction:
            raise ValueError("enable_link_prediction=False")
        src = emb[edge_index[0]]
        tgt = emb[edge_index[1]]
        return self.link_predictor(torch.cat([src, tgt], dim=1)).squeeze()


class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 heads=8, dropout_rate=0.6, enable_link_prediction=False, **kwargs):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout_rate)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=dropout_rate)
        self.dropout_rate = dropout_rate
        self.enable_link_prediction = enable_link_prediction
        if enable_link_prediction:
            self.link_predictor = torch.nn.Sequential(
                torch.nn.Linear(hidden_channels * heads * 2, hidden_channels * heads),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout_rate),
                torch.nn.Linear(hidden_channels * heads, 1),
            )

    def forward(self, x, edge_index, return_embeddings=False, **kwargs):
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        h = F.elu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        out = self.conv2(h, edge_index)
        log_probs = F.log_softmax(out, dim=1)
        return (log_probs, h) if return_embeddings else log_probs

    def predict_links(self, emb, edge_index):
        if not self.enable_link_prediction:
            raise ValueError("enable_link_prediction=False")
        src = emb[edge_index[0]]
        tgt = emb[edge_index[1]]
        return self.link_predictor(torch.cat([src, tgt], dim=1)).squeeze()


# ── GNNExplainer wrapper — dot-product or dedicated head ─────────────────────

class _LinkDotWrapper(torch.nn.Module):
    """Wraps the base GNN for GNNExplainer.
    Returns sigmoid score [1,1] for edge (src→dst) using dot product or
    the model's link_predictor head if available."""

    def __init__(self, base_model, src: int, dst: int):
        super().__init__()
        self.base_model = base_model
        self.src = int(src)
        self.dst = int(dst)

    def forward(self, x, edge_index, **kwargs):
        _, emb = self.base_model(x, edge_index, return_embeddings=True)
        if getattr(self.base_model, 'enable_link_prediction', False):
            ep    = torch.tensor([[self.src], [self.dst]], device=x.device, dtype=torch.long)
            score = self.base_model.predict_links(emb, ep)
        else:
            score = (emb[self.src] * emb[self.dst]).sum()
        return score.sigmoid().view(1, 1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_lp_package(model_type: str, dataset_name: str) -> dict | None:
    path = os.path.join(LP_DIR, f"{model_type}_{dataset_name}_link_pred.pkl")
    if not os.path.exists(path):
        print(f"  [SKIP] Package not found: {path}")
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def build_model(package: dict, model_type: str, num_features: int, num_classes: int):
    args = dict(package.get('model_init_args', {}))
    if model_type == 'GCN':
        model = GCNNet(in_channels=num_features, out_channels=num_classes, **args)
    elif model_type == 'GAT':
        model = GATNet(in_channels=num_features, out_channels=num_classes, **args)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    sd = {k: torch.tensor(v) if isinstance(v, list) else v
          for k, v in package.get('model_state_dict', {}).items()}
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def get_edge_mask(exp) -> np.ndarray | None:
    for attr in ('edge_mask', '_edge_mask', 'edge_weight', '_edge_weight'):
        val = getattr(exp, attr, None)
        if val is not None:
            return val.detach().cpu().numpy()
    if hasattr(exp, 'to_dict'):
        d = exp.to_dict()
        for k in ('edge_mask', 'edge_weight'):
            if k in d and d[k] is not None:
                return d[k].detach().cpu().numpy()
    return None


def parquet_path(dataset_name: str, model_type: str, sigma: float) -> str:
    key   = f"link_{dataset_name.lower()}_{model_type.lower()}_{sigma}"
    fname = f"stability_{key.replace('.', '_')}.parquet"
    return os.path.join(CACHE_DIR, fname)


def _safe(v):
    """Replace NaN/Inf floats with None for clean serialisation."""
    return None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v


# ── Core computation ──────────────────────────────────────────────────────────

def run_link_stability(model, data: Data, sigma: float) -> pd.DataFrame:
    """Full-graph link stability: returns a DataFrame with one row per edge."""
    x          = data.x
    edge_index = data.edge_index
    num_nodes  = data.num_nodes
    ei_cpu     = edge_index.cpu().numpy()

    # Out-degree per node
    degrees = np.zeros(num_nodes, dtype=np.int32)
    for s in ei_cpu[0]:
        degrees[int(s)] += 1

    # Neighbour sets for common-neighbour count (O(E) build)
    adj: list[set] = [set() for _ in range(num_nodes)]
    for col in range(ei_cpu.shape[1]):
        u, v = int(ei_cpu[0, col]), int(ei_cpu[1, col])
        adj[u].add(v)
        adj[v].add(u)

    # Deduplicate directed → undirected edges
    seen:  set   = set()
    edges: list  = []
    for col in range(ei_cpu.shape[1]):
        u, v = int(ei_cpu[0, col]), int(ei_cpu[1, col])
        key  = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            edges.append((u, v))
    print(f"  Unique undirected edges: {len(edges)}", flush=True)

    # Batch confidence — single forward pass for all edges
    with torch.no_grad():
        _, emb_clean = model(x, edge_index, return_embeddings=True)
        srcs = torch.tensor([u for u, _ in edges], dtype=torch.long)
        dsts = torch.tensor([v for _, v in edges], dtype=torch.long)
        if getattr(model, 'enable_link_prediction', False):
            scores = model.predict_links(emb_clean, torch.stack([srcs, dsts]))
        else:
            scores = (emb_clean[srcs] * emb_clean[dsts]).sum(dim=1)
        confs = torch.sigmoid(scores).cpu().numpy().ravel()

    rows:         list = []
    n_ok = n_fail = 0
    skip_reasons: dict = {}

    for i, (u, v) in enumerate(edges):
        try:
            conf    = _safe(float(confs[i]))
            deg_sum = int(degrees[u]) + int(degrees[v])
            cn      = len(adj[u] & adj[v])
            correct = bool((conf or 0.0) > 0.5)

            wrapper = _LinkDotWrapper(model, u, v)
            wrapper.eval()
            expl = Explainer(
                model=wrapper,
                algorithm=GNNExplainer(epochs=20),
                explanation_type='model',
                edge_mask_type='object',
                model_config=dict(
                    mode='binary_classification',
                    task_level='graph',
                    return_type='raw',
                ),
            )

            exp_orig  = expl(x=x, edge_index=edge_index)
            mask_orig = get_edge_mask(exp_orig)

            noise     = torch.randn_like(x) * sigma
            exp_pert  = expl(x=(x + noise), edge_index=edge_index)
            mask_pert = get_edge_mask(exp_pert)

            if mask_orig is None or mask_pert is None:
                raise ValueError("edge mask extraction failed")

            k       = min(10, len(mask_orig))
            top_o   = set(np.argsort(mask_orig)[-k:].tolist())
            top_p   = set(np.argsort(mask_pert)[-k:].tolist())
            union   = top_o | top_p
            jaccard = _safe(len(top_o & top_p) / len(union) if union else 1.0)

            rows.append({
                'source':           u,
                'target':           v,
                'confidence':       conf,
                'stability':        jaccard,
                'degree_sum':       deg_sum,
                'common_neighbors': cn,
                'correct':          correct,
            })
            n_ok += 1

            if (i + 1) % 50 == 0:
                pct = 100 * (i + 1) / len(edges)
                print(f"  [{i+1}/{len(edges)}] {pct:.1f}% — ok={n_ok} fail={n_fail}",
                      flush=True)

        except Exception as e:
            n_fail += 1
            reason = f"{type(e).__name__}: {str(e)[:80]}"
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1

    print(f"  Finished: {n_ok} ok / {n_fail} failed")
    if skip_reasons:
        print(f"  Failure breakdown: {skip_reasons}")
    return pd.DataFrame(rows)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute link-stability cache (parquet)")
    parser.add_argument('--datasets', nargs='+', default=['Cora', 'CiteSeer'],
                        help='Dataset names matching models/link_prediction/ filenames')
    parser.add_argument('--models',   nargs='+', default=['GCN', 'GAT'])
    parser.add_argument('--sigmas',   nargs='+', type=float, default=[0.01, 0.05, 0.1])
    parser.add_argument('--force',    action='store_true',
                        help='Recompute even if parquet already exists')
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
        out = parquet_path(dataset_name, model_type, sigma)
        print(f"── {dataset_name} / {model_type} / σ={sigma}")
        print(f"   {out}")

        if os.path.exists(out) and not args.force:
            df_check = pd.read_parquet(out)
            avg = df_check['stability'].dropna().mean()
            print(f"   Already cached — {len(df_check)} edges, "
                  f"avg S={avg:.3f}  (--force to recompute)\n")
            continue

        # Load link-prediction model package
        pkg = load_lp_package(model_type, dataset_name)
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

        # Instantiate model (dot-product mode; no link prediction head required)
        try:
            model = build_model(pkg, model_type, data.num_features, ds.num_classes)
        except Exception as e:
            print(f"  [SKIP] Model build failed: {e}\n")
            continue

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

        df.to_parquet(out, index=False)

        avg_s = float(df['stability'].dropna().mean())
        n_ok  = int(df['correct'].sum())
        print(f"\n  ✓ Saved {len(df)} edges → {out}")
        print(f"    avg_stability={avg_s:.3f}  correct={n_ok}/{len(df)}"
              f"  time={elapsed:.0f}s\n")

    print("All done.")


if __name__ == '__main__':
    main()

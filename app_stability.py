"""
Stability backend helpers extracted from app.py.
Contains cache helpers, node/link stability compute routines and fast prediction-consistency routine.
"""

import os
import pickle
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

from torch_geometric.explain import Explainer, GNNExplainer

STABILITY_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stability_cache')
os.makedirs(STABILITY_CACHE_DIR, exist_ok=True)


def get_stability_cache_key(dataset_name: str, model_type: str, sigma: float) -> str:
    return f"{dataset_name.lower()}_{model_type.lower()}_{sigma}"


def get_stability_cache_path(cache_key: str) -> str:
    filename = f"stability_{cache_key.replace('.', '_').replace(' ', '_')}.pkl"
    return os.path.join(STABILITY_CACHE_DIR, filename)


def load_stability_cache(cache_key: str):
    path = get_stability_cache_path(cache_key)
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            print(f"[STABILITY CACHE] Loaded from cache: {path}")
            return data
        except Exception as e:
            print(f"[STABILITY CACHE] Failed to read cache ({path}): {e} — recomputing.")
            return None
    return None


def save_stability_cache(cache_key: str, results: list) -> None:
    path = get_stability_cache_path(cache_key)
    try:
        with open(path, 'wb') as f:
            pickle.dump(results, f)
        print(f"[STABILITY CACHE] Saved to cache: {path}")
    except Exception as e:
        print(f"[STABILITY CACHE] Failed to save cache ({path}): {e}")


# Link-stability (parquet preferred) helpers

def get_link_stability_cache_key(dataset_name: str, model_type: str, sigma: float) -> str:
    return f"link_{dataset_name.lower()}_{model_type.lower()}_{sigma}"


def get_link_stability_cache_path(cache_key: str) -> str:
    fname = f"stability_{cache_key.replace('.', '_').replace(' ', '_')}.parquet"
    return os.path.join(STABILITY_CACHE_DIR, fname)


def load_link_stability_cache(cache_key: str):
    stem = f"stability_{cache_key.replace('.', '_').replace(' ', '_')}"
    parquet_path = os.path.join(STABILITY_CACHE_DIR, stem + '.parquet')
    pkl_path     = os.path.join(STABILITY_CACHE_DIR, stem + '.pkl')

    if os.path.exists(parquet_path):
        try:
            df = pd.read_parquet(parquet_path)
            results = df.to_dict('records')
            print(f"[LINK STABILITY CACHE] Loaded parquet ({len(results)} edges): {parquet_path}")
            return results
        except Exception as e:
            print(f"[LINK STABILITY CACHE] Parquet read failed ({parquet_path}): {e}")

    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            print(f"[LINK STABILITY CACHE] Loaded pkl ({len(data)} edges): {pkl_path}")
            return data
        except Exception as e:
            print(f"[LINK STABILITY CACHE] Pkl read failed ({pkl_path}): {e}")

    return None


def save_link_stability_cache(cache_key: str, results: list) -> None:
    parquet_path = get_link_stability_cache_path(cache_key)
    try:
        pd.DataFrame(results).to_parquet(parquet_path, index=False)
        print(f"[LINK STABILITY CACHE] Saved parquet ({len(results)} edges): {parquet_path}")
    except Exception as e:
        print(f"[LINK STABILITY CACHE] Parquet save failed: {e}  — trying pkl")
        pkl_path = parquet_path.replace('.parquet', '.pkl')
        try:
            with open(pkl_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"[LINK STABILITY CACHE] Saved pkl (fallback): {pkl_path}")
        except Exception as e2:
            print(f"[LINK STABILITY CACHE] Both save methods failed: {e2}")


def link_stability_cache_exists(dataset_name: str, model_type: str, sigma: float) -> bool:
    key  = get_link_stability_cache_key(dataset_name, model_type, sigma)
    stem = f"stability_{key.replace('.', '_').replace(' ', '_')}"
    return (
        os.path.exists(os.path.join(STABILITY_CACHE_DIR, stem + '.parquet')) or
        os.path.exists(os.path.join(STABILITY_CACHE_DIR, stem + '.pkl'))
    )


def _scan_link_stability_cache() -> None:
    files = [f for f in os.listdir(STABILITY_CACHE_DIR)
             if f.startswith('stability_link_') and (f.endswith('.parquet') or f.endswith('.pkl'))]
    if not files:
        print("[LINK STABILITY CACHE] No precomputed files found in stability_cache/")
        print("  Run:  python precompute_link_stability.py")
        return
    print(f"[LINK STABILITY CACHE] {len(files)} precomputed file(s) available:")
    for fname in sorted(files):
        fpath = os.path.join(STABILITY_CACHE_DIR, fname)
        size_kb = os.path.getsize(fpath) // 1024
        try:
            if fname.endswith('.parquet'):
                n = len(pd.read_parquet(fpath))
            else:
                with open(fpath, 'rb') as fh:
                    n = len(pickle.load(fh))
            print(f"  ✓ {fname}  ({n} edges, {size_kb} KB)")
        except Exception:
            print(f"  ? {fname}  ({size_kb} KB, unreadable)")


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Edge mask extraction helper

def _get_edge_mask(explanation):
    mask = getattr(explanation, 'edge_mask', None)
    if mask is not None:
        return mask.detach().cpu().numpy()
    mask = explanation.get('edge_mask', None)
    if mask is not None:
        return mask.detach().cpu().numpy()
    for key in explanation.keys() if hasattr(explanation, 'keys') else []:
        if 'edge' in key.lower() and 'mask' in key.lower():
            val = explanation[key]
            if val is not None:
                print(f"[STABILITY] Found edge mask under key '{key}'")
                return val.detach().cpu().numpy()
    try:
        avail = list(explanation.keys()) if hasattr(explanation, 'keys') else dir(explanation)
        print(f"[STABILITY] WARNING — edge_mask not found. Available keys: {avail}")
    except Exception:
        pass
    return None


# Node-level explainer stability

def run_stability_analysis(model, data, predictions_data, sigma, sample_size):
    import traceback as _tb
    print(f"\n[STABILITY COMPUTE] Starting — sigma={sigma}, sample_size={sample_size}")
    model.eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    true_labels = data.y.cpu().numpy()

    preds_raw = predictions_data.get('preds', [])
    if not preds_raw:
        print("[STABILITY COMPUTE] ERROR: predictions_data['preds'] is empty!")
        return []
    preds = np.array(preds_raw)

    degrees = np.zeros(data.num_nodes, dtype=int)
    ei_cpu = edge_index.cpu().numpy()
    for src in ei_cpu[0]:
        degrees[src] += 1

    with torch.no_grad():
        out = model(x, edge_index)
        if out.min().item() < -20:
            probs = torch.exp(out).cpu().numpy()
        else:
            probs = torch.softmax(out, dim=1).cpu().numpy()

    np.random.seed(42)
    if sample_size == 0 or sample_size >= data.num_nodes:
        sample_nodes = np.arange(data.num_nodes)
    else:
        n_sample = int(sample_size)
        sample_nodes = np.random.choice(data.num_nodes, size=n_sample, replace=False)
    n_sample = len(sample_nodes)

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=20),
        explanation_type='model',
        edge_mask_type='object',
        model_config=dict(mode='multiclass_classification', task_level='node', return_type='log_probs')
    )

    results = []
    skip_reasons = {}

    for i, node_idx in enumerate(sample_nodes):
        node_idx = int(node_idx)
        try:
            target_cls = int(preds[node_idx])
            target = torch.tensor([target_cls], device=device)
            exp_orig = explainer(x=x, edge_index=edge_index, target=target, index=node_idx)
            mask_orig_np = _get_edge_mask(exp_orig)
            if mask_orig_np is None or len(mask_orig_np) == 0:
                continue
            top_k = min(10, len(mask_orig_np))
            top10_orig = set(np.argsort(mask_orig_np)[-top_k:].tolist())

            noise = torch.randn_like(x) * sigma
            x_pert = x + noise
            exp_pert = explainer(x=x_pert, edge_index=edge_index, target=target, index=node_idx)
            mask_pert_np = _get_edge_mask(exp_pert)
            if mask_pert_np is None or len(mask_pert_np) == 0:
                continue
            top10_pert = set(np.argsort(mask_pert_np)[-top_k:].tolist())

            intersection = len(top10_orig & top10_pert)
            union = len(top10_orig | top10_pert)
            jaccard = intersection / union if union > 0 else 0.0

            confidence = float(torch.softmax(model(x, edge_index), dim=1).cpu().numpy()[node_idx, target_cls])
            correct = bool(target_cls == int(true_labels[node_idx]))

            noise_node_np = noise[node_idx].cpu().numpy()
            denom = float(np.linalg.norm(noise_node_np)) + 1e-8
            if mask_orig_np.shape == mask_pert_np.shape:
                lipschitz = float(np.linalg.norm(mask_orig_np - mask_pert_np) / denom)
            else:
                lipschitz = float('nan')

            fidelity = None
            try:
                keep = torch.ones(edge_index.shape[1], dtype=torch.bool, device=device)
                keep[torch.tensor(list(top10_orig), dtype=torch.long, device=device)] = False
                ei_rem = edge_index[:, keep]
                with torch.no_grad():
                    lp_rem = model(x, ei_rem)
                    prob_rem = float(torch.exp(lp_rem[node_idx, target_cls]).item())
                fidelity = float(confidence - prob_rem)
            except Exception:
                pass

            results.append({
                'node_idx': node_idx,
                'confidence': confidence,
                'degree': int(degrees[node_idx]),
                'stability': float(jaccard),
                'lipschitz': float(lipschitz) if not (isinstance(lipschitz, float) and (lipschitz != lipschitz)) else None,
                'fidelity': fidelity,
                'correct': correct
            })

        except Exception:
            _tb.print_exc()
            continue

    return results


# Fast prediction-consistency stability

def run_prediction_consistency_stability(model, data, predictions_data, sigma,
                                         sample_size, n_trials=20):
    import traceback as _tb
    import math as _math

    model.eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    true_labels = data.y.cpu().numpy()

    preds_raw = predictions_data.get('preds', [])
    if not preds_raw:
        return []
    orig_preds = np.array(preds_raw)
    num_nodes = data.num_nodes

    degrees = np.zeros(num_nodes, dtype=int)
    ei_cpu = edge_index.cpu().numpy()
    for src in ei_cpu[0]:
        degrees[src] += 1

    with torch.no_grad():
        out_clean = model(x, edge_index)
        if out_clean.min().item() < -10:
            probs_clean = torch.exp(out_clean).cpu().numpy()
        else:
            probs_clean = torch.softmax(out_clean, dim=1).cpu().numpy()

    n_sample = min(int(sample_size), num_nodes)
    np.random.seed(42)
    if n_sample >= num_nodes:
        sample_nodes = np.arange(num_nodes)
    else:
        sample_nodes = np.random.choice(num_nodes, n_sample, replace=False)

    trial_preds = np.zeros((n_trials, len(sample_nodes)), dtype=int)
    collect_embeddings = hasattr(model, 'conv1')
    if collect_embeddings:
        emb_list = []

    for t in range(n_trials):
        noise = torch.randn_like(x) * sigma
        x_pert = x + noise
        with torch.no_grad():
            if collect_embeddings:
                try:
                    out_t, emb_t = model(x_pert, edge_index, return_embeddings=True)
                    emb_list.append(emb_t[sample_nodes].cpu().numpy())
                except TypeError:
                    out_t = model(x_pert, edge_index)
                    collect_embeddings = False
            else:
                out_t = model(x_pert, edge_index)

        pred_t = out_t.argmax(dim=1).cpu().numpy()
        trial_preds[t] = pred_t[sample_nodes]

    orig_for_sample = orig_preds[sample_nodes]
    stability_scores = (trial_preds == orig_for_sample[np.newaxis, :]).mean(axis=0)

    entropy_scores = np.zeros(len(sample_nodes))
    for i in range(len(sample_nodes)):
        cls_counts = np.bincount(trial_preds[:, i], minlength=int(orig_preds.max()) + 1)
        cls_probs = cls_counts / n_trials
        nz = cls_probs[cls_probs > 0]
        entropy_scores[i] = float(-np.sum(nz * np.log2(nz)))

    emb_var_scores = None
    if collect_embeddings and emb_list:
        emb_array = np.stack(emb_list, axis=0)
        emb_var_scores = emb_array.var(axis=0).mean(axis=1)

    results = []
    for i, node_idx in enumerate(sample_nodes):
        node_idx = int(node_idx)
        conf = float(probs_clean[node_idx, int(orig_preds[node_idx])])
        correct = bool(int(orig_preds[node_idx]) == int(true_labels[node_idx]))
        r = {
            'node_idx': node_idx,
            'confidence': conf,
            'degree': int(degrees[node_idx]),
            'stability': float(stability_scores[i]),
            'entropy': float(entropy_scores[i]),
            'correct': correct,
        }
        if emb_var_scores is not None:
            r['embedding_variance'] = float(emb_var_scores[i])
        for k, v in r.items():
            if isinstance(v, float) and (_math.isnan(v) or _math.isinf(v)):
                r[k] = None
        results.append(r)

    return results


# Link wrapper and per-edge stability (moved from app.py)
class _LinkPredWrapper(torch.nn.Module):
    """Wraps the base GNN so GNNExplainer can explain a single link (src→dst).
    forward(x, edge_index) → sigmoid score [1,1] for binary_classification task."""
    def __init__(self, base_model, src: int, dst: int):
        super().__init__()
        self.base_model = base_model
        self.src = int(src)
        self.dst = int(dst)

    def forward(self, x, edge_index, **kwargs):
        _, emb = self.base_model(x, edge_index, return_embeddings=True)
        ep = torch.tensor([[self.src], [self.dst]], device=x.device, dtype=torch.long)
        score = self.base_model.predict_links(emb, ep)
        return score.sigmoid().view(1, 1)


def run_link_stability_analysis(model, data, sigma: float, sample_size: int):
    """
    Per-edge stability routine. Returns list of dicts with keys:
        source, target, confidence, stability, degree_sum, common_neighbors, correct
    """
    import traceback as _tb

    print(f"\n[LINK STABILITY] Starting — sigma={sigma}, sample_size={sample_size}")
    model.eval()
    device_local = next(model.parameters()).device if any(p.numel() for p in model.parameters()) else torch.device('cpu')
    x = data.x.to(device_local)
    edge_index = data.edge_index.to(device_local)
    num_nodes = data.num_nodes
    ei_cpu = edge_index.cpu().numpy()

    # Out-degree per node
    degrees = np.zeros(num_nodes, dtype=int)
    for s in ei_cpu[0]:
        degrees[int(s)] += 1

    # Neighbour sets for common-neighbour count
    adj = [set() for _ in range(num_nodes)]
    for i in range(ei_cpu.shape[1]):
        u, v = int(ei_cpu[0, i]), int(ei_cpu[1, i])
        adj[u].add(v)
        adj[v].add(u)

    # Deduplicate to undirected edges
    seen, candidate_edges = set(), []
    for i in range(ei_cpu.shape[1]):
        u, v = int(ei_cpu[0, i]), int(ei_cpu[1, i])
        key = (min(u, v), max(u, v))
        if key not in seen:
            seen.add(key)
            candidate_edges.append((u, v))
    print(f"[LINK STABILITY] Unique undirected edges: {len(candidate_edges)}")

    # Sampling
    if sample_size == 0 or sample_size >= len(candidate_edges):
        sampled = candidate_edges
        print(f"[LINK STABILITY] Running on ALL {len(sampled)} edges")
    else:
        idx = np.random.choice(len(candidate_edges), size=int(sample_size), replace=False)
        sampled = [candidate_edges[i] for i in idx]
        print(f"[LINK STABILITY] Sampled {len(sampled)} edges")

    # Batch confidence for all sampled edges (single forward pass)
    with torch.no_grad():
        _, emb_clean = model(x, edge_index, return_embeddings=True)
        srcs = torch.tensor([u for u, v in sampled], device=device_local, dtype=torch.long)
        dsts = torch.tensor([v for u, v in sampled], device=device_local, dtype=torch.long)
        scores_batch = model.predict_links(emb_clean, torch.stack([srcs, dsts]))
        confs = torch.sigmoid(scores_batch).cpu().numpy().ravel()

    results = []
    n_ok = 0
    n_fail = 0
    skip_reasons = {}

    for i, (u, v) in enumerate(sampled):
        try:
            conf = float(confs[i])
            correct = conf > 0.5
            deg_sum = int(degrees[u]) + int(degrees[v])
            cn = len(adj[u].intersection(adj[v]))

            # Build per-edge GNNExplainer
            wrapper = _LinkPredWrapper(model, u, v)
            wrapper.eval()
            link_explainer = Explainer(
                model=wrapper,
                algorithm=GNNExplainer(epochs=20),
                explanation_type='model',
                edge_mask_type='object',
                model_config=dict(mode='binary_classification', task_level='graph', return_type='raw')
            )

            exp_orig = link_explainer(x=x, edge_index=edge_index)
            mask_orig = _get_edge_mask(exp_orig)

            noise = torch.randn_like(x) * sigma
            exp_pert = link_explainer(x=(x + noise), edge_index=edge_index)
            mask_pert = _get_edge_mask(exp_pert)

            if mask_orig is None or mask_pert is None:
                raise ValueError("Could not extract edge mask")

            k = min(10, len(mask_orig))
            top_o = set(np.argsort(mask_orig)[-k:].tolist())
            top_p = set(np.argsort(mask_pert)[-k:].tolist())
            union = top_o | top_p
            jaccard = len(top_o & top_p) / len(union) if union else 1.0

            results.append({
                'source': int(u),
                'target': int(v),
                'confidence': conf,
                'stability': float(jaccard),
                'degree_sum': deg_sum,
                'common_neighbors': cn,
                'correct': bool(correct),
            })
            n_ok += 1
            if (i + 1) % 25 == 0:
                print(f"[LINK STABILITY]   {i+1}/{len(sampled)} edges done")

        except Exception as e:
            n_fail += 1
            reason = type(e).__name__ + ': ' + str(e)[:60]
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            if n_fail <= 3:
                _tb.print_exc()

    print(f"[LINK STABILITY] Done — {n_ok} ok / {n_fail} failed")
    if skip_reasons:
        print(f"[LINK STABILITY] Failures: {skip_reasons}")
    return results

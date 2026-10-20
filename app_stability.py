"""
Stability analyses shared by the app and the precompute scripts: node and link
GNNExplainer stability, and the fast prediction-consistency routine.

The node and link analyses take an optional ``progress(done, total)`` callable,
called before each node / edge and once at the end; it may raise to stop the
run (gnnvis/jobs.py uses this for cancelling).
"""

import numpy as np
import torch

from gnnvis.explainers.gnnexplainer import (
    STABILITY_EPOCHS, LinkLogit, edge_mask, link_mask_explainer, model_device, node_mask_explainer,
)

# Caches of these results: gnnvis/stability_cache.py


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Shared with the app's explanations (gnnvis/explainers/gnnexplainer.py)
_get_edge_mask = edge_mask
_model_device = model_device
_LinkPredWrapper = LinkLogit


# Node-level explainer stability

def run_stability_analysis(model, data, predictions_data, sigma, sample_size, seed=42, progress_every=0,
                           progress=None):
    """Node-level GNNExplainer top-k Jaccard stability.

    This is the single implementation used by both the app and
    precompute_node_stability.py, so cached and live results are comparable.
    Lipschitz = ||mask_orig - mask_pert|| / ||noise||, with the noise norm taken
    over the whole feature matrix (noise is applied to every node).
    """
    import traceback as _tb
    print(f"\n[STABILITY COMPUTE] Starting — sigma={sigma}, sample_size={sample_size}")
    model.eval()
    dev = _model_device(model)
    x = data.x.to(dev)
    edge_index = data.edge_index.to(dev)
    true_labels = data.y.cpu().numpy()

    preds_raw = predictions_data.get('preds', [])
    if not preds_raw:
        print("[STABILITY COMPUTE] ERROR: predictions_data['preds'] is empty!")
        return []
    preds = np.array(preds_raw)

    degrees = np.bincount(edge_index[0].cpu().numpy(), minlength=data.num_nodes)

    # Models return log-probabilities; one clean forward pass serves every node.
    with torch.no_grad():
        probs = torch.exp(model(x, edge_index)).cpu().numpy()

    # Seed both numpy (node sampling) and torch (noise + GNNExplainer mask init)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if sample_size == 0 or sample_size >= data.num_nodes:
        sample_nodes = np.arange(data.num_nodes)
    else:
        sample_nodes = rng.choice(data.num_nodes, size=int(sample_size), replace=False)
    n_sample = len(sample_nodes)

    explainer = node_mask_explainer(model, epochs=STABILITY_EPOCHS)

    results = []
    skip_reasons = {}

    for i, node_idx in enumerate(sample_nodes):
        if progress:
            progress(i, n_sample)
        node_idx = int(node_idx)
        try:
            target_cls = int(preds[node_idx])
            target = torch.tensor([target_cls], device=dev)
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

            confidence = float(probs[node_idx, target_cls])
            correct = bool(target_cls == int(true_labels[node_idx]))

            denom = float(torch.norm(noise).item())
            if mask_orig_np.shape == mask_pert_np.shape and denom > 1e-9:
                lipschitz = float(np.linalg.norm(mask_orig_np - mask_pert_np) / denom)
            else:
                lipschitz = float('nan')

            fidelity = None
            try:
                keep = torch.ones(edge_index.shape[1], dtype=torch.bool, device=dev)
                keep[torch.tensor(list(top10_orig), dtype=torch.long, device=dev)] = False
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

            if progress_every and ((i + 1) % progress_every == 0 or i + 1 == n_sample):
                print(f"  [{i + 1}/{n_sample}] done — {len(results)} results so far", flush=True)

        except Exception:
            _tb.print_exc()
            continue

    if progress:
        progress(n_sample, n_sample)
    return results


# Fast prediction-consistency stability

def run_prediction_consistency_stability(model, data, predictions_data, sigma,
                                         sample_size, n_trials=20):
    import traceback as _tb
    import math as _math

    model.eval()
    dev = _model_device(model)
    x = data.x.to(dev)
    edge_index = data.edge_index.to(dev)
    true_labels = data.y.cpu().numpy()

    preds_raw = predictions_data.get('preds', [])
    if not preds_raw:
        return []
    orig_preds = np.array(preds_raw)
    num_nodes = data.num_nodes

    degrees = np.bincount(edge_index[0].cpu().numpy(), minlength=num_nodes)

    # Models return log-probabilities
    with torch.no_grad():
        probs_clean = torch.exp(model(x, edge_index)).cpu().numpy()

    # 0 (or None) means all nodes, consistent with run_stability_analysis
    n_sample = int(sample_size or 0)
    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    if n_sample <= 0 or n_sample >= num_nodes:
        sample_nodes = np.arange(num_nodes)
    else:
        sample_nodes = rng.choice(num_nodes, n_sample, replace=False)

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


# Per-edge stability

def run_link_stability_analysis(model, data, sigma: float, sample_size: int, seed=42, progress_every=25,
                                progress=None):
    """
    Per-edge stability routine. Returns list of dicts with keys:
        source, target, confidence, stability, degree_sum, common_neighbors, correct
    """
    import traceback as _tb

    print(f"\n[LINK STABILITY] Starting — sigma={sigma}, sample_size={sample_size}")
    model.eval()
    device_local = _model_device(model)
    x = data.x.to(device_local)
    edge_index = data.edge_index.to(device_local)
    num_nodes = data.num_nodes
    ei_cpu = edge_index.cpu().numpy()

    # Seed edge sampling, noise and GNNExplainer mask init
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Out-degree per node
    degrees = np.bincount(ei_cpu[0], minlength=num_nodes)

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
        idx = rng.choice(len(candidate_edges), size=int(sample_size), replace=False)
        sampled = [candidate_edges[i] for i in idx]
        print(f"[LINK STABILITY] Sampled {len(sampled)} edges")

    # Batch confidence for all sampled edges (single forward pass)
    with torch.no_grad():
        _, emb_clean = model(x, edge_index, return_embeddings=True)
        srcs = torch.tensor([u for u, v in sampled], device=device_local, dtype=torch.long)
        dsts = torch.tensor([v for u, v in sampled], device=device_local, dtype=torch.long)
        if getattr(model, 'enable_link_prediction', False):
            scores_batch = model.predict_links(emb_clean, torch.stack([srcs, dsts]))
        else:
            scores_batch = (emb_clean[srcs] * emb_clean[dsts]).sum(dim=1)
        confs = torch.sigmoid(scores_batch).cpu().numpy().ravel()

    results = []
    n_ok = 0
    n_fail = 0
    skip_reasons = {}

    for i, (u, v) in enumerate(sampled):
        if progress:
            progress(i, len(sampled))
        try:
            conf = float(confs[i])
            correct = conf > 0.5
            deg_sum = int(degrees[u]) + int(degrees[v])
            cn = len(adj[u].intersection(adj[v]))

            # Build per-edge GNNExplainer
            link_explainer = link_mask_explainer(model, u, v, epochs=STABILITY_EPOCHS)

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
            if progress_every and (i + 1) % progress_every == 0:
                print(f"[LINK STABILITY]   {i+1}/{len(sampled)} edges done", flush=True)

        except Exception as e:
            n_fail += 1
            reason = type(e).__name__ + ': ' + str(e)[:60]
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            if n_fail <= 3:
                _tb.print_exc()

    if progress:
        progress(len(sampled), len(sampled))
    print(f"[LINK STABILITY] Done — {n_ok} ok / {n_fail} failed")
    if skip_reasons:
        print(f"[LINK STABILITY] Failures: {skip_reasons}")
    return results

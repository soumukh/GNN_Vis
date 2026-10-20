"""Precomputed stability results (stability_cache/), versioned.

Results files keep their formats and names:

    node  stability_{dataset}_{model}_{sigma}.pkl          list of dicts
    link  stability_link_{dataset}_{model}_{sigma}.parquet (or .pkl)

Each has a sidecar ``<file>.meta.json`` saying what produced it, including a
fingerprint of the model weights. A file whose metadata does not match the
current model (e.g. it was computed before the model was retrained) is stale
and is not used.

Files written before the metadata existed (legacy) are checked once against
the current model: their stored confidences must equal the model's. If they
do, the file is stamped with metadata; otherwise it is stale.
"""

import hashlib
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import torch

from gnnvis.explainers.gnnexplainer import STABILITY_EPOCHS

SCHEMA = 1
SEED = 42
NODE, LINK = 'node', 'link'
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'stability_cache')

# load() states
HIT, STAMPED, STALE, MISSING = 'hit', 'stamped', 'stale', 'missing'
# metadata fields that must match for a file to be used
_MATCH = ('schema', 'kind', 'dataset', 'model_type', 'sigma', 'model_fingerprint', 'explainer', 'epochs', 'seed')


def model_fingerprint(package):
    """Short hash of a package's model weights (changes whenever it is retrained)."""
    h = hashlib.sha256()
    for name, value in sorted((package.get('model_state_dict') or {}).items()):
        h.update(name.encode())
        h.update(torch.as_tensor(value).detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def expected_meta(kind, dataset_name, model_type, sigma, fingerprint):
    return {'schema': SCHEMA, 'kind': kind, 'dataset': dataset_name.lower(), 'model_type': model_type.lower(),
            'sigma': float(sigma), 'model_fingerprint': fingerprint, 'explainer': 'GNNExplainer',
            'epochs': STABILITY_EPOCHS, 'seed': SEED}


def _stem(kind, dataset_name, model_type, sigma):
    key = f"{dataset_name.lower()}_{model_type.lower()}_{sigma}"
    if kind == LINK:
        key = 'link_' + key
    return os.path.join(CACHE_DIR, f"stability_{key.replace('.', '_').replace(' ', '_')}")


def results_path(kind, dataset_name, model_type, sigma):
    """The results file to read (an existing one) or to write."""
    stem = _stem(kind, dataset_name, model_type, sigma)
    if kind == NODE:
        return stem + '.pkl'
    if not os.path.exists(stem + '.parquet') and os.path.exists(stem + '.pkl'):
        return stem + '.pkl'
    return stem + '.parquet'


def meta_path(path):
    return path + '.meta.json'


def _read_results(path):
    if path.endswith('.parquet'):
        return pd.read_parquet(path).to_dict('records')
    with open(path, 'rb') as f:
        return pickle.load(f)


def _write_meta(path, meta, n_results):
    with open(meta_path(path), 'w', encoding='utf-8') as f:
        json.dump({**meta, 'n_results': n_results, 'written': time.strftime('%Y-%m-%d %H:%M:%S')}, f, indent=1)


def load(kind, dataset_name, model_type, sigma, fingerprint, verify=None):
    """(results, state, path). results is None unless state is HIT or STAMPED.

    verify(results) -> bool decides whether a legacy file (no metadata) was
    produced by the current model; without it legacy files count as stale."""
    path = results_path(kind, dataset_name, model_type, sigma)
    if not os.path.exists(path):
        return None, MISSING, path
    expected = expected_meta(kind, dataset_name, model_type, sigma, fingerprint)
    try:
        results = _read_results(path)
    except Exception as e:
        print(f"[STABILITY CACHE] Unreadable ({path}): {e}")
        return None, STALE, path

    if os.path.exists(meta_path(path)):
        try:
            with open(meta_path(path), encoding='utf-8') as f:
                meta = json.load(f)
        except Exception:
            meta = {}
        diff = [k for k in _MATCH if meta.get(k) != expected[k]]
        if diff:
            print(f"[STABILITY CACHE] Stale ({os.path.basename(path)}): {', '.join(diff)} differ")
            return None, STALE, path
        return results, HIT, path

    if verify is not None and results and verify(results):
        _write_meta(path, expected, len(results))
        print(f"[STABILITY CACHE] Legacy file matches the current model; stamped {os.path.basename(path)}")
        return results, STAMPED, path
    print(f"[STABILITY CACHE] Legacy file does not match the current model: {os.path.basename(path)}")
    return None, STALE, path


def save(kind, dataset_name, model_type, sigma, fingerprint, results):
    """Write results and their metadata; returns the results path."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    stem = _stem(kind, dataset_name, model_type, sigma)
    path = stem + ('.pkl' if kind == NODE else '.parquet')
    if path.endswith('.parquet'):
        pd.DataFrame(results).to_parquet(path, index=False)
    else:
        with open(path, 'wb') as f:
            pickle.dump(results, f)
    _write_meta(path, expected_meta(kind, dataset_name, model_type, sigma, fingerprint), len(results))
    print(f"[STABILITY CACHE] Saved {len(results)} results: {path}")
    return path


# ── legacy verification: stored confidences must be the current model's ──

@torch.no_grad()
def node_results_match(model, data, results, atol=1e-4):
    """Node results store confidence = p(predicted class) for node_idx."""
    model.eval()
    probs = torch.exp(model(data.x, data.edge_index)).cpu().numpy()
    idx = np.array([int(r['node_idx']) for r in results])
    stored = np.array([float(r['confidence']) for r in results])
    current = probs[idx, probs[idx].argmax(axis=1)]
    return bool(np.allclose(stored, current, atol=atol))


@torch.no_grad()
def link_results_match(model, data, results, atol=1e-4):
    """Link results store confidence = sigmoid(link score) for (source, target)."""
    model.eval()
    _, emb = model(data.x, data.edge_index, return_embeddings=True)
    pairs = torch.tensor([[int(r['source']), int(r['target'])] for r in results], dtype=torch.long).t()
    current = torch.sigmoid(model.predict_links(emb, pairs.to(emb.device))).cpu().numpy().ravel()
    stored = np.array([float(r['confidence']) for r in results])
    return bool(np.allclose(stored, current, atol=atol))


def scan():
    """Startup summary of the files in stability_cache/."""
    if not os.path.isdir(CACHE_DIR):
        print("[STABILITY CACHE] No stability_cache/ folder; run precompute_node_stability.py")
        return
    files = sorted(f for f in os.listdir(CACHE_DIR) if f.startswith('stability_') and f.endswith(('.pkl', '.parquet')))
    if not files:
        print("[STABILITY CACHE] No precomputed files; run precompute_node_stability.py / precompute_link_stability.py")
        return
    print(f"[STABILITY CACHE] {len(files)} precomputed file(s):")
    for fname in files:
        tag = 'versioned' if os.path.exists(meta_path(os.path.join(CACHE_DIR, fname))) else 'legacy, checked on first use'
        print(f"  {fname}  ({tag})")

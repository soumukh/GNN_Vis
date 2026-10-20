"""Runtime state shared by the callbacks: paths, device, caches of model packages / datasets / models, the model lock, and the Plotly Express thread-safety lock."""

import functools
import numpy as np
import os
import plotly.express as px
import plotly.io as pio
import threading
import torch
import traceback
from torch_geometric.data import Data

from gnnvis.models import load_model, load_package


MODEL_DIR = 'models'


DATA_DIR = 'data'


# --- Plotly Express thread safety ---
# Dash's dev server runs callbacks in parallel threads. Plotly Express reads
# shared template objects, and two concurrent calls can fail with
# "ValueError: Invalid value" (seen at page load: it blanked the embedding plot
# AND the confusion matrix). Serialise px calls and load templates up front.
_PX_LOCK = threading.RLock()


def _px_locked(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _PX_LOCK:
            return fn(*args, **kwargs)
    return wrapper


for _px_name in ('scatter', 'bar', 'line', 'histogram', 'density_heatmap', 'density_contour', 'imshow'):
    setattr(px, _px_name, _px_locked(getattr(px, _px_name)))


for _tpl in ('plotly', 'plotly_white'):
    pio.templates[_tpl]  # force the lazy template load on the main thread


# Project root (this file is gnnvis/runtime.py): models/ and data/ live there
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


MODEL_ABS_DIR = os.path.join(SCRIPT_DIR, MODEL_DIR)


DATA_ABS_DIR = os.path.join(SCRIPT_DIR, DATA_DIR)


# --- Server-side cache helpers ---
PACKAGE_CACHE = {}


DATASET_CACHE = {}


MODEL_CACHE = {}


DATA_TENSOR_CACHE = {}


# The cached model is shared by all callbacks. PyG's Explainer temporarily
# injects edge masks into its conv layers, so callbacks that run concurrently
# (RUN triggers GNNExplainer and CPA-IV together) corrupt each other's forward
# passes (AssertionError in message passing). Run model-using callbacks one at a time.
MODEL_LOCK = threading.RLock()


# Without a precomputed file, live link stability takes ~0.5-1 s per edge
# (Cora: 5278 edges, ~45 min to 2 h). Live runs use a seeded sample (~1-2 min).
LINK_STABILITY_LIVE_SAMPLE = 100


def private_model(package_meta, num_features, num_classes):
    """A fresh model instance, not the shared cached one. Long stability runs
    use it so they need not hold MODEL_LOCK (and block every other analysis)
    for minutes: GNNExplainer's injected masks then only touch this copy."""
    package = get_cached_package_or_load(package_meta)
    if not package:
        return None
    return instantiate_model(package, package_meta.get('model_type'), num_features, num_classes)


def serialize_model_use(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with MODEL_LOCK:
            return fn(*args, **kwargs)
    return wrapper


def get_package_cache_key(model_type: str, dataset_name: str):
    return (model_type, dataset_name)


def cache_package(package, model_type: str, dataset_name: str):
    PACKAGE_CACHE[get_package_cache_key(model_type, dataset_name)] = package
    return package


def get_cached_package(package_meta):
    if not package_meta:
        return None
    key = get_package_cache_key(package_meta.get('model_type'), package_meta.get('dataset_name'))
    return PACKAGE_CACHE.get(key)


def load_dataset_into_cache(dataset_name: str):
    if dataset_name in DATASET_CACHE:
        return DATASET_CACHE[dataset_name]
    from torch_geometric.datasets import Planetoid
    dataset_obj = Planetoid(root=DATA_ABS_DIR, name=dataset_name)
    data = dataset_obj[0]
    full_dataset = {
        'name': dataset_name,
        'x': data.x.cpu(),
        'y': data.y.cpu(),
        'num_nodes': int(data.num_nodes),
        'num_features': int(data.num_node_features),
        'num_classes': int(dataset_obj.num_classes),
        'original_edge_index': data.edge_index.cpu(),
        'test_mask': data.test_mask.cpu(),
    }
    DATASET_CACHE[dataset_name] = full_dataset
    return full_dataset


def get_cached_dataset(dataset_meta):
    if not dataset_meta or 'name' not in dataset_meta:
        return None
    dataset_name = dataset_meta.get('name')
    if dataset_name in DATASET_CACHE:
        return DATASET_CACHE[dataset_name]
    return load_dataset_into_cache(dataset_name)


def get_dataset_tensors(dataset_meta):
    dataset = get_cached_dataset(dataset_meta)
    if not dataset:
        return None
    cache_key = dataset['name']
    if cache_key in DATA_TENSOR_CACHE:
        return DATA_TENSOR_CACHE[cache_key]
    tensors = {
        'x': dataset['x'].to(device),
        'y': dataset['y'].to(device)
    }
    DATA_TENSOR_CACHE[cache_key] = tensors
    return tensors


def get_cached_model(package_meta, num_features, num_classes):
    if not package_meta:
        return None
    package = get_cached_package(package_meta)
    if not package:
        return None
    cache_key = (package_meta.get('model_type'), package_meta.get('dataset_name'), num_features, num_classes)
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]
    model = instantiate_model(package, package_meta.get('model_type'), num_features, num_classes)
    if model is not None:
        MODEL_CACHE[cache_key] = model
    return model


def prepare_torch_data(dataset_meta, graph_data=None, include_labels=True):
    dataset = get_cached_dataset(dataset_meta)
    if not dataset:
        return None

    x = dataset['x'].to(device)
    if graph_data and graph_data.get('edge_index') is not None:
        edge_index = torch.tensor(graph_data['edge_index'], dtype=torch.long)
    else:
        edge_index = dataset['original_edge_index']

    data_kwargs = {
        'x': x,
        'edge_index': edge_index.to(device),
        'num_nodes': dataset['num_nodes']
    }
    if include_labels:
        data_kwargs['y'] = dataset['y'].to(device)
    return Data(**data_kwargs)


def ensure_dataset_loaded(dataset_meta):
    return get_cached_dataset(dataset_meta)


def package_meta_from_selection(model_type, dataset_name):
    return {'model_type': model_type, 'dataset_name': dataset_name}


def dataset_meta_from_data(dataset_name, data):
    return {
        'name': dataset_name,
        'num_nodes': int(data.num_nodes),
        'num_features': int(data.num_node_features),
        'num_classes': int(data.num_classes),
        'original_edge_index': data.edge_index.cpu().tolist(),
    }


def get_cached_package_and_dataset(package_meta, dataset_meta):
    return get_cached_package(package_meta), get_cached_dataset(dataset_meta)


def cache_full_dataset(dataset_name: str, full_dataset):
    DATASET_CACHE[dataset_name] = full_dataset
    return full_dataset


def cache_package_if_missing(package, model_type: str, dataset_name: str):
    if package is None:
        return None
    key = get_package_cache_key(model_type, dataset_name)
    if key not in PACKAGE_CACHE:
        PACKAGE_CACHE[key] = package
    return package


def get_cached_package_if_missing(package_meta):
    return get_cached_package_or_load(package_meta)


def get_attention_weights(package_meta):
    package = get_cached_package_if_missing(package_meta)
    if isinstance(package, dict):
        return package.get('attention_weights')
    return None


def get_cached_package_or_load(package_meta):
    """Return cached package or load it from disk and cache it."""
    if not package_meta:
        return None
    package = get_cached_package(package_meta)
    if package is None:
        package = load_precomputed_package(package_meta.get('model_type'), package_meta.get('dataset_name'))
        if package is not None:
            cache_package(package, package_meta.get('model_type'), package_meta.get('dataset_name'))
    return package


def get_cached_dataset_if_missing(dataset_meta):
    return get_cached_dataset(dataset_meta)


def get_package_target_identifier(package_meta):
    return f"{package_meta.get('model_type')}::{package_meta.get('dataset_name')}"


# --- CPU Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(f"--- GNN Explainer App v2 ---")


print(f"Using device: {device}")


def load_precomputed_package(model_type, dataset_name):
    return load_package(model_type, dataset_name, MODEL_ABS_DIR)


def instantiate_model(package, model_type, num_features, num_classes, verbose=False):
    """The model saved in `package`, on `device` in eval mode; None on failure."""
    if not package:
        return None
    try:
        return load_model(package, model_type, num_features, num_classes, device=device)
    except Exception as e:
        print(f"ERROR: Could not instantiate {model_type} model. Error: {e}")
        traceback.print_exc()
        return None


_LINK_SPLIT_CACHE = {}  # dataset_name -> held-out test split used in training


def link_test_split(dataset_name):
    """The link-prediction train/test split used by train_models.py:
    create_edge_splits(val_ratio=0.05, test_ratio=0.15, random_state=42) on the
    Planetoid graph. It seeds the global torch/numpy/random RNGs, so their state
    is saved and restored around the call."""
    if dataset_name not in _LINK_SPLIT_CACHE:
        from link_prediction_utils import create_edge_splits
        full = get_cached_dataset({'name': dataset_name})
        data = Data(x=full['x'], edge_index=full['original_edge_index'], num_nodes=full['num_nodes'])
        import random as _random
        torch_state, np_state, py_state = torch.get_rng_state(), np.random.get_state(), _random.getstate()
        try:
            split = create_edge_splits(data, val_ratio=0.05, test_ratio=0.15, random_state=42)
        finally:
            torch.set_rng_state(torch_state)
            np.random.set_state(np_state)
            _random.setstate(py_state)
        _LINK_SPLIT_CACHE[dataset_name] = {
            'train_edge_index': split['train_edge_index'],
            'test_pos': split['test_pos_edges'].t().tolist(),
            'test_neg': split['test_neg_edges'].t().tolist(),
        }
    return _LINK_SPLIT_CACHE[dataset_name]

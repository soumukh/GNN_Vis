"""
TASK MAP (high-level)
----------------------
This file is organized into named TASK sections to make it easy for
contributors to find, modify, and reason about the code. Each TASK is
annotated with an explanatory header (purpose, inputs, outputs, side-effects)
so you can make targeted changes without wading through unrelated logic.

Tasks:
    1) Imports & Dependencies     - external libraries and local utilities
    2) Configuration             - warnings, paths, constants
    3) Stability Cache helpers   - file-based caches for stability results
    4) Server-side caches        - in-memory caches for packages/datasets/models
    5) Device setup              - torch device selection
    6) Model definitions         - `GCNNet`, `GATNet`, and link heads
    7) Data Loading & Model I/O  - load precomputed packages, instantiate models
    8) Explanation & Inference   - helpers to run model inference and explainers
    9) Stability Backends        - node/link stability compute routines
 10) Link Stability Cache      - parquet/pickle cache helpers
 11) Visualization helpers     - subgraph extraction and cytoscape rendering
 12) Dash app layout & callbacks - UI wiring (bottom of file)

When editing, keep each TASK focused and avoid mixing responsibilities.
If you split logic into new files, update the TASK MAP above and add
import stubs at the top.
"""

# torch must be imported before dash: on Windows, loading dash first makes
# torch's c10.dll fail to initialise (WinError 1114).
import torch

import dash

from dash import dcc, html, Input, Output, State, callback, ctx, no_update, ALL

import dash.dependencies

import dash_cytoscape as cyto

import plotly.express as px

import plotly.io as pio

import plotly.graph_objects as go

import torch

import torch.nn.functional as F


from torch_geometric.data import Data

from torch_geometric.utils import k_hop_subgraph, to_networkx, remove_self_loops

from torch_geometric.explain import Explainer, GNNExplainer

import pandas as pd

import numpy as np



import pickle

import os
import functools

import traceback

import warnings

import sys

import threading

from link_prediction_utils import create_link_prediction_candidates, get_top_k_predictions, generate_negative_samples

from gnnvis.models import load_package, load_model, available_models, model_capabilities

# View code (figures, components, theme): gnnvis/ui
from gnnvis.ui.components import cell_details_placeholder, confusion_matrix_style, explanation_cards, metric_tiles, stability_unsupported_message, subgraph_color_key
from gnnvis.ui.figures.attention import create_attention_panel
from gnnvis.ui.figures.confusion import _render_cm_detail_page, link_confusion_figure, node_confusion_figure
from gnnvis.ui.figures.cpa import create_cpa_iv_panel
from gnnvis.ui.figures.embeddings import get_2d_embeddings, plot_embeddings
from gnnvis.ui.figures.full_graph import build_full_graph_figure
from gnnvis.ui.figures.importance import compute_neighbor_importance_from_pyg, get_one_hop_neighbors, plot_feature_importance, plot_neighbor_importance
from gnnvis.ui.figures.stability import build_drilldown_scatter, build_region_highlight, build_stability_plots
from gnnvis.ui.figures.subgraph import STYLESHEET, create_subgraph_cytoscape
from gnnvis.ui.theme import CORRECT_COLOR, INCORRECT_COLOR, class_color_map, class_label, class_name, empty_figure

from cpa_iv_link_prediction import run_cpa_iv_link

# Logs contain emoji/arrows: when stdout is redirected on Windows the default
# cp1252 encoding raises UnicodeEncodeError inside callbacks (e.g. aborting the
# model load). Force UTF-8 and never fail on a print.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Fix for threading shutdown errors

sys.stdout.flush()

sys.stderr.flush()

# Stability helpers moved to separate module
from app_stability import (
    STABILITY_CACHE_DIR,
    get_stability_cache_key,
    get_stability_cache_path,
    load_stability_cache,
    save_stability_cache,
    link_stability_cache_exists,
    _scan_link_stability_cache,
    get_link_stability_cache_key,
    get_link_stability_cache_path,
    load_link_stability_cache,
    save_link_stability_cache,
    _get_edge_mask,
    run_stability_analysis,
    run_link_stability_analysis,
    _LinkPredWrapper,
)

# Explanation helpers moved to app_expl.py
from app_expl import (
    run_inference,
    explain_link_prediction,
    run_cpa_iv,
    _path_fingerprints,
)

# --- TASK: Configuration ---
# Purpose: set global warnings, constants, and path variables used across the app.
# Inputs: none (reads environment + project layout)
# Outputs: module-level constants like `MODEL_DIR`, `DATA_DIR`, `STABILITY_CACHE_DIR`
# Side-effects: creates `stability_cache` directory if missing

warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings("ignore", category=FutureWarning)

# Suppress Dash callback graph warnings

warnings.filterwarnings("ignore", message="Can not create edge")

# --- TASK: Global Variables & Setup ---
# Purpose: declare global constants and simple look-up lists used by the app.
# Inputs: none
# Outputs: `MODEL_DIR`, `DATA_DIR`, `DEFAULT_*`, `AVAILABLE_*`
# Notes: Keep these lightweight; larger configuration should be moved to a
#       dedicated config file (e.g., `config.py`) when the project grows.

# --- Global Variables & Setup ---

MODEL_DIR = 'models'

DATA_DIR = 'data'

DEFAULT_DATASET = 'Cora'

DEFAULT_MODEL = 'GCN'

AVAILABLE_DATASETS = ['Cora', 'CiteSeer']

AVAILABLE_MODELS = available_models()   # every model registered in gnnvis/models

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
















# --- TASK: Path Management ---
# Purpose: determine absolute paths for models, data and other artifacts.
# Inputs: current script location (`__file__`) or CWD when running interactively.
# Outputs: `SCRIPT_DIR`, `MODEL_ABS_DIR`, `DATA_ABS_DIR`
# Notes: If workspace layout changes, update these derivations accordingly.

# --- Path Management ---

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

MODEL_ABS_DIR = os.path.join(SCRIPT_DIR, MODEL_DIR)

DATA_ABS_DIR = os.path.join(SCRIPT_DIR, DATA_DIR)

# --- TASK: Stability Cache (file-backed) ---
# Purpose: read/write stability precomputations to disk to avoid expensive
#          re-computation during interactive exploration.
# Inputs: cache key (dataset/model/sigma), results list
# Outputs: pickle/parquet files under `stability_cache/`
# Side-effects: creates `stability_cache` dir if missing; prints logging

# --- Stability Cache ---

# Stability cache and compute helpers moved to `app_stability.py`.
# See `app_stability.py` for: get_stability_cache_key, load/save, and compute routines.


# --- TASK: Server-side cache helpers ---
# Purpose: in-memory caches to avoid repeated disk reads and model instantiation.
# Inputs: package/dataset identifiers
# Outputs: returns cached objects or loads & caches them on-demand
# Notes: these caches are process-local; consider redis/memcached for multi-process

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


# Link stability cache helpers (link_stability_cache_exists, _scan_link_stability_cache)
# live in `app_stability.py` and are imported above.
# Use functions from the module instead of local definitions.


# --- TASK: Device setup ---
# Purpose: choose compute device (CPU/GPU) and expose as module-level `device`.
# Inputs: environment GPU availability via `torch.cuda.is_available()`
# Outputs: `device` variable used across tensor operations and model placement
# Notes: For testing or CI, you can override to CPU by setting `device = torch.device('cpu')`

# --- CPU Device Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"--- GNN Explainer App v2 ---")

print(f"Using device: {device}")

#print("Forced to use CPU for compatibility")

# --- TASK: Models ---
# Model classes, the model registry and package loading live in gnnvis/models:
# the single definition shared with train_models.py and the precompute
# scripts. These wrappers keep the app's None-on-failure convention.

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

# --- TASK: Explanation & Inference ---
# Purpose: provide safe wrappers around model forward passes and explainer
#          invocations (GNNExplainer, link explainers, CPA-IV helper).
# Inputs: model, data, node/edge indices
# Outputs: JSON-serializable dicts with predictions, confidences, explanations
# Notes: Keep these functions deterministic when possible (seed RNGs) and
#        avoid returning non-serializable objects from callbacks.

# --- Explanation & Inference ---

# Explanation helpers moved to app_expl.py

# --- TASK: Link stability cache helpers ---
# Purpose: read/write per-edge stability results (parquet preferred, pkl fallback).
# Inputs: cache_key (dataset/model/sigma), results list
# Outputs: DataFrame/parquet or legacy pickle files on disk
# Notes: Precompute scripts drop parquet here for fast UI loads.

# ── Link Stability Cache ───────────────────────────────────────────────────────


# Link stability cache functions moved to `app_stability.py`.


# --- TASK: Link stability (per-edge) computations ---
# Purpose: compute per-edge stability S(u,v) by running GNNExplainer for the
#          target edge on clean and noisy inputs; returns per-edge metrics.
# Inputs: model, data, sigma, sample_size
# Outputs: list of dicts per edge with `source`, `target`, `stability`, ...
# Notes: This is expensive; use caching and sampling to keep interactive latency low.

# ── Link Stability: GNNExplainer wrapper for a single edge (u→v) ──────────────

# Link stability computation moved to `app_stability.py`.


# --- TASK: Fast prediction-consistency stability backend ---
# Purpose: fast approximation of stability by running `n_trials` noisy
#          forward passes and measuring prediction consistency per node.
# Inputs: model, data, predictions_data, sigma, sample_size, n_trials
# Outputs: list of per-node dicts (stability, entropy, embedding variance)
# Notes: Much faster than per-node explainer; suitable for interactive UIs.

# --- Fast Prediction-Consistency Stability Backend ---

# Fast prediction-consistency stability moved to `app_stability.py`.


# --- TASK: Path utilities for CPA-IV ---
# Purpose: helper functions to normalize and fingerprint CPA (causal path)
#          outputs so they are comparable and easily highlighted in the UI.
# Inputs: lists/dicts describing paths
# Outputs: hashable fingerprints or transformed path data

# --- Path Utilities (used by CPA-IV link explanation) ---

# Path utilities moved to app_expl.py


# --- TASK: Visualization helpers ---
# Purpose: construct subgraph views, Cytoscape node/edge payloads, and
#          extract visualization-focused attributes (colors, sizes, highlights).
# Inputs: `data`, `center_node_idx`, `explanation`, `cpa_data`, `link_predictions`
# Outputs: lists/dicts suitable for Dash Cytoscape and Plotly rendering
# Notes: Keep visualization logic UI-focused; avoid embedding heavy compute.

# --- Visualization Helpers ---





































# --- Dash App ---

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[

    'https://codepen.io/chriddyp/pen/bWLwgP.css',

    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css',

    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'

])

# Enable callback exception suppression for dynamic components

app.config.suppress_callback_exceptions = True

server = app.server

app.title = "🧠 Interactive GNN Explainer"


app.layout = html.Div(style={

    'fontFamily': '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',

    'backgroundColor': '#f5f7fb',

    'minHeight': '100vh',

    'margin': 0,

    'padding': 0,

    'color': '#333'

}, children=[

    dcc.Location(id='url', refresh=True),

    html.Div(style={'maxWidth': '100%', 'margin': '0 auto', 'padding': '20px'}, children=[

    dcc.Store(id='precomputed-package-store'),

    dcc.Store(id='full-dataset-store'),

    dcc.Store(id='editable-graph-store'),

    dcc.Store(id='current-predictions-store'),

    dcc.Store(id='selected-node-store'),

    dcc.Store(id='explanation-store'),

    dcc.Store(id='cpa-iv-store'),

    dcc.Store(id='link-cpa-iv-store'),
    
    dcc.Store(id='selected-path-store'),

    # Placeholder removed: actual header component is defined in Layer 3

    dcc.Store(id='selected-link-store'),

    dcc.Store(id='analysis-mode-store', data='node_classification'),

    dcc.Store(id='graph-type-store', data='full'),

    dcc.Store(id='removed-edges-store', data=None),

    dcc.Store(id='edge-status-store', data=None),

    dcc.Store(id='link-status-store', data=None),

    dcc.Store(id='cpa-status-store', data=None),

    dcc.Store(id='subgraph-eval-store', data=None),
    dcc.Store(id='cm-detail-paging-store', data=None),

    dcc.Store(id='stability-results-store', data=None),

    dcc.Store(id='drilldown-click-store', data=None),

    dcc.Store(id='link-prediction-store'),

    # Hidden placeholder components

    html.Div(style={'display': 'none'}, children=[

        dcc.Input(id='link-source-input', value=None),

        dcc.Input(id='link-target-input', value=None),

        html.Button(id='analyze-link-button', n_clicks=0),

        html.Button(id='predict-links-button', n_clicks=0),

        dcc.Input(id='select-node-input-nc', value=None),

        html.Button(id='run-cpa-iv-button', n_clicks=0),

        html.Button(id='add-edge-button-lp', n_clicks=0),

        html.Button(id='remove-edge-button-lp', n_clicks=0),

    ]),

    # --- HEADER: title + live status / next-step guidance ---
    html.Div(className='app-header', children=[
        html.Div(className='app-brand', children=[
            html.H1("Interactive GNN Explainer", className='app-title'),
            html.Span("Visual analytics for GCN / GAT predictions, explanations and stability",
                      className='app-subtitle'),
        ]),
        html.Div(id='status-bar', className='status-bar'),
    ]),

    # --- LAYER 1: TOP (USER CONTROLS ONLY) ---

    html.Div(id='layer-1', style={'borderBottom': 'none', 'paddingTop': '1px', 'paddingBottom': '15px', 'marginBottom': '20px'}, children=[

        # Layer label
        #html.Div(style={'marginBottom': '10px', 'display': 'flex', 'alignItems': 'center', 'gap': '8px'}, children=[
        #    html.Span("LAYER 1", style={
        #        'backgroundColor': '#6c757d', 'color': 'white', 'fontSize': '10px',
        #        'fontWeight': 'bold', 'padding': '2px 8px', 'borderRadius': '10px',
        #        'letterSpacing': '1px',
        #    }),
        #    html.Span("Controls", style={'fontSize': '12px', 'color': '#999', 'fontStyle': 'italic'}),
        #]),
    

        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '12px'}, children=[

            # Left Controls

            html.Div(
    style={
        'display': 'flex',
        'alignItems': 'center',
        'gap': '12px',
        'flexWrap': 'wrap'
    },
    children=[

        # Task
        html.Div(
            style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'},
            children=[
                html.Label(
                    "Task",
                    style={
                        'fontWeight': 'bold',
                        'fontSize': '12px',
                        'textTransform': 'uppercase',
                        'color': '#666',
                        'marginBottom': '0'
                    }
                ),
                dcc.RadioItems(
                    id='analysis-mode-switch',
                    options=[
                        {'label': ' Node', 'value': 'node_classification'},
                        {'label': ' Link', 'value': 'link_prediction'}
                    ],
                    value='node_classification',
                    inline=True,
                    labelStyle={'marginRight': '10px'}
                )
            ]
        ),

        # Separator
        html.Div(
            style={
                'width': '1px',
                'height': '30px',
                'backgroundColor': '#d0d0d0'
            }
        ),

        # Dataset
        html.Div(
            style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'},
            children=[
                html.Label(
                    "Dataset",
                    style={
                        'fontWeight': 'bold',
                        'fontSize': '12px',
                        'textTransform': 'uppercase',
                        'color': '#666',
                        'marginBottom': '0'
                    }
                ),
                dcc.Dropdown(
                    id='dataset-dropdown',
                    options=AVAILABLE_DATASETS,
                    value=DEFAULT_DATASET,
                    clearable=False,
                    style={'width': '120px'}
                )
            ]
        ),

        # Separator
        html.Div(
            style={
                'width': '1px',
                'height': '30px',
                'backgroundColor': '#d0d0d0'
            }
        ),

        # Model
        html.Div(
            style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'},
            children=[
                html.Label(
                    "Model",
                    style={
                        'fontWeight': 'bold',
                        'fontSize': '12px',
                        'textTransform': 'uppercase',
                        'color': '#666',
                        'marginBottom': '0'
                    }
                ),
                dcc.Dropdown(
                    id='model-dropdown',
                    options=AVAILABLE_MODELS,
                    value=DEFAULT_MODEL,
                    clearable=False,
                    style={'width': '100px'}
                )
            ]
        ),

        # Separator
        html.Div(
            style={
                'width': '1px',
                'height': '30px',
                'backgroundColor': '#d0d0d0'
            }
        ),

        # Selection
        html.Div(
            style={'display': 'flex', 'alignItems': 'center', 'gap': '8px'},
            children=[
                html.Label(
                    "Selection",
                    style={
                        'fontWeight': 'bold',
                        'fontSize': '12px',
                        'textTransform': 'uppercase',
                        'color': '#666',
                        'marginBottom': '0'
                    }
                ),
                html.Div(
                    id='selection-interface',
                    style={'minWidth': '150px'}
                )
            ]
        )
    ]
),

            # Compact performance metrics centered between left controls and right buttons
            html.Div(id='performance-metrics', style={'flex': '0 0 auto', 'alignSelf': 'center', 'display': 'none', 'gap': '6px', 'alignItems': 'center', 'fontSize': '18px', 'fontWeight': 'bold'}),

            # Right Control (Buttons),

            html.Div(style={'display': 'flex', 'gap': '10px'}, children=[

                html.Button(

                    "RESET", 

                    id='reset-button', 

                    n_clicks=0,

                    style={

                        'backgroundColor': '#6c757d',

                        'color': 'white',

                        'border': 'none',

                        'padding': '8px 14px',

                        'fontSize': '13px',

                        'fontWeight': 'bold',

                        'borderRadius': '4px',

                        'cursor': 'pointer',

                        'boxShadow': 'none'

                    }

                ),

                html.Button(

                    "RUN ANALYSIS", 

                    id='global-run-button', 

                    n_clicks=0,

                    style={

                        'backgroundColor': '#007bff',

                        'color': 'white',

                        'border': 'none',

                        'padding': '8px 14px',

                        'fontSize': '13px',

                        'fontWeight': 'bold',

                        'borderRadius': '4px',

                        'cursor': 'pointer',

                        'boxShadow': 'none'

                    }

                )

            ])

        ]),

        # dynamic-analysis-tabs removed — GNN/Causal/Stability handled entirely in Layer 3 tabs
        html.Div(id='dynamic-analysis-tabs', style={'display': 'none'}),

    ]),

    # --- LAYER 2: MID (GLOBAL VIEW ONLY) ---

    html.Div(id='layer-2', style={'display': 'none', 'borderTop': '0', 'border': 'none', 'paddingTop': '0', 'paddingBottom': '0', 'marginTop': '0', 'marginBottom': '0', 'height': '160px', 'maxHeight': '160px', 'overflow': 'hidden', 'boxSizing': 'border-box', 'cursor': 'pointer'}, children=[

        # ── ANALYTICS — full-width 3-column layout ──
        html.Div(style={
                'display': 'flex',
                'flexDirection': 'row',
                'justifyContent': 'space-between',
                'alignItems': 'stretch',
                'gap': '12px',
                'width': '100%',
                'maxWidth': 'none',
                'margin': '0',
                'paddingTop': '0',
                'paddingRight': '0',
                'paddingBottom': '0',
                'paddingLeft': '0',
                'boxSizing': 'border-box',
                'height': '100%',
                'alignItems': 'stretch'
            }, children=[

                # ── Panel 1: Node Embeddings ──
                html.Div(style={
                    'flex': '1.25',   # a little wider than the other two panels
                    'minWidth': '300px',
                    'minHeight': '0',
                    'height': '100%',
                    'overflow': 'hidden',
                    'backgroundColor': '#ffffff',
                    'border': '1px solid #d0d7e3',
                    'borderRadius': '8px',
                    'padding': '8px',
                    'margin': '0',
                    'boxShadow': '0 1px 4px rgba(0,0,0,0.07)',
                    'boxSizing': 'border-box',
                    'display': 'flex',
                    'flexDirection': 'column',
                }, children=[
                    html.Div(className='subgraph-toolbar', style={'marginBottom': '2px'}, children=[
                        html.Span("Colour by", className='control-label'),
                        dcc.RadioItems(
                            id='embedding-color-mode',
                            options=[
                                {'label': 'Actual', 'value': 'actual'},
                                {'label': 'Predicted', 'value': 'predicted'},
                                {'label': 'Correctness', 'value': 'correctness'},
                                {'label': 'Confidence', 'value': 'confidence'},
                            ],
                            value='actual',
                            inline=True,
                            className='segmented',
                        ),
                    ]),
                    dcc.Graph(
                        id='embeddings-view',
                        style={'flex': '1', 'minHeight': '0', 'width': '100%', 'cursor': 'pointer', 'margin': '0', 'padding': '0'},
                        config={'displayModeBar': False},
                    ),
                ]),

                # ── Panel 2: Confusion Matrix (clickable) ──
                html.Div(style={
                    'flex': '1',
                    'minWidth': '300px',
                    'minHeight': '0',
                    'height': '100%',
                    'overflow': 'auto',      # scroll when the panel is smaller than the matrix
                    'backgroundColor': '#ffffff',
                    'border': '1px solid #d0d7e3',
                    'borderRadius': '8px',
                    'padding': '8px',
                    'margin': '0',
                    'boxShadow': '0 1px 4px rgba(0,0,0,0.07)',
                    'boxSizing': 'border-box',
                }, children=[
                    # No dcc.Loading wrapper: its inner div has no height, so the
                    # graph fell back to Plotly's 450px default and was clipped
                    # in the shorter panel (the matrix computes instantly anyway).
                    dcc.Graph(
                        id='confusion-matrix-plot',
                        # Fills the panel; the minimum height (per view, see
                        # confusion_matrix_style) makes the panel scroll only when
                        # the matrix would otherwise be squeezed
                        style=confusion_matrix_style('node_classification'),
                        config={'displayModeBar': False, 'responsive': True},
                    ),
                ]),

                # ── Panel 3: Confusion Matrix Details ──
                html.Div(id='gnn-mid-layer-content', style={
                    'flex': '0.8',      # narrower: details list needs less room
                    'minWidth': '260px',
                    'minHeight': '0',
                    'height': '100%',
                    'maxHeight': '100%',
                    'margin': '0',
                    'overflow': 'hidden',
                    'boxSizing': 'border-box',
                    'display': 'flex',
                    'flexDirection': 'column',
                }, children=[
                    html.Div(style={
                        'backgroundColor': '#ffffff',
                        'border': '1px solid #d0d7e3',
                        'borderRadius': '8px',
                        'padding': '8px',
                        'height': '100%',
                        'maxHeight': '100%',
                        'minHeight': '0',
                        'margin': '0',
                        'boxSizing': 'border-box',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'overflow': 'hidden',
                        'boxShadow': '0 1px 4px rgba(0,0,0,0.07)',
                    }, children=[
                        html.Div(id='global-link-analysis-title', style={'display': 'none'}),
                           html.Div(id='edge-predictions-display',
                                 style={'flex': '1', 'display': 'flex', 'flexDirection': 'column', 'overflow': 'auto',
                                     'margin': '0', 'padding': '0', 'minHeight': '0'},
                                 children=[cell_details_placeholder()]),
                        html.Div(id='link-prediction-results-container',
                                 style={'margin': '0', 'padding': '0', 'flex': '0 0 auto',
                                        'maxHeight': 'none', 'overflow': 'hidden', 'boxSizing': 'border-box'}, children=[
                            html.Div(id='link-prediction-output-active',
                                     style={'minHeight': '0', 'maxHeight': '100%', 'overflow': 'hidden', 'margin': '0', 'padding': '0'}),
                            dcc.Graph(id='link-confidence-plot-active',
                                     style={'height': '0px', 'width': '100%', 'margin': '0', 'padding': '0', 'display': 'none'},
                                     config={'displayModeBar': False}),
                        ]),
                    ]),
                ]),

            ]),

        ]),

    ]),

    # --- LAYER 3: BOTTOM — THREE EXPLANATION FRAMES ---

    html.Div(id='layer-3', style={'display': 'none', 'borderTop': '2px solid #dee2e6', 'paddingTop': '0', 'marginTop': '0', 'minHeight': '0', 'overflow': 'visible'}, children=[
        html.Div(id='loading-layer-3', children=[

            dcc.Tabs(
                id='layer3-tabs',
                value='tab-gnn',
                style={'borderBottom': '2px solid #dee2e6', 'minHeight': '100%'},
                colors={'border': '#dee2e6', 'primary': '#007bff', 'background': '#f8f9fa'},
                children=[

                    # ══════════════════════════════════════════════════
                    # TAB A: EXPLANATION
                    # ══════════════════════════════════════════════════
                    dcc.Tab(
                        label='Explanation',
                        value='tab-gnn',
                        style={
                            'fontWeight': 'bold', 'fontSize': '12px',
                            'padding': '8px 18px', 'letterSpacing': '0.3px',
                            'borderTop': '3px solid transparent',
                            'color': '#495057',
                        },
                        selected_style={
                            'fontWeight': 'bold', 'fontSize': '12px',
                            'padding': '8px 18px', 'letterSpacing': '0.3px',
                            'borderTop': '3px solid #007bff',
                            'color': '#007bff',
                            'backgroundColor': '#fff',
                        },
                        children=[
                            html.Div(style={
                                'padding': '16px',
                                'backgroundColor': '#fff',
                                'border': '1px solid #dee2e6',
                                'borderTop': 'none',
                                'borderRadius': '0 0 6px 6px',
                            }, children=[

                                # placeholder for controls (expand button moved below graph)
                                html.Div(style={'height': '0px', 'marginBottom': '0px'}),

                                # ── A1, A2 & A3: Local Subgraph (50%) | Neighbor (25%) | Feature (25%) ──
                                html.Div(className='explain-row', style={
                                    'display': 'flex',
                                    'gap': '14px',
                                    'marginBottom': '14px',
                                }, children=[
                                    # Left: Local Subgraph (50%)
                                    html.Div(style={
                                        'flex': '2 1 0',          # 2 : 1 : 1 of the space left after gaps
                                        
                                        'minWidth': '0',
                                        'backgroundColor': '#f8f9ff',
                                        'border': '1px solid #dde',
                                        'borderRadius': '6px',
                                        'padding': '10px',
                                        'boxSizing': 'border-box',
                                        'position': 'relative'
                                    }, children=[
                                             html.H6("Local Subgraph",
                                                  style={'fontWeight': 'bold', 'fontSize': '12px',
                                                      'color': '#0056b3', 'margin': '0 0 6px 0',
                                                      'textTransform': 'uppercase', 'letterSpacing': '0.3px'}),
                                        html.Div(className='subgraph-toolbar', children=[
                                            html.Span("Colour by", className='control-label'),
                                            dcc.RadioItems(
                                                id='subgraph-color-mode',
                                                options=[
                                                    {'label': 'Actual class', 'value': 'actual'},
                                                    {'label': 'Predicted class', 'value': 'predicted'},
                                                    {'label': 'Correctness', 'value': 'correctness'},
                                                ],
                                                value='actual',
                                                inline=True,
                                                className='segmented',
                                            ),
                                            html.Span("Edge width = explainer importance", className='hint',
                                                      title="Green edges were selected by GNNExplainer (mask > 0.1); thicker = more important."),
                                        ]),
                                        dcc.Loading(type='dot', color='#2563eb', children=cyto.Cytoscape(
                                            id='graph-view',
                                            stylesheet=STYLESHEET,
                                            layout={'name': 'cose', 'animate': False, 'randomize': False, 'fit': True},
                                            style={
                                                'width': '100%',
                                                'height': '340px',
                                                'border': '1px solid #e0e4f0',
                                                'backgroundColor': '#f9f9f9',
                                                'borderRadius': '4px',
                                            }
                                        )),
                                        # Inline legend placed over the bottom-left of the subgraph panel
                                        html.Div(id='subgraph-legend-inline', style={
                                            'marginTop': '8px',
                                            'backgroundColor': '#ffffff',
                                            'padding': '8px 10px',
                                            'borderRadius': '6px',
                                            'border': '1px solid #e2e7f0',
                                            'fontSize': '12px',
                                        }),
                                    ]),

                                    # Middle: Neighbor Importance (25%)
                                    html.Div(style={
                                        'flex': '1 1 0',
                                        
                                        'minWidth': '200px',
                                        'backgroundColor': '#f8f9ff',
                                        'border': '1px solid #dde',
                                        'borderRadius': '6px',
                                        'padding': '10px',
                                        'boxSizing': 'border-box',
                                        'position': 'sticky',
                                        'top': '16px',
                                        'height': '340px',
                                        'overflow': 'hidden'
                                    }, children=[
                                        html.H6("Neighbor Importance",
                                            id='layer3-right-header',
                                            style={'fontWeight': 'bold', 'fontSize': '12px',
                                                   'color': '#0056b3', 'margin': '0 0 6px 0',
                                                   'textTransform': 'uppercase', 'letterSpacing': '0.3px'}),
                                        dcc.Loading(type='dot', color='#2563eb', children=dcc.Graph(
                                            id='neighbor-importance-view',
                                            style={
                                                'height': '320px',
                                                'width': '100%',
                                                'overflowY': 'auto',
                                                'overflowX': 'auto'
                                            },
                                            config={'displayModeBar': False},
                                        )),
                                    ]),

                                    # Right: Feature Importance (25%)
                                    html.Div(style={
                                        'flex': '1 1 0',
                                        
                                        'minWidth': '200px',
                                        'backgroundColor': '#f8f9ff',
                                        'border': '1px solid #dde',
                                        'borderRadius': '6px',
                                        'padding': '10px',
                                        'boxSizing': 'border-box',
                                        'position': 'sticky',
                                        'top': '16px',
                                        'height': '340px',
                                        'overflow': 'auto'
                                    }, children=[
                                        html.H6("Feature Importance",
                                                style={'fontWeight': 'bold', 'fontSize': '12px',
                                                       'color': '#0056b3', 'margin': '0 0 6px 0',
                                                       'textTransform': 'uppercase', 'letterSpacing': '0.3px'}),
                                        dcc.Loading(type='dot', color='#2563eb', children=dcc.Graph(
                                            id='feature-importance-view',
                                            style={'height': '300px', 'width': '100%'},
                                            config={'displayModeBar': False},
                                        )),
                                    ]),
                                ]),

                                # Full-width row; explanation_cards() sizes the CPA-IV card to the
                                # Local Subgraph column. Flex reserves each column's padding+border
                                # (22px x 3) before the 2:1:1 split, so that column is
                                # 22 + (W - 28 gaps - 66) / 2 = (W - 50) / 2.
                                html.Div(className='cpa-frame', style={
                                    'width': '100%',
                                    'boxSizing': 'border-box',
                                    'marginTop': '16px'
                                }, children=[
                                    html.Div(id='gnn-causal-explanation-view', style={'display': 'none'})
                                ]),

                                # ── A6: Subgraph Legend (moved inline into the Local Subgraph panel)

                            ]),
                        ],
                    ),

                    # ══════════════════════════════════════════════════
                    # TAB B: FULL GRAPH VIEW
                    # ══════════════════════════════════════════════════
                    dcc.Tab(
                        label='Full Graph View',
                        value='tab-full-graph',
                        style={
                            'fontWeight': 'bold', 'fontSize': '12px',
                            'padding': '8px 18px', 'letterSpacing': '0.3px',
                            'borderTop': '3px solid transparent',
                            'color': '#495057',
                        },
                        selected_style={
                            'fontWeight': 'bold', 'fontSize': '12px',
                            'padding': '8px 18px', 'letterSpacing': '0.3px',
                            'borderTop': '3px solid #28a745',
                            'color': '#28a745',
                            'backgroundColor': '#fff',
                        },
                        children=[
                            html.Div(style={
                                'padding': '0',
                                'backgroundColor': '#fff',
                                'border': '1px solid #dee2e6',
                                'borderTop': 'none',
                                'borderRadius': '0 0 6px 6px',
                                'display': 'flex',
                                'flexDirection': 'column',
                                'overflow': 'hidden'
                            }, children=[
                                html.Div(style={
                                    'padding': '12px 16px',
                                    'borderBottom': '1px solid #eceeef',
                                    'backgroundColor': '#fafbff',
                                    'display': 'flex',
                                    'alignItems': 'center',
                                    'gap': '16px',
                                    'flexWrap': 'wrap'
                                }, children=[
                                    html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'}, children=[
                                        html.Label("Node Color", style={'fontWeight': 'bold', 'fontSize': '12px', 'color': '#333', 'marginBottom': '0'}),
                                        dcc.Dropdown(
                                            id='full-graph-color-mode',
                                            options=[
                                                {'label': 'Predicted class', 'value': 'predicted'},
                                                {'label': 'Actual class', 'value': 'actual'},
                                                {'label': 'Correctness', 'value': 'correctness'},
                                            ],
                                            value='predicted',
                                            clearable=False,
                                            style={'width': '220px', 'fontSize': '13px'}
                                        )
                                    ]),
                                    html.Div(style={'display': 'flex', 'alignItems': 'center', 'gap': '10px'}, children=[
                                        html.Label("Filter by", style={'fontWeight': 'bold', 'fontSize': '12px', 'color': '#333', 'marginBottom': '0'}),
                                        dcc.Dropdown(
                                            id='full-graph-filter-metric',
                                            options=[
                                                {'label': 'Confidence', 'value': 'confidence'},
                                                {'label': 'Stability', 'value': 'stability'},
                                            ],
                                            value='confidence',
                                            clearable=False,
                                            style={'width': '180px', 'fontSize': '13px'}
                                        )
                                    ]),
                                    html.Div(style={'minWidth': '280px', 'maxWidth': '380px'}, children=[
                                        html.Label("Filter range", style={'fontWeight': 'bold', 'fontSize': '12px', 'color': '#333', 'marginBottom': '6px', 'display': 'block'}),
                                        dcc.RangeSlider(
                                            id='full-graph-filter-range',
                                            min=0, max=1, step=0.01, value=[0, 1],
                                            marks={0: '0', 0.5: '0.5', 1: '1'},
                                            allowCross=False,
                                            tooltip={'placement': 'bottom', 'always_visible': False}
                                        )
                                    ]),
                                ]),
                                dcc.Loading(type='dot', color='#2563eb', children=dcc.Graph(
                                    id='full-graph-view',
                                    style={'width': '100%', 'height': '760px', 'minHeight': '760px'},
                                    config={
                                        'displayModeBar': True,
                                        'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                                        'scrollZoom': True,
                                        'toImageButtonOptions': {
                                            'format': 'png',
                                            'filename': 'full_graph',
                                            'scale': 2,
                                        },
                                    },
                                    figure=go.Figure(layout={
                                        'title': '',
                                        'paper_bgcolor': 'rgba(0,0,0,0)',
                                        'plot_bgcolor': '#f9fbff',
                                    })
                                ))
                            ]),
                        ],
                    ),

                    # ══════════════════════════════════════════════════
                    # TAB C: STABILITY ANALYSIS
                    # ══════════════════════════════════════════════════
                    dcc.Tab(
                        label='Stability Analysis',
                        value='tab-stability',
                        style={
                            'fontWeight': 'bold', 'fontSize': '12px',
                            'padding': '8px 18px', 'letterSpacing': '0.3px',
                            'borderTop': '3px solid transparent',
                            'color': '#495057',
                        },
                        selected_style={
                            'fontWeight': 'bold', 'fontSize': '12px',
                            'padding': '8px 18px', 'letterSpacing': '0.3px',
                            'borderTop': '3px solid #17a2b8',
                            'color': '#17a2b8',
                            'backgroundColor': '#fff',
                        },
                        children=[
                            html.Div(style={
                                'padding': '16px',
                                'backgroundColor': '#fff',
                                'border': '1px solid #dee2e6',
                                'borderTop': 'none',
                                'borderRadius': '0 0 6px 6px',
                            }, children=[

                                html.Div(id='stability-section', children=[

                                    # ── Mode toggle: Node / Link ──────────────────────────────────
                                    # Node Stability button removed; show node stability controls directly

                                    # ── NODE stability section ────────────────────────────────────
                                    html.Div(id='node-stability-section', children=[

                                    # ── Parameter controls ──
                                    html.Div(style={
                                        'display': 'flex', 'gap': '14px',
                                        'alignItems': 'flex-end', 'flexWrap': 'wrap',
                                        'marginBottom': '14px',
                                    }, children=[

                                        #html.Div([
                                            html.Label("Noise σ ⓘ",
                                                       title="Standard deviation of the Gaussian noise added to every node's features before re-running GNNExplainer.",
                                                       style={'fontWeight': 'bold', 'fontSize': '12px',
                                                              'color': '#555', 'display': 'block',
                                                              'marginBottom': '4px', 'cursor': 'help'}),
                                            dcc.RadioItems(
                                                id='stability-sigma-selector',
                                                options=[
                                                    {'label': ' 0.01', 'value': 0.01},
                                                    {'label': ' 0.05', 'value': 0.05},
                                                    {'label': ' 0.10', 'value': 0.1},
                                                ],
                                                value=0.05,
                                                inline=True,
                                                inputStyle={'marginRight': '4px'},
                                                style={'fontSize': '13px', 'color': '#333'},
                                            ),
                                        #]),

                                        html.Div(
                                            style={
                                                'width': '1px',
                                                'height': '24px',
                                                'backgroundColor': '#ddd'
                                                }
                                            ),
                                         # Visualization type dropdowns
                                            html.Div(style={
                                                'display': 'flex', 'alignItems': 'center', 'gap': '10px', 'flexWrap': 'wrap'
                                            }, children=[
                                                html.Span("Visualization Type:", style={'fontSize': '13px', 'color': '#555', 'fontWeight': 'bold'}),
                                                dcc.Dropdown(
                                                    id='stability-plot-type',
                                                    options=[
                                                        {'label': 'Scatter Plot', 'value': 'scatter'},
                                                        {'label': 'Hexbin',       'value': 'hexbin'},
                                                        {'label': 'KDE',          'value': 'kde'},
                                                    ],
                                                    value='scatter',
                                                    clearable=False,
                                                    style={'width': '120px', 'fontSize': '13px'}
                                                ),

                                                html.Div(
                                                    style={
                                                        'width': '1px',
                                                        'height': '24px',
                                                        'backgroundColor': '#ddd'
                                                    }
                                                ),

                                                html.Span("Axis Metrics ⓘ:",
                                                          title="Confidence: predicted-class probability. Degree: number of neighbours. "
                                                                "Lipschitz: change in the edge mask per unit of input noise (lower = smoother). "
                                                                "Correct vs Wrong: stability split by whether the prediction is right.",
                                                          style={'fontSize': '13px', 'color': '#555', 'fontWeight': 'bold', 'marginLeft': '10px', 'cursor': 'help'}),
                                                dcc.Dropdown(
                                                    id='stability-metric-selector',
                                                    options=[
                                                        {'label': 'Confidence vs Stability', 'value': 'confidence'},
                                                        {'label': 'Degree vs Stability', 'value': 'degree'},
                                                        {'label': 'Lipschitz vs Stability', 'value': 'lipschitz'},
                                                        {'label': 'Correct vs Wrong Stability', 'value': 'fidelity'},
                                                    ],
                                                    value='confidence',
                                                    clearable=False,
                                                    style={'width': '220px', 'fontSize': '13px'}
                                                ),
                                            ]),

                                             html.Div(
                                                    style={
                                                        'width': '1px',
                                                        'height': '24px',
                                                        'backgroundColor': '#ddd'
                                                    }
                                                ),


                                        html.Div([
                                            #html.Label("Nodes to Analyze  (0 = Full Graph)",
                                            #           style={'fontWeight': 'bold', 'fontSize': '12px',
                                            #                  'color': '#555', 'display': 'block',
                                            #                  'marginBottom': '4px'}),
                                            dcc.Input(
                                                id='stability-sample-size', type="hidden",
                                                #type='number', value=0,
                                                min=0, max=5000, step=50, placeholder='0 = all',
                                                style={'width': '90px', 'padding': '5px 8px',
                                                       'border': '1px solid #ccc', 'borderRadius': '4px',
                                                       'fontSize': '13px'},
                                            ),
                                            #html.Div("Default: full graph",
                                            #         style={'fontSize': '10px', 'color': '#17a2b8',
                                            #                'marginTop': '3px', 'fontStyle': 'italic'}),
                                        ]),

                                        # Hidden inputs — keep callback signatures intact
                                        dcc.Input(id='stability-method-selector',
                                                  type='hidden', value='gnn_explainer'),
                                        dcc.Input(id='stability-trials',
                                                  type='hidden', value=20),

                                        html.Button(
                                            "Run Stability Analysis",
                                            id='run-stability-button', n_clicks=0,
                                            style={
                                                'backgroundColor': '#17a2b8', 'color': 'white',
                                                'border': 'none', 'padding': '8px 20px',
                                                'borderRadius': '4px', 'cursor': 'pointer',
                                                'fontWeight': 'bold', 'fontSize': '12px',
                                                'whiteSpace': 'nowrap',
                                            },
                                        ),

                                        html.Div(id='stability-status',
                                                 style={'fontSize': '12px', 'color': '#666',
                                                        'alignSelf': 'center', 'fontStyle': 'italic'},
                                                 children=" "),
                                    ]),

                                    html.Div(className='help-line', children=[
                                        html.B("Stability S"), " = Jaccard overlap of the top-10 explanation edges before and after adding noise σ "
                                        "(1 = identical explanation, 0 = no overlap). ",
                                        html.B("Fidelity"), " = drop in confidence when those 10 edges are removed. "
                                        "Click a region of the plot to list the nodes in it.",
                                    ]),

                                    # ── Split-Screen Layout ───────────────────────────────────────
                                    html.Div(style={
                                        'display': 'flex', 'gap': '20px',
                                        'alignItems': 'flex-start',
                                        'marginTop': '10px'
                                    }, children=[
                                        # --- LEFT COLUMN (50%) ---
                                        html.Div(style={'flex': '1', 'minWidth': '0', 'display': 'flex', 'flexDirection': 'column', 'gap': '12px'}, children=[
                                           
                                            # Main Plot Container
                                            html.Div(style={'border': '1px solid #dee2e6', 'borderRadius': '4px', 'padding': '8px', 'backgroundColor': '#fafafa'}, children=[
                                                dcc.Loading(type='dot', color='#2563eb', children=dcc.Graph(
                                                    id='main-stability-plot', 
                                                    figure=go.Figure(layout={'title': 'Select a metric to view', 'height': 360, 'margin': dict(l=40, r=15, t=40, b=40), 'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)'}),
                                                    style={'height': '360px'}, 
                                                    config={'displayModeBar': False}
                                                ))
                                            ]),

                                            # Stats Panel
                                            html.Div(
                                                id='stability-stats-panel',
                                                style={'border': '1px solid #ccc', 'padding': '10px', 'backgroundColor': '#fff', 'borderRadius': '4px', 'fontSize': '12px'},
                                                children=html.Div("Run analysis to see statistics.", style={'color': '#999', 'textAlign': 'center', 'paddingTop': '10px'})
                                            ),
                                        ]),

                                        # --- RIGHT COLUMN (50%) ---
                                        html.Div(id='drilldown-right-panel', style={'flex': '1', 'minWidth': '0', 'display': 'none', 'flexDirection': 'column', 'gap': '12px'}, children=[
                                            # Drill-down controls
                                            html.Div(style={
                                                'display': 'flex', 'alignItems': 'center', 'gap': '10px',
                                                'padding': '8px 12px', 'backgroundColor': '#f0f7ff', 'borderRadius': '6px',
                                                'border': '1px solid #b8d4f0', 'flexWrap': 'wrap'
                                            }, children=[
                                                html.Span("Region Size (ε):", style={'fontSize': '12px', 'fontWeight': 'bold', 'color': '#1a5276', 'whiteSpace': 'nowrap'}),
                                                html.Div(style={'flex': '1', 'minWidth': '120px'}, children=[
                                                    dcc.Slider(
                                                        id='drilldown-epsilon-slider',
                                                        min=0.01, max=0.30, step=0.01, value=0.08,
                                                        marks={v: f'{v:.2f}' for v in [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]},
                                                        tooltip={'placement': 'bottom', 'always_visible': False},
                                                    ),
                                                ]),
                                                html.Button(
                                                    "Reset Selection", id='drilldown-reset-btn', n_clicks=0,
                                                    style={'fontSize': '12px', 'padding': '4px 12px', 'backgroundColor': '#dc3545', 'color': '#fff', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer', 'whiteSpace': 'nowrap'}
                                                ),
                                            ]),
                                            html.Div(
                                                id='drilldown-hint', 
                                                style={'fontSize': '11px', 'color': '#666', 'fontStyle': 'italic', 'textAlign': 'center'}, 
                                                children="Click any region on the left plot to drill down"
                                            ),

                                            # Drill-down scatter panel
                                            html.Div(style={'border': '1px solid #dee2e6', 'borderRadius': '6px', 'backgroundColor': '#fafafa', 'padding': '8px'}, children=[
                                                html.Div(
                                                    id='drilldown-panel-header',
                                                    children="Click on a region to explore node-level details",
                                                    style={'fontSize': '12px', 'fontWeight': 'bold', 'color': '#555', 'marginBottom': '6px', 'padding': '4px 8px', 'backgroundColor': '#e9ecef', 'borderRadius': '4px'}
                                                ),
                                                dcc.Graph(
                                                    id='drilldown-scatter',
                                                    figure=go.Figure(layout=dict(
                                                        height=360, margin=dict(l=45, r=15, t=40, b=40), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                                        annotations=[dict(xref='paper', yref='paper', x=0.5, y=0.5, text='Click on a plot region to see details here', showarrow=False, font=dict(size=13, color='#aaa'), xanchor='center', yanchor='middle')]
                                                    )),
                                                    style={'height': '360px'},
                                                    config={'displayModeBar': True, 'modeBarButtonsToRemove': ['lasso2d', 'select2d']},
                                                ),
                                            ]),
                                        ]),
                                    ]),

                                    ]),  # ← close node-stability-section

                                    # Link stability section removed in this version

                                ]),

                            ]),
                        ],
                    ),

                ],
            ),

        ])

    ]),

    # Legacy sections (HIDDEN),

    html.Div(id='node-classification-section', style={'display': 'none'}),

]) # Close main layout

# --- Callbacks ---

@callback(
    Output('full-graph-view', 'figure'),
    Input('layer3-tabs', 'value'),
    Input('full-graph-color-mode', 'value'),
    Input('full-graph-filter-metric', 'value'),
    Input('full-graph-filter-range', 'value'),
    Input('selected-node-store', 'data'),
    Input('current-predictions-store', 'data'),
    State('full-dataset-store', 'data'),
    State('editable-graph-store', 'data'),
    State('stability-results-store', 'data'),
)
def update_full_graph_view(active_tab, color_mode, filter_metric, filter_range, selected_node, predictions_data, dataset, graph_data, stability_results):
    return build_full_graph_figure(active_tab, color_mode, filter_metric, filter_range, selected_node, predictions_data, dataset, graph_data, stability_results)

@callback(
    Output('edge-predictions-display', 'children', allow_duplicate=True),
    Output('cm-detail-paging-store', 'data', allow_duplicate=True),
    Input('analysis-mode-store', 'data'),
    Input('dataset-dropdown', 'value'),
    Input('model-dropdown', 'value'),
    prevent_initial_call=True,
)
def reset_cell_details(analysis_mode, _dataset, _model):
    """A clicked cell belongs to one matrix; clear it when the matrix changes."""
    return cell_details_placeholder(analysis_mode), None


@callback(
    Output('confusion-matrix-plot', 'style'),
    Input('analysis-mode-store', 'data'),
)
def update_confusion_matrix_style(analysis_mode):
    return confusion_matrix_style(analysis_mode)


@callback(
    Output('stability-results-store', 'data', allow_duplicate=True),
    Output('stability-status', 'children', allow_duplicate=True),
    Input('dataset-dropdown', 'value'),
    Input('model-dropdown', 'value'),
    prevent_initial_call=True,
)
def reset_stability_on_model_change(_dataset, model_type):
    """Stability results belong to one dataset/model: clear them on a switch so
    another model's plots are never shown under the new selection."""
    if not model_capabilities(model_type)['stability']:
        return [], stability_unsupported_message(model_type)
    return [], " "


@callback(
    Output('status-bar', 'children'),
    Input('dataset-dropdown', 'value'),
    Input('model-dropdown', 'value'),
    Input('analysis-mode-store', 'data'),
    Input('current-predictions-store', 'data'),
    Input('selected-node-store', 'data'),
    Input('selected-link-store', 'data'),
    Input('explanation-store', 'data'),
    Input('link-status-store', 'data'),
)
def update_status_bar(dataset_name, model_type, analysis_mode, predictions_data, selected_node, selected_link, explanation, link_status):
    """Breadcrumb of the current context plus the next step to take."""
    task = 'Link prediction' if analysis_mode == 'link_prediction' else 'Node classification'
    chips = [html.Span(f"{dataset_name} · {model_type} · {task}", className='chip chip-context')]

    if not predictions_data:
        chips.append(html.Span("Loading model…", className='chip chip-busy'))
        return chips

    if analysis_mode == 'link_prediction':
        if isinstance(link_status, dict) and link_status.get('type') == 'error':
            chips.append(html.Span(link_status.get('text', 'Error'), className='chip chip-error'))
        if selected_link:
            chips.append(html.Span(f"Link {selected_link.get('source')} → {selected_link.get('target')}", className='chip'))
            step = "Inspect the causal paths and subgraph below."
        else:
            step = ("Enter source and target node IDs and press RUN ANALYSIS, or click a "
                    "confusion-matrix cell and then a link in its list.")
    else:
        node_id = (selected_node or {}).get('id')
        explained = (isinstance(explanation, dict) and explanation.get('explanation_type') == 'node_classification'
                     and not explanation.get('error')
                     and str(explanation.get('node_idx')) == str(node_id))
        if explanation and isinstance(explanation, dict) and explanation.get('error'):
            chips.append(html.Span(explanation['error'], className='chip chip-error'))
        if node_id is None:
            step = "Click a node in the embedding plot or full graph, or type an ID under Selection."
        else:
            chips.append(html.Span(f"Node {node_id}", className='chip'))
            step = ("Explanation ready: see the Explanation tab." if explained
                    else "Press RUN ANALYSIS to explain this node.")
    chips.append(html.Span(step, className='next-step'))
    return chips


@callback(
    Output('analysis-mode-store', 'data'),
    Output('selection-interface', 'children'),
    Output('dynamic-analysis-tabs', 'children'),
    Output('graph-type-store', 'data'),
    Input('analysis-mode-switch', 'value'),
)
def update_analysis_mode(mode):
    """Layer 1 only controls task mode. All explanation logic lives in Layer 3 tabs."""
    graph_type = 'full' if mode == 'node_classification' else 'sub'

    if mode == 'link_prediction':
        # Source + Target for link prediction (used by LP inference + Causal analysis)
        selection_interface = html.Div(style={'padding': '5px 0', 'width': '100%'}, children=[
            html.Div(style={'display': 'flex', 'gap': '10px', 'marginBottom': '5px'}, children=[
                html.Div(style={'flex': '1'}, children=[
                    dcc.Input(
                        id={'type': 'lp-input', 'index': 'source-active'},
                        type='number', placeholder='Source', min=0, step=1,
                        style={'width': '100px'}
                    )
                ]),
                html.Div(style={'flex': '1'}, children=[
                    dcc.Input(
                        id={'type': 'lp-input', 'index': 'target-active'},
                        type='number', placeholder='Target', min=0, step=1,
                        style={'width': '130px'}
                    )
                ])
            ])
        ])

    else:
        # Node classification — node ID input
        selection_interface = dcc.Input(
            id='select-node-input-nc-active',
            type='number', placeholder='Enter Node ID...',
            min=0, step=1, style={'width': '100%', 'marginTop': '5px'}
        )

    # Keep DOM components that callbacks depend on (hidden — they were inside the old tabs)
    kept_components = html.Div(style={'display': 'none'}, children=[
        dcc.Input(id='edge-source-input', type='number', value=None),
        dcc.Input(id='edge-target-input', type='number', value=None),
        html.Button(id='add-edge-button-nc', n_clicks=0),
        html.Button(id='remove-edge-button-nc', n_clicks=0),
        html.Div(id='subgraph-eval-status'),
        dcc.Input(id={'type': 'lp-input', 'index': 'pred-node'}, type='number', value=None),
    ])

    return mode, selection_interface, kept_components, graph_type

@callback(

    Output('node-classification-section', 'style'),

    Input('analysis-mode-switch', 'value')

)

def toggle_node_classification_section(mode):

    """Show/hide node classification section based on analysis mode."""

    if mode == 'link_prediction':

        # Hide the section in link prediction mode

        return {'display': 'none'}

    else:

        # Show the section in node classification mode

        return {'margin': 0, 'marginTop': '20px'}

# Removed problematic default loading callback - using dropdown default values instead

@callback(

    Output('precomputed-package-store', 'data'), Output('full-dataset-store', 'data'),

    Output('editable-graph-store', 'data'), Output('current-predictions-store', 'data'),

    Input('dataset-dropdown', 'value'), Input('model-dropdown', 'value'),

    prevent_initial_call=False

)
@serialize_model_use
def load_initial_data(dataset_name, model_type):

    try:

        print(f"\n{'='*60}")

        print(f"LOADING: {model_type} model for {dataset_name} dataset...")

        print(f"{ '='*60}")

        # Load package metadata into server-side cache only

        package_meta = package_meta_from_selection(model_type, dataset_name)
        package = get_cached_package_if_missing(package_meta)
        if package is None:
            error_msg = f"Failed to load {model_type} package for {dataset_name}"
            print(f"ERROR: {error_msg}")
            return None, None, None, None

        print(f"Package loaded into server cache. Keys: {list(package.keys())}")

        # Load dataset metadata into server-side cache only

        dataset_meta = {'name': dataset_name}
        dataset = get_cached_dataset_if_missing(dataset_meta)
        if dataset is None:
            error_msg = f"Failed to load dataset {dataset_name}"
            print(f"ERROR: {error_msg}")
            return None, None, None, None

        print(f"Dataset loaded into server cache: {dataset['num_nodes']} nodes, {dataset['num_features']} features, {dataset['num_classes']} classes")

        # Prepare lightweight metadata for browser-side stores

        true_labels = None
        if 'y' in dataset:
            try:
                true_labels = dataset['y'].cpu().tolist() if hasattr(dataset['y'], 'cpu') else list(dataset['y'])
            except Exception:
                try:
                    true_labels = list(dataset['y'])
                except Exception:
                    true_labels = None

        full_dataset_meta = {
            'name': dataset_name,
            'num_nodes': dataset['num_nodes'],
            'num_features': dataset['num_features'],
            'num_classes': dataset['num_classes'],
            'original_edge_index': dataset['original_edge_index'].cpu().tolist(),
            'y': true_labels,
        }
        editable_graph = {'edge_index': dataset['original_edge_index'].cpu().tolist()}

        # Get predictions by live inference on the full graph, so the initial view
        # matches what later re-predictions (after graph edits) compute. The saved
        # package predictions can come from the training graph (GCN cached=True)
        # and carry no confidences.

        initial_predictions = None
        model = get_cached_model(package_meta, dataset['num_features'], dataset['num_classes'])
        if model is not None:
            pyg_data = prepare_torch_data(dataset_meta, None, include_labels=False)
            if pyg_data is not None:
                initial_predictions = run_inference(model, pyg_data)
        if initial_predictions is None:
            print("WARNING: Live inference failed; falling back to saved package predictions")
            initial_predictions = {'preds': package.get('predictions', []),
                                   'confidences': package.get('confidences', [])}
        predictions = initial_predictions['preds']
        if not predictions:
            print("WARNING: No predictions available")

        print(f"SUCCESS: {model_type} model loaded successfully for {dataset_name}")

        print(f"  - Model args: {package.get('model_init_args', {})}")

        print(f"  - Predictions: {len(predictions)} nodes")

        print(f"  - Embeddings: {{'embeddings_2d' in package}}")

        print(f"{ '='*60}\n")

        print(f"✅ {model_type} model loaded successfully for {dataset_name}")

        return package_meta, full_dataset_meta, editable_graph, initial_predictions

    except Exception as e:

        error_msg = f"Failed to load {model_type} model: {str(e)}"

        print(f"\nERROR in load_initial_data: {error_msg}")

        import traceback

        traceback.print_exc()

        print(f"{ '='*60}\n")

        return None, None, None, None

@callback(

    Output('editable-graph-store', 'data', allow_duplicate=True),

    Output('edge-status-store', 'data'),

    Input('add-edge-button-nc', 'n_clicks'), Input('remove-edge-button-nc', 'n_clicks'),

    Input('add-edge-button-lp', 'n_clicks'), Input('remove-edge-button-lp', 'n_clicks'),

    State('edge-source-input', 'value'), State('edge-target-input', 'value'),

    State('editable-graph-store', 'data'), State('full-dataset-store', 'data'),

    prevent_initial_call=True

)

def update_edges(add_clicks, remove_clicks, add_clicks_lp, remove_clicks_lp, source, target, graph_data, dataset):

    if not all([source is not None, target is not None, graph_data, dataset]): raise dash.exceptions.PreventUpdate

    if not (0 <= source < dataset['num_nodes'] and 0 <= target < dataset['num_nodes']):

        return no_update, {'text': 'Node ID out of range.', 'type': 'error'}

    # Copy so the State object is never mutated in place
    edge_index = [list(graph_data['edge_index'][0]), list(graph_data['edge_index'][1])]

    triggered_id = ctx.triggered_id

    status = no_update

    if triggered_id in ['add-edge-button-nc', 'add-edge-button-lp']:

        if (source, target) not in set(zip(edge_index[0], edge_index[1])):

             edge_index[0].extend([source, target])

             edge_index[1].extend([target, source])

             status = {'text': f'Added edge {source}-{target}. Re-running model...', 'type': 'info'}

        else:

             return no_update, {'text': f'Edge {source}-{target} already exists.', 'type': 'info'}

    elif triggered_id in ['remove-edge-button-nc', 'remove-edge-button-lp']:

        new_edge_index = [[], []]

        removed = False

        for u, v in zip(edge_index[0], edge_index[1]):

            if not ((u == source and v == target) or (u == target and v == source)):

                new_edge_index[0].append(u)

                new_edge_index[1].append(v)

            else:

                removed = True

        if not removed:

            return no_update, {'text': f'Edge {source}-{target} does not exist.', 'type': 'info'}

        edge_index = new_edge_index

        status = {'text': f'Removed edge {source}-{target}. Re-running model...', 'type': 'info'}

    return {'edge_index': edge_index}, status

@callback(

    Output('current-predictions-store', 'data', allow_duplicate=True),

    Input('editable-graph-store', 'data'),

    State('full-dataset-store', 'data'),

    State('precomputed-package-store', 'data'),

    State('model-dropdown', 'value'),

    prevent_initial_call=True

)
@serialize_model_use
def update_predictions_on_edit(graph_data, dataset_meta, package_meta, model_type):

    try:

        if not all([graph_data, dataset_meta, package_meta]):
            raise dash.exceptions.PreventUpdate

        dataset = get_cached_dataset_if_missing(dataset_meta)
        package = get_cached_package_if_missing(package_meta)
        if not dataset or not package:
            raise dash.exceptions.PreventUpdate

        model = get_cached_model(package_meta, dataset_meta['num_features'], dataset_meta['num_classes'])
        if model is None:
            raise dash.exceptions.PreventUpdate

        pyg_data = prepare_torch_data(dataset_meta, graph_data, include_labels=False)
        res = run_inference(model, pyg_data)
        # res is a dict {'preds': [...], 'confidences': [...]}
        return res

    except Exception as e:

        print(f"ERROR in update_predictions_on_edit: {e}")
        raise dash.exceptions.PreventUpdate

@callback(
    Output('selected-node-store', 'data'),
    Input('graph-view', 'tapNodeData'),
    Input('embeddings-view', 'clickData'),
    Input('full-graph-view', 'clickData'),
    prevent_initial_call=True,
)
def update_selected_node(tapNode, emb_click, full_click):
    trigger_id, node_id = ctx.triggered_id, None

    if trigger_id == 'graph-view' and tapNode:
        node_id = tapNode['id']
    elif trigger_id in ('embeddings-view', 'full-graph-view'):
        click = emb_click if trigger_id == 'embeddings-view' else full_click
        cd = ((click or {}).get('points') or [{}])[0].get('customdata')
        if isinstance(cd, (list, tuple)):
            cd = cd[0] if cd else None
        if cd is not None:
            node_id = cd

    if node_id is not None:
        print(f"[SELECTED_NODE] update_selected_node triggered by {trigger_id}, node_id={node_id}")
        # Store ids as strings everywhere (Cytoscape ids are strings)
        return {'id': str(node_id)}
    else:
        print(f"[SELECTED_NODE] update_selected_node triggered by {trigger_id}, no node selected")
        return no_update

# Separate callback for node input that only triggers when the component exists

@callback(

    Output('selected-node-store', 'data', allow_duplicate=True),

    Input('select-node-input-nc', 'value'),

    prevent_initial_call=True

)

def update_selected_node_from_input(input_value):

    if input_value is not None:
        print(f"[SELECTED_NODE] update_selected_node_from_input value={input_value}")
        return {'id': str(input_value)}

    return no_update

# Callback for active node input (when in node classification mode)

@callback(

    Output('selected-node-store', 'data', allow_duplicate=True),

    Input('select-node-input-nc-active', 'value'),

    prevent_initial_call=True

)

def update_selected_node_from_input_active(input_value):

    if input_value is not None:
        print(f"[SELECTED_NODE] update_selected_node_from_input_active value={input_value}")
        return {'id': str(input_value)}

    return no_update

@callback(
    Output('selected-node-store', 'data', allow_duplicate=True),
    Input('global-run-button', 'n_clicks'),
    prevent_initial_call=True
)
def clear_node_selection_on_run(run_clicks):
    """Clear node selection when RUN is clicked so Layer 3 stays hidden until a node is picked."""
    # Preserve the selected node on Run to avoid hiding CPA/Layer3 unexpectedly.
    # Previous behavior cleared selection unconditionally; this caused CPA views
    # to disappear when the user clicked Run. Keep selection intact.
    return no_update

@callback(

    Output('selected-link-store', 'data'),

    # Triggers

    Input('global-run-button', 'n_clicks'),

    Input({'type': 'link-prediction-item', 'index': ALL, 'source': ALL, 'target': ALL}, 'n_clicks'),

    # States for manual inputs (kept for signature compatibility but ignored for selection)

    State({'type': 'lp-input', 'index': ALL}, 'value'),

    State({'type': 'lp-input', 'index': ALL}, 'id'),

    State('analysis-mode-store', 'data'),

    State('layer3-tabs', 'value'),

    State('full-dataset-store', 'data'),

    prevent_initial_call=True

)

def update_selected_link_master(run_clicks, item_clicks, lp_input_values, lp_input_ids, analysis_mode, active_layer3_tab, dataset_meta):

    """

    STRICT INTERACTION FIX:

    - Updates selected-link-store ONLY when a link prediction item is clicked.

    - CLEARS selection (returns None) when RUN ANALYSIS is clicked.

    - IGNORES manual inputs for opening Layer 3.

    """

    if not ctx.triggered:

        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered_id

    # Priority 1: A specific link was clicked from the prediction list

    # This is the ONLY allowed trigger for opening Layer 3

    if isinstance(triggered_id, dict) and triggered_id.get('type') == 'link-prediction-item':

        if not any(item_clicks): raise dash.exceptions.PreventUpdate

        source = triggered_id.get('source')

        target = triggered_id.get('target')

        if source is not None and target is not None:

            print(f"Selected link from prediction list: ({source}, {target})")

            return {'source': int(source), 'target': int(target)}

    # Priority 2: "RUN ANALYSIS" clicked
    # Preserve current selection on Run to avoid race conditions where
    # the selection is cleared before downstream callbacks read it.
    elif triggered_id == 'global-run-button':
        # Link mode: a typed source + target pair becomes the selected link, which
        # opens its subgraph, explanation and CPA-IV (invalid pairs are reported
        # by run_link_prediction_unified via link-status-store).
        if analysis_mode == 'link_prediction' and lp_input_values and lp_input_ids:
            vals = {i.get('index'): v for v, i in zip(lp_input_values, lp_input_ids)}
            try:
                u, v = int(vals.get('source-active')), int(vals.get('target-active'))
            except (TypeError, ValueError):
                return no_update
            n = (dataset_meta or {}).get('num_nodes') or 0
            if u != v and 0 <= u < n and 0 <= v < n:
                print(f"RUN ANALYSIS with typed link ({u}, {v}) - selecting it.")
                return {'source': u, 'target': v}
        print("RUN ANALYSIS clicked - preserving selected link (no clear).")
        return no_update

    raise dash.exceptions.PreventUpdate

@callback(
    Output('selected-path-store', 'data'),
    Input({'type': 'cpa-path-item', 'index': ALL}, 'n_clicks'),
    Input({'type': 'cpa-path-item-side', 'index': ALL}, 'n_clicks'),
    Input('global-run-button', 'n_clicks'),
    prevent_initial_call=True
)
def update_selected_path(main_clicks, side_clicks, run_clicks):
    trigger = ctx.triggered_id
    
    # Reset on Run
    if trigger == 'global-run-button':
        return None
        
    # Handle clicks
    if isinstance(trigger, dict):
        if 'index' in trigger:
            print(f"Path selected: {trigger['index']}")
            return {'index': trigger['index']}
            
    return no_update

@callback(

    Output('explanation-store', 'data', allow_duplicate=True),

    Output('link-prediction-store', 'data', allow_duplicate=True),

    Input('selected-link-store', 'data'),

    State('global-run-button', 'n_clicks'),

    State('editable-graph-store', 'data'),

    State('precomputed-package-store', 'data'), 

    State('full-dataset-store', 'data'),

    State('model-dropdown', 'value'),

    State('analysis-mode-store', 'data'),

    prevent_initial_call=True

)
@serialize_model_use
def generate_link_explanation_on_select(selected_link, run_clicks, graph_data, package_meta, dataset_meta, model_type, analysis_mode):

    """Generates a link explanation automatically when a link is selected."""

    # DEBUG: log select-triggered generation attempts
    try:
        with open('link_debug.log', 'a') as _log:
            _log.write(f"GEN_LINK_SELECT run_clicks={run_clicks} selected_link_present={bool(selected_link)} analysis_mode={analysis_mode}\n")
    except Exception:
        pass

    if not run_clicks or run_clicks == 0 or not selected_link or analysis_mode != 'link_prediction':

        return no_update, no_update

    try:

        dataset = get_cached_dataset_if_missing(dataset_meta)
        package = get_cached_package_if_missing(package_meta)
        if not all([package, dataset, graph_data]):
            return no_update, no_update

        model = get_cached_model(package_meta, dataset_meta['num_features'], dataset_meta['num_classes'])
        if model is None or not getattr(model, 'enable_link_prediction', False):
            return no_update, no_update

        source_node = selected_link.get('source')

        target_node = selected_link.get('target')

        if source_node is not None and target_node is not None:

            pyg_data = prepare_torch_data(dataset_meta, graph_data, include_labels=False)

            model.eval()

            link_explanation = explain_link_prediction(model, pyg_data, source_node, target_node)

            # Create a single-item list for link prediction store to enforce (u,v) display in graph

            single_prediction = [{

                'source': source_node,

                'target': target_node,

                'score': link_explanation.get('link_score', 0.0)

            }]

            print(f"SUCCESS (on-select): Link explanation completed for ({source_node}, {target_node})")

            return link_explanation, single_prediction

        else:

            return no_update, no_update

    except Exception as e:

        print(f"ERROR in generate_link_explanation_on_select: {e}")

        return {'error': f'On-select explanation failed: {str(e)}'}, no_update

@callback(

    Output('explanation-store', 'data', allow_duplicate=True),

    Output('global-run-button', 'children', allow_duplicate=True),

    Output('global-run-button', 'disabled', allow_duplicate=True),

    Input('global-run-button', 'n_clicks'),

    State('selected-node-store', 'data'),

    State('selected-link-store', 'data'),

    State({'type': 'lp-input', 'index': ALL}, 'value'), # Capture manual link inputs

    State('editable-graph-store', 'data'),

    State('precomputed-package-store', 'data'), State('full-dataset-store', 'data'),

    State('current-predictions-store', 'data'), State('model-dropdown', 'value'),

    State('analysis-mode-store', 'data'),

    State('layer3-tabs', 'value'),

    prevent_initial_call=True

)
@serialize_model_use
def run_explainer(n_clicks, selected_node, selected_link, lp_inputs, graph_data, package_meta, dataset_meta, predictions_data, model_type, analysis_mode, active_layer3_tab):

    try:

        if n_clicks == 0:
            return no_update, no_update, no_update

        # Link mode: run_link_prediction_unified owns the run-button reset
        if analysis_mode != 'node_classification':
            return no_update, no_update, no_update

        # From here on this callback must always re-enable the run button,
        # otherwise the clientside "Processing..." state never clears.
        if not all([package_meta, dataset_meta, graph_data, predictions_data]):
            return no_update, "RUN ANALYSIS", False

        if active_layer3_tab == 'tab-stability':
            return no_update, "RUN ANALYSIS", False

        dataset = get_cached_dataset_if_missing(dataset_meta)
        package = get_cached_package_if_missing(package_meta)
        if not dataset or not package:
            return no_update, "RUN ANALYSIS", False

        print(f"Running explainer with {model_type} model in {analysis_mode} mode...")

        model = get_cached_model(package_meta, dataset_meta['num_features'], dataset_meta['num_classes'])
        if model is None:
            print(f"ERROR: Could not instantiate {model_type} model for explanation")
            return no_update, "RUN ANALYSIS", False

        if analysis_mode == 'node_classification' and selected_node and selected_node.get('id') is not None:
            try:
                node_idx = int(selected_node['id'])
            except (TypeError, ValueError):
                return {'error': f"Invalid node ID: {selected_node['id']}"}, "RUN ANALYSIS", False
            num_nodes = dataset_meta.get('num_nodes') or len(predictions_data.get('preds', []))
            if not (0 <= node_idx < num_nodes):
                return {'error': f"Node {node_idx} is out of range (0-{num_nodes - 1})."}, "RUN ANALYSIS", False
            pyg_data = prepare_torch_data(dataset_meta, graph_data, include_labels=False)
            if pyg_data is None:
                return no_update, "RUN ANALYSIS", False
            target = torch.tensor([predictions_data['preds'][node_idx]], device=device)
            explainer = Explainer(model=model, algorithm=GNNExplainer(epochs=100), explanation_type='model', edge_mask_type='object', model_config=dict(mode='multiclass_classification', task_level='node', return_type='log_probs'))
            explanation = explainer(x=pyg_data.x, edge_index=pyg_data.edge_index, target=target, index=node_idx)

            # Calculate node importance from edge mask

            node_importance = {}

            if explanation.get('edge_mask') is not None:

                edge_mask = explanation.get('edge_mask').cpu()

                edge_index_cpu = pyg_data.edge_index.cpu()

                # Aggregate edge importance for each neighbor

                for i in range(edge_index_cpu.size(1)):

                    src, dst = edge_index_cpu[0, i].item(), edge_index_cpu[1, i].item()

                    # Find edges connected to the target node

                    if dst == node_idx or src == node_idx:

                        if i < len(edge_mask):

                            importance = abs(edge_mask[i].item())

                            neighbor = src if dst == node_idx else dst

                            if neighbor in node_importance:

                                node_importance[neighbor] = max(node_importance[neighbor], importance)

                            else:

                                node_importance[neighbor] = importance

            print(f"SUCCESS: Node explanation completed with {model_type}")

            # Try to compute feature importance via gradient*input as a fallback
            feature_imp = None
            try:
                model.eval()
                x_in = pyg_data.x.clone().detach().to(device)
                x_in.requires_grad = True
                edge_index_dev = pyg_data.edge_index.to(device)
                # Forward
                out = model(x_in, edge_index_dev)
                # predicted class for node
                pred_class = int(predictions_data['preds'][node_idx]) if predictions_data and 'preds' in predictions_data else None
                if pred_class is None:
                    pred_class = int(torch.argmax(out[node_idx]).item())
                score = out[node_idx, pred_class]
                # Backward
                model.zero_grad()
                if x_in.grad is not None:
                    x_in.grad.zero_()
                score.backward(retain_graph=False)
                grads = x_in.grad[node_idx].cpu().abs().numpy()
                vals = (grads * x_in.detach().cpu().numpy()[node_idx]).astype(float)
                # If multiplication yields zeros, fall back to abs gradients
                if not vals.any():
                    vals = grads
                feature_imp = {i: float(vals[i]) for i in range(len(vals))}
            except Exception:
                feature_imp = None

            return {
                'edge_mask': explanation.get('edge_mask').cpu().tolist() if explanation.get('edge_mask') is not None else None,
                'node_importance': node_importance,
                'feature_importance': feature_imp,
                'explanation_type': 'node_classification',
                'node_idx': node_idx,
            }, "RUN ANALYSIS", False

        return no_update, "RUN ANALYSIS", False

    except Exception as e:

        print(f"ERROR in run_explainer: {e}")

        import traceback

        traceback.print_exc()

        return {'error': f'Explanation failed with {model_type}: {str(e)}'}, "RUN ANALYSIS", False

@callback(

    Output('link-prediction-store', 'data', allow_duplicate=True),

    Output('link-status-store', 'data'),

    Output('global-run-button', 'children', allow_duplicate=True),

    Output('global-run-button', 'disabled', allow_duplicate=True),

    Input('predict-links-button', 'n_clicks'),

    Input('global-run-button', 'n_clicks'),

    State('selected-node-store', 'data'),

    State('selected-link-store', 'data'),

    State({'type': 'lp-input', 'index': ALL}, 'value'),

    State({'type': 'lp-input', 'index': ALL}, 'id'),

    State('link-source-input', 'value'),

    State('precomputed-package-store', 'data'),

    State('full-dataset-store', 'data'),

    State('editable-graph-store', 'data'),

    State('model-dropdown', 'value'),

    State('analysis-mode-store', 'data'),

    prevent_initial_call=True

)
@serialize_model_use
def run_link_prediction_unified(predict_clicks, global_clicks,

                                selected_node, selected_link, lp_inputs, lp_input_ids, link_source_input,

                                package_meta, dataset_meta, graph_data, model_type, analysis_mode):

    try:
        # DEBUG: record entry and basic state for reproduction tracing
        try:
            with open('link_debug.log', 'a') as _log:
                _log.write(f"RUN_LINK_UNIFIED triggered_id={ctx.triggered_id} selected_node={selected_node} selected_link_present={bool(selected_link)} link_source_input={link_source_input} analysis_mode={analysis_mode}\n")
        except Exception:
            pass

        triggered_id = ctx.triggered_id

        if not triggered_id:

            return no_update, no_update, no_update, no_update

        dataset = get_cached_dataset_if_missing(dataset_meta)
        package = get_cached_package_if_missing(package_meta)
        if not dataset:

            return no_update, {'text': 'Dataset not yet loaded.', 'type': 'error'}, "RUN ANALYSIS", False

        n_clicks = global_clicks if triggered_id == 'global-run-button' else predict_clicks

        if not n_clicks or n_clicks == 0:

            return no_update, no_update, no_update, no_update

        manual_source = None

        manual_target = None

        if lp_inputs and lp_input_ids:

            for val, id_dict in zip(lp_inputs, lp_input_ids):

                if id_dict['index'] == 'source-active':

                    manual_source = val

                elif id_dict['index'] == 'target-active':

                    manual_target = val

                elif id_dict['index'] == 'pred-node' and manual_source is None:

                    manual_source = val

        if triggered_id == 'global-run-button':

            if analysis_mode != 'link_prediction':

                return no_update, no_update, no_update, no_update

            if manual_source is not None and manual_target is not None:

                try:

                    u, v = int(manual_source), int(manual_target)

                    if not (0 <= u < dataset['num_nodes'] and 0 <= v < dataset['num_nodes']):
                         return no_update, {'text': f"Node IDs must be between 0 and {dataset['num_nodes'] - 1} (got {u}, {v}).", "type": "error"}, "RUN ANALYSIS", False
                    if u == v:
                         return no_update, {'text': "Source and target must be different nodes.", "type": "error"}, "RUN ANALYSIS", False

                    model = get_cached_model(package_meta, dataset_meta['num_features'], dataset_meta['num_classes'])

                    if model is None:
                        return no_update, {'text': "Model load failed.", "type": "error"}, "RUN ANALYSIS", False

                    pyg_data = prepare_torch_data(dataset_meta, graph_data, include_labels=False)

                    with torch.no_grad():

                        _, embeddings = model(pyg_data.x, pyg_data.edge_index, return_embeddings=True)
                        edge_tensor = torch.tensor([[u], [v]], device=device)
                        link_score = model.predict_links(embeddings, edge_tensor)
                        prob = torch.sigmoid(link_score).item()

                    prediction_result = [{'source': u, 'target': v, 'score': prob}]

                    status = {'text': f"Predicted link {u} → {v} with score {prob:.4f}", "type": "success"}

                    return prediction_result, status, "RUN ANALYSIS", False

                except ValueError:
                    return no_update, {'text': "Invalid node input (must be integers).", "type": "error"}, "RUN ANALYSIS", False

            # Link mode needs an explicit pair: both source and target are required
            return no_update, {'text': "Enter both a source and a target node ID, then press RUN ANALYSIS.", "type": "error"}, "RUN ANALYSIS", False

        node_idx = None

        if manual_source is not None:

            node_idx = int(manual_source)

        elif selected_node and selected_node.get('id') is not None:

            node_idx = int(selected_node['id'])

        elif selected_link and selected_link.get('source') is not None:

            node_idx = int(selected_link['source'])

        elif link_source_input is not None:

            node_idx = int(link_source_input)

        if node_idx is None:
            # If in link prediction mode and Run pressed without explicit source,
            # fallback to node 0 and continue so the UI shows the Link CM instead
            # of silently doing nothing.
            try:
                if analysis_mode == 'link_prediction':
                    node_idx = 0
                    print("[LINK-PRED] No source provided on Run — defaulting to node 0")
                else:
                    return no_update, {'text': "Please select a source node for link prediction.", "type": "error"}, "RUN ANALYSIS", False
            except Exception:
                return no_update, {'text': "Please select a source node for link prediction.", "type": "error"}, "RUN ANALYSIS", False

        if not isinstance(node_idx, int) or node_idx < 0 or node_idx >= dataset['num_nodes']:

            return no_update, {'text': f"Node {node_idx} is out of range for the current graph.", "type": "error"}, "RUN ANALYSIS", False

        model = get_cached_model(package_meta, dataset_meta['num_features'], dataset_meta['num_classes'])

        if model is None:
            return no_update, {'text': "Link prediction failed: Could not load model.", "type": "error"}, "RUN ANALYSIS", False

        if not hasattr(model, 'enable_link_prediction') or not model.enable_link_prediction:

            return no_update, {'text': "Link prediction not enabled for this model.", "type": "error"}, "RUN ANALYSIS", False

        pyg_data = prepare_torch_data(dataset_meta, graph_data, include_labels=False)

        with torch.no_grad():

            _, embeddings = model(pyg_data.x, pyg_data.edge_index, return_embeddings=True)

        candidates = create_link_prediction_candidates(pyg_data, target_node=node_idx, exclude_existing=True)

        if candidates.size(1) == 0:

            return [], {'text': f"No candidate links found for Node {node_idx}.", "type": "info"}, "RUN ANALYSIS", False

        top_predictions = get_top_k_predictions(embeddings, candidates.to(device), model, k=50)

        status = {'text': f"Found {len(top_predictions)} link predictions for Node {node_idx}.", "type": "success"}

        return top_predictions, status, "RUN ANALYSIS", False

    except Exception as e:

        print(f"Link prediction callback error: {e}")

        import traceback

        traceback.print_exc()

        return no_update, {'text': f"Link prediction failed: {str(e)}", "type": "error"}, "RUN ANALYSIS", False

@callback(

    Output('cpa-iv-store', 'data'),

    Output('cpa-status-store', 'data'),

    Input('run-cpa-iv-button', 'n_clicks'),

    State('selected-node-store', 'data'),

    State('precomputed-package-store', 'data'),

    State('full-dataset-store', 'data'),

    State('editable-graph-store', 'data'),

    State('model-dropdown', 'value'),

    prevent_initial_call=True

)
@serialize_model_use
def run_cpa_iv_callback(n_clicks, selected_node, package_meta, dataset_meta, graph_data, model_type):

    try:

        print(f"[CPA-IV-SIMPLE] Triggered run_cpa_iv_callback; n_clicks={n_clicks}, selected_node={selected_node}")
        if n_clicks == 0 or not selected_node:
            return no_update, no_update

        dataset = get_cached_dataset_if_missing(dataset_meta)
        package = get_cached_package_if_missing(package_meta)
        if not all([package, dataset, graph_data]):
            return no_update, no_update

        try:
            node_idx = int(selected_node['id'])
        except Exception as e:
            print(f"[CPA-IV-SIMPLE] Failed to parse selected_node id: {selected_node} -> {e}")
            return no_update, {'text': "CPA-IV failed: invalid selected node.", "type": "error"}
        print(f"[CPA-IV-SIMPLE] Resolved node_idx={node_idx}")

        model = get_cached_model(package_meta, dataset_meta['num_features'], dataset_meta['num_classes'])
        if model is None:
            return no_update, {'text': "CPA-IV failed: Could not load model.", "type": "error"}

        pyg_data = prepare_torch_data(dataset_meta, graph_data, include_labels=False)
        causal_paths = run_cpa_iv(model, pyg_data, node_idx)

        status = {'text': f"CPA-IV found {len(causal_paths)} influential paths for Node {node_idx}.", "type": "success"}
        return causal_paths, status

    except Exception as e:

        print(f"CPA-IV callback error: {e}")
        return no_update, {'text': f"CPA-IV failed: {str(e)}", "type": "error"}

# Callback for active CPA-IV — triggered by Layer 1 global button OR Layer 3 causal button

@callback(
    Output('cpa-iv-store', 'data', allow_duplicate=True),
    Output('link-cpa-iv-store', 'data', allow_duplicate=True),
    Output('cpa-status-store', 'data', allow_duplicate=True),
    Input('global-run-button', 'n_clicks'),
    Input('run-cpa-iv-button', 'n_clicks'),
    Input('selected-link-store', 'data'),
    Input('selected-node-store', 'data'),
    State('selected-link-store', 'data'),
    State('precomputed-package-store', 'data'),
    State('full-dataset-store', 'data'),
    State('editable-graph-store', 'data'),
    State('model-dropdown', 'value'),
    State('analysis-mode-store', 'data'),
    State('layer3-tabs', 'value'),
    State({'type': 'lp-input', 'index': ALL}, 'value'),
    State({'type': 'lp-input', 'index': ALL}, 'id'),
    State('link-source-input', 'value'),
    State('link-target-input', 'value'),
    prevent_initial_call=True
)
@serialize_model_use
def run_cpa_iv_callback_active(global_clicks, layer3_clicks, selected_link_trigger,
                               selected_node, selected_link_state, package_meta, dataset_meta, graph_data, model_type,
                               analysis_mode, active_layer3_tab,
                               lp_inputs, lp_input_ids,
                               layer3_source, layer3_target):
    triggered = ctx.triggered_id
    # A row click in the cell-details table sets BOTH selected-link-store and
    # selected-node-store; ctx.triggered_id may then name the node store, whose
    # branch bails out in link mode. Resolve against every input that changed.
    _changed = {t['prop_id'].split('.')[0] for t in (ctx.triggered or [])}
    if analysis_mode == 'link_prediction' and 'selected-link-store' in _changed:
        triggered = 'selected-link-store'
    elif analysis_mode == 'node_classification' and 'selected-node-store' in _changed:
        triggered = 'selected-node-store'

    if triggered == 'global-run-button':
        if (global_clicks or 0) == 0:
            return no_update, no_update, no_update
        if active_layer3_tab == 'tab-stability':
            return no_update, no_update, no_update
    elif triggered == 'run-cpa-iv-button':
        if (layer3_clicks or 0) == 0:
            return no_update, no_update, no_update
    # If the selected-link-store changed, attempt to run CPA for the selected link
    elif triggered == 'selected-link-store':
        # Only run for link prediction mode and when a valid selection exists
        if analysis_mode != 'link_prediction':
            return no_update, no_update, no_update
        if not selected_link_trigger:
            return no_update, no_update, no_update
        # Proceed — treat selection as the trigger
    # If the selected-node-store changed, attempt to run CPA for the selected node
    elif triggered == 'selected-node-store':
        # Only run for node classification mode and when a valid selection exists
        if analysis_mode != 'node_classification':
            return no_update, no_update, no_update
        if not selected_node:
            return no_update, no_update, no_update
        # Proceed — treat selection as the trigger
    else:
        return no_update, no_update, no_update

    dataset = get_cached_dataset_if_missing(dataset_meta)
    package = get_cached_package_if_missing(package_meta)
    if not all([package, dataset, graph_data]):
        return no_update, no_update, {'text': "Load a dataset and model first.", "type": "error"}

    print(f"[CPA-IV] Triggered by {triggered}. Mode: {analysis_mode}")
    print(f"[CPA-IV] layer3_source={layer3_source}, selected_node={selected_node}")
    # selected_link may come from either State or the selected-link input; prefer trigger
    sel_link = selected_link_trigger if triggered == 'selected-link-store' else selected_link_state
    print(f"[CPA-IV] selected_link (raw): {sel_link}")

    model = get_cached_model(package_meta, dataset_meta['num_features'], dataset_meta['num_classes'])
    if model is None:
        return no_update, no_update, {'text': "CPA-IV failed: could not load model.", "type": "error"}

    pyg_data = prepare_torch_data(dataset_meta, graph_data, include_labels=False)

    # ── Mode A: Node Classification ──────────────────────────────────────────
    if analysis_mode == 'node_classification':
        # Layer 3 source input overrides selected-node-store when present
        node_idx = None
        if layer3_source is not None:
            try: node_idx = int(layer3_source)
            except (ValueError, TypeError): pass
        if node_idx is None and selected_node:
            try: node_idx = int(selected_node['id'])
            except (ValueError, TypeError, KeyError): pass
        if node_idx is None:
            # Silent no-op when no node is selected in node-classification mode
            return no_update, no_update, no_update
        try:
            if int(node_idx) < 0:
                # Treat negative indices (e.g. -1 used for 'no selection') as no selection
                return no_update, no_update, no_update
        except Exception:
            # If we can't interpret node_idx as int, fall through to error handling below
            pass

        causal_paths = run_cpa_iv(model, pyg_data, node_idx)
        status = {'text': f"CPA-IV: {len(causal_paths)} influential paths for Node {node_idx}.", "type": "success"}
        return causal_paths, no_update, status


    

    # ── Mode B: Link Prediction ───────────────────────────────────────────────
    elif analysis_mode == 'link_prediction':
        u, v = None, None

        # If the callback was triggered by a selected-link change, prefer that selection
        if triggered == 'selected-link-store' and sel_link and isinstance(sel_link, dict):
            try:
                u = int(sel_link.get('source'))
                v = int(sel_link.get('target'))
                print(f"[CPA-IV] Triggered selection used: {u} → {v}")
            except Exception as e:
                print(f"[CPA-IV] Failed to parse triggered selection: {sel_link} -> {e}")

        # Priority 1: Layer 3 dedicated inputs (causal tab) — only use if no triggered selection
        if (u is None or v is None) and layer3_source is not None and layer3_target is not None:
            try:
                u, v = int(layer3_source), int(layer3_target)
                print(f"[CPA-IV] Layer 3 inputs: {u} → {v}")
            except (ValueError, TypeError):
                u, v = None, None

        # Priority 2: Layer 1 lp-input pattern components (GNN/non-causal LP mode)
        if (u is None or v is None) and lp_inputs and lp_input_ids:
            manual_source, manual_target = None, None
            for val, id_dict in zip(lp_inputs, lp_input_ids):
                if id_dict['index'] == 'source-active':   manual_source = val
                elif id_dict['index'] == 'target-active': manual_target = val
            if manual_source is not None and manual_target is not None:
                try:
                    u, v = int(manual_source), int(manual_target)
                    print(f"[CPA-IV] Layer 1 lp-inputs: {u} → {v}")
                except (ValueError, TypeError):
                    u, v = None, None

        # Priority 3: Graph-click selected link
        sel = sel_link if sel_link is not None else None
        if (u is None or v is None) and sel and isinstance(sel, dict):
            s = sel.get('source')
            t = sel.get('target')
            try:
                if s is not None and t is not None:
                    u = int(s)
                    v = int(t)
                    print(f"[CPA-IV] Selected link parsed: {u} → {v}")
            except Exception as e:
                print(f"[CPA-IV] Failed to parse selected_link source/target: {sel} -> {e}")
                u, v = None, None

        if u is None or v is None:
            return no_update, no_update, {
                'text': "CPA-IV: enter Source & Target node IDs in the Explanation panel.",
                "type": "error"
            }
        if not (0 <= u < dataset['num_nodes'] and 0 <= v < dataset['num_nodes']):
            return no_update, no_update, {'text': f"Node IDs {u}, {v} out of range (max {dataset['num_nodes']-1}).", "type": "error"}

        print(f"[CPA-IV] Running link analysis {u} → {v}…")
        cpa_result = run_cpa_iv_link(model, pyg_data, u, v)

        if 'error' in cpa_result:
            return no_update, no_update, {'text': f"CPA-IV Error: {cpa_result['error']}", "type": "error"}

        num_paths = len(cpa_result.get('paths', []))
        print(f"[CPA-IV] Link success: {num_paths} paths, verdict: {cpa_result.get('verdict')}")
        status = {'text': f"CPA-IV: {num_paths} causal paths for link {u}→{v}.", "type": "success"}
        return no_update, cpa_result, status

    return no_update, no_update, no_update

# ── Layer 3 Causal status line ────────────────────────────────────────────────

@callback(
    Output('causal-layer3-status', 'children'),
    Output('causal-layer3-status', 'style'),
    Input('cpa-status-store', 'data'),
)
def update_causal_layer3_status(status_data):
    base_style = {'fontSize': '11px', 'marginBottom': '10px',
                  'minHeight': '16px', 'fontStyle': 'italic',
                  'padding': '5px 8px', 'borderRadius': '4px'}
    if not status_data:
        return ("Select 'Causal' explainer, enter node IDs above, then click Run.",
                {**base_style, 'color': '#999', 'backgroundColor': 'transparent'})

    msg  = status_data.get('text', '')
    kind = status_data.get('type', 'info')
    if kind == 'success':
        style = {**base_style, 'color': '#155724', 'backgroundColor': '#d4edda', 'border': '1px solid #c3e6cb'}
    elif kind == 'error':
        style = {**base_style, 'color': '#721c24', 'backgroundColor': '#f8d7da', 'border': '1px solid #f5c6cb'}
    else:
        style = {**base_style, 'color': '#0c5460', 'backgroundColor': '#d1ecf1', 'border': '1px solid #bee5eb'}
    return msg, style


# --- SPLIT CALLBACKS ---

# 1. Update Mid Layer (Global Views or Causal Summary)

@callback(
    Output('embeddings-view', 'figure'),
    Output('confusion-matrix-plot', 'figure', allow_duplicate=True),
    Output('gnn-mid-layer-content', 'style'),
    Input('global-run-button', 'n_clicks'),
    Input('current-predictions-store', 'data'),
    Input('cpa-iv-store', 'data'),
    Input('link-cpa-iv-store', 'data'),
    Input('embedding-color-mode', 'value'),
    Input('selected-node-store', 'data'),
    Input('selected-link-store', 'data'),
    Input('analysis-mode-store', 'data'),
    State('precomputed-package-store', 'data'),
    State('full-dataset-store', 'data'),
    State('layer3-tabs', 'value'),
    State('link-prediction-store', 'data'),
    prevent_initial_call=True
)
def update_mid_layer(n_clicks, predictions_data, cpa_data, link_cpa_data, embedding_color_mode, selected_node, selected_link, analysis_mode, package, dataset, active_layer3_tab, link_prediction_store):
    try:
        empty_fig = empty_figure("Loading model…")
        gnn_style = {'flex': '0.8', 'minWidth': '260px', 'minHeight': '0', 'height': '100%', 'maxHeight': '100%',
                     'margin': '0', 'overflow': 'hidden', 'boxSizing': 'border-box',
                     'display': 'flex', 'flexDirection': 'column'}

        # Ensure we have the full cached dataset (callbacks may pass metadata only)
        dataset_full = get_cached_dataset_if_missing(dataset)
        package_meta = package
        package = get_cached_package_if_missing(package_meta)
        if not predictions_data or not package or not dataset_full:
            return empty_fig, empty_fig, gnn_style

        preds = np.array(predictions_data['preds'])
        true_labels = np.array(dataset_full['y'])
        num_classes = dataset_full['num_classes']
        dataset_name = dataset_full.get('name')
        class_to_color_map = class_color_map(num_classes)

        selected_node_id_str = selected_node['id'] if selected_node else None
        target_node_id_str = None
        if analysis_mode == 'link_prediction' and selected_link and isinstance(selected_link, dict):
            target = selected_link.get('target')
            if target is not None:
                target_node_id_str = str(target)
            # mark the link's source with the star (not a node selected in node view)
            if selected_link.get('source') is not None:
                selected_node_id_str = str(selected_link['source'])

        embeddings_2d = get_2d_embeddings(package)
        if embeddings_2d is None:
            embeddings_fig = empty_figure("Embeddings not available")
        else:
            try:
                embeddings_fig = plot_embeddings(np.array(embeddings_2d), preds, true_labels, selected_node_id_str, class_to_color_map, num_classes,
                                                 target_node_id=target_node_id_str, dataset_name=dataset_name,
                                                 color_mode=embedding_color_mode or 'actual',
                                                 confidences=predictions_data.get('confidences'))
            except Exception:
                # Keep going: a failed embedding plot must not blank the confusion matrix
                traceback.print_exc()
                embeddings_fig = empty_figure("Could not draw embeddings - change colour mode to retry")


        # In Link Prediction mode the confusion matrix is produced by the
        # subgraph/link-evaluation callback. Avoid overwriting that figure
        # by returning `no_update` for the CM output so other callbacks can
        # set it (prevents brief flash + overwrite behavior).
        if analysis_mode == 'link_prediction':
            return embeddings_fig, no_update, gnn_style

        cm_fig = node_confusion_figure(true_labels, preds, num_classes, dataset_name)

        return embeddings_fig, cm_fig, gnn_style

    except Exception as e:
        print(f"Error in update_mid_layer: {e}")
        import traceback
        traceback.print_exc()
        return (empty_figure("Could not draw embeddings"), empty_figure("Could not draw confusion matrix"),
                {'flex': '0.8', 'minWidth': '260px', 'minHeight': '0', 'height': '100%', 'maxHeight': '100%',
                 'margin': '0', 'overflow': 'hidden', 'boxSizing': 'border-box',
                 'display': 'flex', 'flexDirection': 'column'})


# Auto-run CPA-IV callback removed; CPA is handled by the main CPA trigger and GNN panel.

# cpa_combined_debug callback removed (debug output suppressed)


# Node classification performance metrics — populate the same `performance-metrics` area
@callback(
    Output('performance-metrics', 'children', allow_duplicate=True),
    Output('performance-metrics', 'style', allow_duplicate=True),
    Input('global-run-button', 'n_clicks'),
    Input('current-predictions-store', 'data'),
    Input('analysis-mode-store', 'data'),
    State('full-dataset-store', 'data'),
    prevent_initial_call=True,
)
def update_node_performance_metrics(run_clicks, predictions_data, analysis_mode, dataset_meta):
    try:
        # Node-classification metrics, shown as soon as predictions exist
        if analysis_mode != 'node_classification':
            return no_update, no_update

        if not predictions_data or not dataset_meta:
            return no_update, no_update

        dataset_full = get_cached_dataset_if_missing({'name': dataset_meta.get('name')}) if isinstance(dataset_meta, dict) else None
        if not dataset_full or 'y' not in dataset_full:
            return no_update, no_update

        preds = np.array(predictions_data.get('preds', []))
        true = np.asarray(dataset_full['y'])
        if preds.size == 0 or true.size == 0:
            return no_update, no_update

        from sklearn.metrics import accuracy_score, f1_score

        items = []
        test_mask = dataset_full.get('test_mask')
        if test_mask is not None:
            tm = np.asarray(test_mask, dtype=bool)
            items.append(("Test acc", float(accuracy_score(true[tm], preds[tm])),
                          f"Accuracy on the {int(tm.sum())} standard Planetoid test nodes"))
        items.append(("All-node acc", float(accuracy_score(true, preds)),
                      "Accuracy over every node, including the training nodes"))
        items.append(("Macro F1", float(f1_score(true, preds, average='macro', zero_division=0)),
                      "Macro-averaged F1 over every node"))
        metrics = metric_tiles(items)

        perf_style = {'display': 'flex'}

        return metrics, perf_style

    except Exception as e:
        print(f"Error in update_node_performance_metrics: {e}")
        return no_update, no_update

# 2. Update Bottom Layer (Local/Link Views)

@callback(

    Output('graph-view', 'elements'),

    Output('neighbor-importance-view', 'figure'),

    Output('feature-importance-view', 'figure'),

    Output('subgraph-legend-inline', 'children'),

    Output('layer3-right-header', 'children'),

    Output('gnn-causal-explanation-view', 'children'),

    Output('gnn-causal-explanation-view', 'style'),

    Input('selected-link-store', 'data'),

    Input('selected-node-store', 'data'),

    Input('explanation-store', 'data'),

    Input('cpa-iv-store', 'data'),
    
    Input('link-cpa-iv-store', 'data'),
    
    Input('selected-path-store', 'data'),

    Input('analysis-mode-store', 'data'),

    Input('subgraph-color-mode', 'value'),

    State('current-predictions-store', 'data'),

    State('full-dataset-store', 'data'),

    State('editable-graph-store', 'data'),

    State('precomputed-package-store', 'data'),

    State('model-dropdown', 'value'),

    State('layer3-tabs', 'value'),

    prevent_initial_call=True

)

def update_bottom_layer(selected_link, selected_node, explanation, cpa_data, link_cpa_data, selected_path, analysis_mode, subgraph_color_mode, predictions_data, dataset, graph_data, package_meta, model_type, active_layer3_tab):
    try:
        empty_fig = empty_figure()
        subgraph_color_mode = subgraph_color_mode or 'actual'

        # Visibility Rule: Only show if specific selection exists
        visible = False
        target_node_idx = None

        if analysis_mode == 'link_prediction':
            if selected_link and isinstance(selected_link, dict):
                visible = True
                target_node_idx = selected_link['source']  # Focus on source

        elif analysis_mode == 'node_classification':
            # Automatically use FULL graph for Node Task (always visible)
            visible = True
            if selected_node and isinstance(selected_node, dict):
                target_node_idx = int(selected_node['id'])
            else:
                target_node_idx = -1  # Indicate no selection

        # Ensure we have the full cached dataset (may be metadata passed in state)
        dataset = get_cached_dataset_if_missing(dataset)
        if not visible or not dataset or not graph_data or not predictions_data:
            hint = ("Pick a predicted link from the list above to inspect it."
                    if analysis_mode == 'link_prediction' else "Load a model to begin.")
            return [], empty_fig, empty_fig, html.Div(hint, className='empty-hint'), "Details", None, {'display': 'none'}

        preds = np.array(predictions_data['preds'])
        num_classes = dataset['num_classes']
        dataset_name = dataset.get('name')
        class_to_color_map = class_color_map(num_classes)
        # Class colours are already keyed in the node-embeddings view; only the
        # two-colour correctness key is shown here
        color_key = (subgraph_color_key(subgraph_color_mode, dataset_name, num_classes)
                     if subgraph_color_mode == 'correctness' else None)
        true_labels = dataset.get('y') if isinstance(dataset, dict) else None
        yvals = true_labels

        pyg_data = prepare_torch_data(dataset, graph_data, include_labels=False)
        if pyg_data is None:
            return [], empty_fig, empty_fig, "Unable to construct graph data", "Details", None, {'display': 'none'}

        # Path Selection
        selected_path_idx = selected_path.get('index') if selected_path else None

        # --- LOGIC BRANCHING ---
        # A. LINK PREDICTION MODE (STRICT LOCAL VIEW)
        if analysis_mode == 'link_prediction':
            u = selected_link['source']
            v = selected_link['target']
            # Use link_cpa_data for path info
            active_cpa_data = link_cpa_data

            # 1. CALCULATE INDUCED SUBGRAPH (Single Source of Truth)
            subset, sub_edge_index, _, _ = k_hop_subgraph(
                torch.tensor([u, v]),
                1,  # 1-hop neighborhood
                pyg_data.edge_index,
                relabel_nodes=False,
                num_nodes=dataset['num_nodes']
            )

            subgraph_nodes = subset.tolist()
            subgraph_edges = sub_edge_index.tolist()

            # 2. PREPARE DATA OBJECT FOR VIZ
            strict_explanation = {
                'explanation_type': 'link_prediction',
                'source_node': u,
                'target_node': v,
                'subgraph_nodes': subgraph_nodes,
                'subgraph_edges': subgraph_edges,
                'link_score': 0,  # Placeholder
                'node_importance': {}
            }

            # Check if we have a real explanation to augment our strict data
            real_score = 0.0
            if explanation and explanation.get('explanation_type') == 'link_prediction':
                if explanation.get('source_node') == u and explanation.get('target_node') == v:
                    strict_explanation['node_importance'] = explanation.get('node_importance', {})
                    real_score = explanation.get('link_score', 0.0)
                    strict_explanation['link_score'] = real_score

            # 3. GENERATE NEIGHBORHOOD DETAILS (Right Panel)
            # Hide the right-panel header when in link-detail view (we show the list elsewhere)
            header_text = ""

            # Determine prediction class (Edge vs No Edge) based on threshold
            predicted_label = 1 if real_score > 0.5 else 0
            pred_text = "Edge Predicted" if predicted_label == 1 else "No Edge"
            pred_color = CORRECT_COLOR if predicted_label == 1 else INCORRECT_COLOR

            # Ground Truth Check
            is_existing_link = False
            if 'original_edge_index' in dataset:
                edges = torch.tensor(dataset['original_edge_index'], dtype=torch.long)
            else:
                edges = pyg_data.edge_index

            src_edges, dst_edges = edges[0], edges[1]
            mask = ((src_edges == u) & (dst_edges == v)) | ((src_edges == v) & (dst_edges == u))
            if mask.any():
                is_existing_link = True

            # Override from CPA if available
            if active_cpa_data and isinstance(active_cpa_data, dict):
                is_existing_link = active_cpa_data.get('is_existing_link', is_existing_link)

            ground_truth_label = 1 if is_existing_link else 0
            is_correct = (predicted_label == ground_truth_label)
            correctness_label = "(True)" if is_correct else "(False)"
            badge_color = CORRECT_COLOR if is_correct else INCORRECT_COLOR

            # Additional Path Info
            path_info_div = html.Div()
            if selected_path_idx is not None and active_cpa_data and 'paths' in active_cpa_data:
                try:
                    p = active_cpa_data['paths'][selected_path_idx]
                    path_str = ' → '.join(map(str, p.get('nodes', [])))
                    path_info_div = html.Div([
                        html.Span("Highlighted Path:", style={'fontWeight': 'bold', 'color': '#e74c3c'}),
                        html.Div(path_str, style={'fontFamily': 'monospace', 'fontSize': '12px', 'marginTop': '3px'})
                    ], style={'marginTop': '10px', 'padding': '5px', 'border': '1px dashed #e74c3c', 'backgroundColor': '#fdedec'})
                except Exception:
                    pass

            mode_label = 'Predicted Class'
            if subgraph_color_mode == 'actual':
                mode_label = 'Actual Class'
            elif subgraph_color_mode == 'correctness':
                mode_label = 'Correctness'

            legend_content = html.Div([
                html.Div([
                    html.Span("Color Mode:", style={'fontWeight': 'bold', 'color': '#333'}),
                    html.Span(mode_label, style={'marginLeft': '5px', 'fontFamily': 'monospace', 'fontSize': '14px'})
                ], style={'marginBottom': '10px'}),
                html.Div([
                    html.Span("Selected Link:", style={'fontWeight': 'bold', 'color': '#333'}),
                    html.Span(f" {u} → {v}", style={'marginLeft': '5px', 'fontFamily': 'monospace', 'fontSize': '14px'})
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Span("Subgraph Size:", style={'fontWeight': 'bold', 'color': '#333'}),
                    html.Span(f" {len(subgraph_nodes)} nodes", style={'marginLeft': '5px'})
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Span("Link Score:", style={'fontWeight': 'bold', 'color': '#333'}),
                    html.Span(f" {real_score:.4f}", style={'marginLeft': '5px', 'fontWeight': 'bold'})
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Span("Prediction:", style={'fontWeight': 'bold', 'color': '#333'}),
                    html.Span(f" {pred_text}", style={'marginLeft': '5px', 'color': pred_color, 'fontWeight': 'bold'}),
                    html.Span(correctness_label, style={
                        'marginLeft': '8px',
                        'backgroundColor': badge_color,
                        'color': 'white',
                        'padding': '2px 8px',
                        'borderRadius': '10px',
                        'fontSize': '11px',
                        'fontWeight': 'bold',
                        'display': 'inline-block'
                    })
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Span("View Radius:", style={'fontWeight': 'bold', 'color': '#333'}),
                    html.Span(f" 1-hop", style={'marginLeft': '5px', 'color': '#666'})
                ], style={'marginBottom': '8px'}),
                path_info_div,
                color_key,
            ], style={'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px', 'border': '1px solid #dee2e6'})

            # 4. RENDER GRAPH
            layer3_link_predictions = [{'source': u, 'target': v, 'score': real_score}]
            graph_elements, _ = create_subgraph_cytoscape(
                pyg_data, u, preds, class_to_color_map,
                strict_explanation, active_cpa_data, layer3_link_predictions, analysis_mode, selected_path_idx,
                color_mode=subgraph_color_mode,
                true_labels=true_labels
            )

            # 5. IMPORTANCE PLOT - show neighbor importance and feature importance in link-detail view
            # Compute neighbor importance for the induced subgraph (exclude source/target)
            try:
                node_imp = {}
                from torch_geometric.utils import to_networkx
                import networkx as nx
                # sub_edge_index is already a torch tensor from k_hop_subgraph
                nx_graph = to_networkx(Data(edge_index=sub_edge_index, num_nodes=pyg_data.num_nodes), to_undirected=True)
                sub_G = nx_graph.subgraph(subgraph_nodes)
                centrality = nx.degree_centrality(sub_G)
                # Exclude the source and target nodes from neighbor importance
                for n, s in centrality.items():
                    if int(n) not in (int(u), int(v)):
                        node_imp[int(n)] = float(s)
                if node_imp:
                    neighbor_fig = plot_neighbor_importance(node_imp, u, preds=preds, y_labels=true_labels,
                                                            dataset_name=dataset_name)
                    try:
                        neighbor_fig.update_layout(title=f'Top Neighbor Importance')
                    except Exception:
                        pass
                else:
                    neighbor_fig = empty_fig
            except Exception:
                neighbor_fig = empty_fig

            # Feature importance for link-prediction: prefer CPA feature_analysis, else fallback to raw features
            try:
                feature_fig = empty_fig
                print(f"[LINK-FEAT] link {u}->{v} active_cpa_data? {bool(active_cpa_data)}")
                print(f"[LINK-FEAT] pyg_data.x present? {hasattr(pyg_data,'x') and pyg_data.x is not None}")
                # 1) CPA-driven feature analysis
                if active_cpa_data and isinstance(active_cpa_data, dict) and 'feature_analysis' in active_cpa_data:
                    fa = active_cpa_data['feature_analysis']
                    print(f"[LINK-FEAT] CPA feature_analysis keys: {list(fa.keys()) if isinstance(fa, dict) else 'n/a'}")
                    rows = []
                    for item in fa.get('source_importance', []) or []:
                        rows.append({'Feature': f"{item.get('index')}", 'Score': float(item.get('score', 0)), 'Type': 'Source (u)'} )
                    for item in fa.get('target_importance', []) or []:
                        rows.append({'Feature': f"{item.get('index')}", 'Score': float(item.get('score', 0)), 'Type': 'Target (v)'} )
                    df_fa = pd.DataFrame(rows)
                    print(f"[LINK-FEAT] CPA df rows: {len(df_fa)}")
                    if not df_fa.empty:
                        # keep top features by absolute score across both
                        df_agg = df_fa.groupby('Feature', as_index=False).agg({'Score': lambda s: float(np.abs(s).sum())})
                        top_feats = df_agg.sort_values('Score', ascending=False).head(10)['Feature'].tolist()
                        df_plot = df_fa[df_fa['Feature'].isin(top_feats)]
                        print(f"[LINK-FEAT] CPA df_plot rows: {len(df_plot)}; head: {df_plot.head().to_dict('list')}")
                        feature_fig = px.bar(df_plot, x='Feature', y='Score', color='Type', barmode='group',
                                            title=f'Top Feature Importance for Link {u} → {v}')
                        feature_fig.update_layout(xaxis_type='category', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        print(f"[LINK-FEAT] created CPA feature_fig with {len(feature_fig.data)} traces")
                        for ti, tr in enumerate(feature_fig.data):
                            print(f"[LINK-FEAT] trace {ti}: name={getattr(tr,'name',None)}, x_sample={tr.x[:5] if hasattr(tr,'x') else None}, y_sample={tr.y[:5] if hasattr(tr,'y') else None}")
                else:
                    # 2) Fallback: use raw node features
                    if hasattr(pyg_data, 'x') and pyg_data.x is not None:
                        xu = pyg_data.x[int(u)]
                        xv = pyg_data.x[int(v)]
                        print(f"[LINK-FEAT] xu shape: {getattr(xu,'shape', None)}, xv shape: {getattr(xv,'shape', None)}")
                        if hasattr(xu, 'detach'):
                            xu_arr = xu.detach().cpu().numpy()
                            xv_arr = xv.detach().cpu().numpy()
                        else:
                            xu_arr = np.array(xu)
                            xv_arr = np.array(xv)
                        print(f"[LINK-FEAT] xu_arr size: {xu_arr.size}, xv_arr size: {xv_arr.size}")
                        if xu_arr.size > 0 and xv_arr.size > 0:
                            sum_abs = np.abs(xu_arr) + np.abs(xv_arr)
                            idxs = np.argsort(sum_abs)[-10:][::-1]
                            print(f"[LINK-FEAT] selected feature idxs: {idxs}")
                            rows = []
                            for i in idxs:
                                rows.append({'Feature': str(int(i)), 'Score': float(xu_arr[int(i)]), 'Type': 'Source (u)'} )
                                rows.append({'Feature': str(int(i)), 'Score': float(xv_arr[int(i)]), 'Type': 'Target (v)'} )
                            df_plot = pd.DataFrame(rows)
                            print(f"[LINK-FEAT] fallback df_plot rows: {len(df_plot)}; head: {df_plot.head().to_dict('list')}")
                            feature_fig = px.bar(df_plot, x='Feature', y='Score', color='Type', barmode='group',
                                                title=f'Top Features for Link {u} → {v}')
                            feature_fig.update_layout(xaxis_type='category', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                            print(f"[LINK-FEAT] created fallback feature_fig with {len(feature_fig.data)} traces")
                            for ti, tr in enumerate(feature_fig.data):
                                print(f"[LINK-FEAT] trace {ti}: name={getattr(tr,'name',None)}, x_sample={tr.x[:5] if hasattr(tr,'x') else None}, y_sample={tr.y[:5] if hasattr(tr,'y') else None}")
                # If still empty, keep empty_fig
            except Exception:
                feature_fig = empty_fig
            causal_content = None
            attention_content = None
            causal_style = {'display': 'none'}
            if active_cpa_data:
                causal_content = create_cpa_iv_panel(active_cpa_data, f"{u} \u2192 {v}")
            if model_capabilities(model_type)['attention']:
                attention_content = create_attention_panel(
                    get_attention_weights(package_meta),
                    f"{u} \u2192 {v}"
                )
            if causal_content or attention_content:
                causal_content = explanation_cards(causal_content, attention_content)
                causal_style = {'display': 'block', 'marginTop': '16px'}
            return graph_elements, neighbor_fig, feature_fig, legend_content, header_text, causal_content, causal_style

        # B. NODE CLASSIFICATION MODE (Standard)
        else:
            header_text = "Node Neighborhood & Importance"
            # Never show one node's explanation on another node's subgraph
            if isinstance(explanation, dict) and explanation.get('node_idx') is not None \
                    and int(explanation['node_idx']) != int(target_node_idx):
                explanation = None
            if target_node_idx != -1:
                # Determine actual class if available in dataset
                actual_class = None
                try:
                    if yvals is None:
                        if isinstance(dataset, dict) and 'y' in dataset:
                            yvals = dataset.get('y')
                        elif hasattr(dataset, 'y'):
                            yvals = dataset.y
                    if yvals is not None and len(yvals) > target_node_idx:
                        val = yvals[target_node_idx]
                        if hasattr(val, 'item'):
                            actual_class = int(val.item())
                        else:
                            actual_class = int(val)
                except Exception:
                    actual_class = None
                    yvals = None

                pred_cls = int(preds[target_node_idx])
                is_correct = actual_class is not None and pred_cls == actual_class
                conf = None
                try:
                    conf = float(predictions_data.get('confidences', [])[target_node_idx])
                except Exception:
                    pass
                conf_text = f"  ({conf:.0%})" if conf is not None else ""
                legend_content = html.Div([
                    html.Div([
                        html.Span(f"Node {target_node_idx}", className='legend-title'),
                        html.Span("correct" if is_correct else "misclassified",
                                  className='pill', style={'backgroundColor': CORRECT_COLOR if is_correct else INCORRECT_COLOR})
                        if actual_class is not None else None,
                    ], className='legend-row'),
                    html.Div([html.Span("Predicted", className='legend-label'),
                              html.Span(class_name(dataset_name, pred_cls) + conf_text)], className='legend-row'),
                    html.Div([html.Span("Actual", className='legend-label'),
                              html.Span(class_name(dataset_name, actual_class) if actual_class is not None else 'N/A')],
                             className='legend-row'),
                    color_key,
                ], className='legend-card')
            else:
                legend_content = html.Div([
                    html.Div("No node selected", className='legend-title'),
                    html.Div("Click a point in the embedding plot or the full graph, or type a node ID "
                             "under Selection, then press RUN ANALYSIS to explain it.", className='empty-hint'),
                ], className='legend-card')

            # Standard Graph Rendering
            graph_elements, _ = create_subgraph_cytoscape(
                pyg_data, target_node_idx, preds, class_to_color_map,
                explanation, cpa_data, None, analysis_mode, selected_path_idx,
                color_mode=subgraph_color_mode,
                true_labels=yvals
            )

            # Importance Plot - only when a node is selected
            if target_node_idx == -1:
                neighbor_fig = empty_figure("Select a node to list its neighbours")
            else:
                node_imp = None
                if explanation and isinstance(explanation, dict) and 'node_importance' in explanation:
                    # Store round-trips through JSON, so keys arrive as strings
                    node_imp = {int(k): float(v) for k, v in (explanation.get('node_importance') or {}).items()}

                if not node_imp or len(node_imp) == 0:
                    try:
                        node_imp = compute_neighbor_importance_from_pyg(pyg_data, target_node_idx, hops=2)
                    except Exception:
                        node_imp = {}

                one_hop_neighbors = get_one_hop_neighbors(pyg_data, target_node_idx)
                if node_imp and len(node_imp) > 0 and one_hop_neighbors:
                    neighbor_imp = {n: node_imp.get(n, 0.0) for n in one_hop_neighbors}
                    neighbor_fig = plot_neighbor_importance(
                        neighbor_imp,
                        target_node_idx,
                        preds=preds,
                        y_labels=yvals,
                        one_hop_neighbors=one_hop_neighbors,
                        dataset_name=dataset_name,
                    )
                else:
                    neighbor_fig = plot_neighbor_importance(
                        {},
                        target_node_idx,
                        preds=preds,
                        y_labels=yvals,
                        one_hop_neighbors=one_hop_neighbors,
                        dataset_name=dataset_name,
                    ) if one_hop_neighbors else empty_figure(f"Node {target_node_idx} has no neighbours")

            # CPA Panel is shown in the merged Explanation frame.
            cpa_panel = create_cpa_iv_panel(cpa_data, target_node_idx) if cpa_data else None
            # Feature importance (from explanation if present)
            feature_fig = empty_fig
            feature_imp = None
            if explanation and isinstance(explanation, dict) and 'feature_importance' in explanation:
                feature_imp = explanation.get('feature_importance') or {}

            if feature_imp and len(feature_imp) > 0:
                try:
                    feature_fig = plot_feature_importance(feature_imp, target_node_idx)
                except Exception:
                    feature_fig = empty_fig
            else:
                # Fallback: show top-5 features from the node's raw feature vector
                try:
                    if target_node_idx != -1 and hasattr(pyg_data, 'x') and pyg_data.x is not None:
                        node_feat = pyg_data.x[target_node_idx]
                        # convert to numpy
                        if hasattr(node_feat, 'detach'):
                            arr = node_feat.detach().cpu().numpy()
                        else:
                            arr = np.array(node_feat)
                        if arr.size > 0:
                            # use absolute magnitude to pick top features
                            idxs = np.argsort(np.abs(arr))[-10:][::-1]
                            feat_dict = {int(i): float(arr[int(i)]) for i in idxs}
                            # Not an explanation yet: label honestly as raw input values
                            feature_fig = plot_feature_importance(
                                feat_dict, target_node_idx, top_k=10, value_label='Raw value',
                                title=f'Raw feature values, node {target_node_idx}')
                        else:
                            feature_fig = empty_fig
                    else:
                        feature_fig = empty_fig
                except Exception:
                    feature_fig = empty_fig

            causal_content = None
            attention_content = None
            causal_style = {'display': 'none'}
            if cpa_data:
                causal_content = create_cpa_iv_panel(cpa_data, target_node_idx)
            if model_capabilities(model_type)['attention']:
                attention_content = create_attention_panel(
                    get_attention_weights(package_meta),
                    target_node_idx if target_node_idx != -1 else 'selected node'
                )
            if causal_content or attention_content:
                causal_content = explanation_cards(causal_content, attention_content)
                causal_style = {'display': 'block', 'marginTop': '16px'}
            return graph_elements, neighbor_fig, feature_fig, legend_content, header_text, causal_content, causal_style

    except Exception as e:
        print(f"Error in update_bottom_layer: {e}")
        import traceback
        traceback.print_exc()
        return [], go.Figure(), go.Figure(), "Error", "Error", None, {'display': 'none'}

# 3. Update Link Prediction List (Top Layer)

@callback(

    Output('link-prediction-output-active', 'children'),

    Output('link-confidence-plot-active', 'figure'),

    Output('link-confidence-plot-active', 'style'),

    Input('link-prediction-store', 'data')

)

def update_link_list(link_predictions):
    """The 'top predicted links for the source node' list is not shown: links
    are opened from the confusion-matrix cell list instead (click a cell, then
    a row). Without a source node the list defaulted to node 0, which was noise."""
    return html.Div([], style={'minHeight': '0'}), go.Figure(), {'display': 'none', 'height': '0px'}

# Subgraph evaluation callback

@callback(

    Output('subgraph-eval-status', 'children'),

    Output('confusion-matrix-plot', 'figure', allow_duplicate=True),

    Output('edge-predictions-display', 'children', allow_duplicate=True),

    Output('performance-metrics', 'children'),
    Output('performance-metrics', 'style'),
    Output('subgraph-eval-store', 'data'),
    Output('global-link-analysis-title', 'children', allow_duplicate=True),

    Input('link-prediction-store', 'data'),
    Input('analysis-mode-store', 'data'),
    Input('precomputed-package-store', 'data'),
    State('full-dataset-store', 'data'),
    State('model-dropdown', 'value'),
    State('dataset-dropdown', 'value'),
    State('graph-type-store', 'data'),
    prevent_initial_call=True
)
def evaluate_subgraph_predictions(link_pred_data, analysis_mode, package, dataset, model_type, dataset_name, graph_type):
    """Link-view confusion matrix. It is a global evaluation of the model on the
    held-out edges (independent of the typed or selected link), so it is built
    when Link view is entered or the model changes, and cached per
    (model, dataset): RUN and link selections reuse it instantly instead of
    waiting behind other model work."""
    if not package or not dataset or analysis_mode != 'link_prediction':
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update
    key = (model_type, dataset_name)
    result = _LINK_EVAL_CACHE.get(key)
    if result is None:
        result = _evaluate_link_predictions(link_pred_data, analysis_mode, package, dataset,
                                            model_type, dataset_name, graph_type)
        if isinstance(result[0], str) and result[0].startswith('✅'):
            _LINK_EVAL_CACHE[key] = result
    return result


_LINK_EVAL_CACHE = {}   # (model_type, dataset_name) -> evaluate_subgraph_predictions outputs
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


@serialize_model_use
def _evaluate_link_predictions(link_pred_data, analysis_mode, package, dataset, model_type, dataset_name, graph_type):

    """Evaluate edge predictions on subgraph and show confusion matrix."""

    try:
        # DEBUG: log evaluation entry and presence of inputs
        try:
            with open('link_debug.log', 'a') as _log:
                _log.write(f"EVAL_SUBGRAPH link_pred_data_present={bool(link_pred_data)} analysis_mode={analysis_mode} package_present={bool(package)} dataset_present={bool(dataset)}\n")
        except Exception:
            pass

        if not package or not dataset or analysis_mode != 'link_prediction':
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update

        # Evaluate on the models' OWN held-out test split. train_models.py builds
        # it with create_edge_splits(..., random_state=42), so it is reproducible
        # here; scoring uses the training graph (test edges removed), exactly as
        # in training. (The old subgraph_models/ edge list came from a different
        # split: ~80% of its "held-out" edges were training edges of the current
        # models, and they were also left in the graph while being scored.)
        try:
            import numpy as np
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            model = get_cached_model(package, dataset.get('num_features'), dataset.get('num_classes'))
            if model is None or not getattr(model, 'enable_link_prediction', False):
                # never show simulated numbers as if they were an evaluation
                return ("❌ This model has no link-prediction head to evaluate", go.Figure(),
                        [cell_details_placeholder('link_prediction')], no_update, {'display': 'none'}, no_update, no_update)

            split = link_test_split(dataset_name)
            positives = [tuple(e) for e in split['test_pos']]
            negatives = [tuple(e) for e in split['test_neg']]
            removed_edges = positives   # held-out positives (name used below)
            full = get_cached_dataset({'name': dataset_name})
            with torch.no_grad():
                _, emb = model(full['x'].to(device), split['train_edge_index'].to(device), return_embeddings=True)
                pairs = torch.tensor(positives + negatives, dtype=torch.long).t().to(emb.device)
                scores = torch.sigmoid(model.predict_links(emb, pairs).reshape(-1)).cpu().tolist()

            pos_data, neg_data = [], []
            for i, ((u, v), score) in enumerate(zip(positives + negatives, scores)):
                rec = {'u': u, 'v': v, 'score': float(score), 'pred': 1 if score > 0.5 else 0,
                       'true': 1 if i < len(positives) else 0}
                (pos_data if i < len(positives) else neg_data).append(rec)

            all_data = pos_data + neg_data

            true_labels = [d['true'] for d in all_data]

            predicted_labels = [d['pred'] for d in all_data]

            # --- CALCULATE METRICS ---

            accuracy = accuracy_score(true_labels, predicted_labels)

            precision = precision_score(true_labels, predicted_labels, zero_division=0)

            recall = recall_score(true_labels, predicted_labels, zero_division=0)

            f1 = f1_score(true_labels, predicted_labels, zero_division=0)

            # --- PREPARE HOVER DATA (Buckets) ---

            tp_list = []

            fp_list = []

            fn_list = []

            tn_list = []

            for d in all_data:

                item = (d['score'], d['u'], d['v'])

                if d['true'] == 1 and d['pred'] == 1:

                    tp_list.append(item)

                elif d['true'] == 0 and d['pred'] == 1:

                    fp_list.append(item)

                elif d['true'] == 1 and d['pred'] == 0:

                    fn_list.append(item)

                elif d['true'] == 0 and d['pred'] == 0:

                    tn_list.append(item)

            # Sort Top K

            # TP: Highest score first

            tp_list.sort(key=lambda x: x[0], reverse=True)

            # FP: Highest score first

            fp_list.sort(key=lambda x: x[0], reverse=True)

            # FN: Lowest score first (closest to 0)

            fn_list.sort(key=lambda x: x[0])

            # TN: Lowest score first

            tn_list.sort(key=lambda x: x[0])

            fig = link_confusion_figure(true_labels, predicted_labels, tp_list, fp_list, fn_list, tn_list)

            # Leave the cell-details panel alone: it is filled only when a matrix
            # cell is clicked (update_edge_list_from_cm) and cleared by
            # reset_cell_details. This callback also re-runs when a link is picked,
            # which would otherwise wipe the list the user just opened.
            edge_predictions = no_update

            # Performance metrics on the held-out (removed) edges
            metrics = metric_tiles([
                ("Accuracy", float(accuracy), "Link accuracy on the removed (held-out) edges and sampled non-edges"),
                ("Precision", float(precision), "Of predicted links, the share that are real"),
                ("Recall", float(recall), "Of real removed edges, the share recovered"),
                ("F1", float(f1), "Harmonic mean of precision and recall"),
            ])

            status_msg = f"✅ Evaluated {len(removed_edges)} held-out test edges"

            perf_style = {'display': 'flex'}

            return status_msg, fig, edge_predictions, metrics, perf_style, {'TP': tp_list, 'FP': fp_list, 'FN': fn_list, 'TN': tn_list}, "Confusion Matrix Details"

        except Exception as e:

            return f"❌ Error loading subgraph model: {str(e)}", go.Figure(), no_update, no_update, no_update, no_update, no_update

    except Exception as e:

        print(f"Subgraph evaluation error: {e}")

        return f"❌ Evaluation failed: {str(e)}", go.Figure(), no_update, no_update, no_update, no_update, no_update





@callback(
    Output('edge-predictions-display', 'children', allow_duplicate=True),
    Output('global-link-analysis-title', 'children', allow_duplicate=True),
    Output('cm-detail-paging-store', 'data', allow_duplicate=True),
    Input('confusion-matrix-plot', 'clickData'),
    State('subgraph-eval-store', 'data'),
    State('analysis-mode-store', 'data'),
    State('current-predictions-store', 'data'),
    State('full-dataset-store', 'data'),
    prevent_initial_call=True
)
def update_edge_list_from_cm(clickData, eval_data, analysis_mode, predictions_data, dataset):
    """
    Handles clicks on either confusion matrix variant:

    NODE CLASSIFICATION  (multi-class CM, x/y = class indices as strings):
        Shows the list of nodes that fall in the clicked (actual_class, pred_class) cell.
        Diagonal cells = correctly classified; off-diagonal = misclassified.

    LINK PREDICTION  (binary CM, x/y = "Predicted: Edge" / "Actual: No Edge" etc.):
        Maps the cell to TP / FP / FN / TN and shows the corresponding edge list
        from subgraph-eval-store, sorted by score.

        Layout of the link-prediction confusion matrix (y = Actual, x = Predicted):

        The plot uses sklearn's confusion_matrix layout and the heatmap axes are:
            x = ['Predicted: No Edge', 'Predicted: Edge']
            y = ['Actual: No Edge',    'Actual: Edge']

        Corresponding cell mapping (sklearn 2x2 matrix):
            [0,0] = TN (Actual: No Edge, Predicted: No Edge)
            [0,1] = FP (Actual: No Edge, Predicted: Edge)
            [1,0] = FN (Actual: Edge,    Predicted: No Edge)
            [1,1] = TP (Actual: Edge,    Predicted: Edge)
    """
    if not clickData:
        return no_update, no_update, no_update

    print(f"[CM Click] clickData received: {clickData}")
    print(f"[CM Click] analysis_mode={analysis_mode}")

    empty_paging = {'rows': [], 'columns': [], 'badge_color': '', 'footer_text': '', 'page': 0, 'page_count': 1}
    try:
        point = clickData['points'][0]

        # Axis values are numeric class indices: 0 is valid, so test for None, not truthiness
        def _first(*keys):
            for k in keys:
                v = point.get(k)
                if v is not None and v != '':
                    return str(v)
            return ''
        x_val = _first('x', 'xaxis', 'xlabel')
        y_val = _first('y', 'yaxis', 'ylabel')
        if not x_val or not y_val:
            print(f"[CM Click] Point keys: {list(point.keys())}")
            return (
                html.Div(
                    'Clicked cell information could not be interpreted. Try clicking a different cell.',
                    style={'color': '#999', 'fontStyle': 'italic', 'fontSize': '12px', 'padding': '10px', 'textAlign': 'center'}
                ),
                'Confusion Matrix Details',
                empty_paging,
            )

        print(f"[CM Click] x_val (predicted)='{x_val}'  y_val (actual)='{y_val}'")

        PANEL_TITLE = 'Confusion Matrix Details'

        # Ensure node-classification uses the full cached dataset so y labels are available.
        if analysis_mode == 'node_classification' and dataset and isinstance(dataset, dict) and 'name' in dataset:
            dataset = get_cached_dataset_if_missing(dataset)

        def _badge_div(label, count, unit, badge_color, description):
            return html.Div(style={'marginBottom': '8px'}, children=[
                html.Span(
                    f'{label}  ({count} {unit})',
                    style={
                        'backgroundColor': badge_color, 'color': 'white',
                        'padding': '3px 10px', 'borderRadius': '12px',
                        'fontSize': '12px', 'fontWeight': 'bold', 'marginRight': '8px',
                    }
                ),
                html.Span(description, style={'fontSize': '11px', 'color': '#666'}),
            ])

        empty_paging = {
            'rows': [],
            'columns': [],
            'badge_color': '',
            'footer_text': '',
            'page': 0,
            'page_count': 1,
        }

        # ─── NODE CLASSIFICATION ─────────────────────────────────────────
        if analysis_mode == 'node_classification':
            print(f"[CM Click] Entering node_classification branch")
            if not predictions_data or not dataset:
                return (
                    html.Div(
                        'No prediction data available. Run analysis first.',
                        style={'color': '#999', 'fontStyle': 'italic',
                               'fontSize': '12px', 'padding': '10px', 'textAlign': 'center'}
                    ),
                    PANEL_TITLE,
                    empty_paging,
                )

            try:
                pred_class   = int(x_val)
                actual_class = int(y_val)
            except (ValueError, TypeError):
                return no_update, no_update, no_update

            preds       = predictions_data.get('preds', [])
            true_labels = dataset.get('y', [])
            if len(preds) == 0 or len(true_labels) == 0:
                print(f"[CM Click] Missing preds/true_labels for node classification: len(preds)={len(preds)}, len(true_labels)={len(true_labels)}")
                return (
                    html.Div(
                        'No prediction or label data available for node classification. Reload the dataset and try again.',
                        style={'color': '#999', 'fontStyle': 'italic', 'fontSize': '12px', 'padding': '10px', 'textAlign': 'center'}
                    ),
                    PANEL_TITLE,
                    empty_paging,
                )

            matching = [
                node_id
                for node_id, (p, t) in enumerate(zip(preds, true_labels))
                if int(p) == pred_class and int(t) == actual_class
            ]
            count = len(matching)
            print(f"[CM Click] Node class cell ({actual_class}→{pred_class}): {count} nodes found")

            is_diagonal = (pred_class == actual_class)
            ds_name = dataset.get('name') if isinstance(dataset, dict) else None
            pred_name, actual_name = class_label(ds_name, pred_class), class_label(ds_name, actual_class)
            if is_diagonal:
                badge_color = CORRECT_COLOR
                label       = f'Correctly classified as {pred_name}'
                description = (f'{count} node{"s" if count != 1 else ""} of class '
                               f'{actual_name} predicted correctly')
            else:
                badge_color = INCORRECT_COLOR
                label       = f'{actual_name} → predicted {pred_name}'
                description = (f'{count} node{"s" if count != 1 else ""} of class '
                               f'{actual_name} wrongly predicted as {pred_name}')

            header = _badge_div(label, count, 'nodes', badge_color, description)

            if count == 0:
                return (
                    html.Div([
                        header,
                        html.Div('No nodes in this cell.',
                                 style={'color': '#999', 'fontStyle': 'italic',
                                        'fontSize': '13px', 'padding': '10px', 'textAlign': 'center'}),
                    ]),
                    PANEL_TITLE,
                    empty_paging,
                )

            # Include confidence if available in predictions_data
            confidences = predictions_data.get('confidences') if predictions_data else None
            rows = []
            for i, nid in enumerate(matching):
                row = {'#': i + 1, 'Node ID': nid, 'True Class': actual_class, 'Predicted Class': pred_class}
                if confidences and len(confidences) > nid:
                    # store as percentage (0-100) with one decimal
                    row['Confidence'] = round(float(confidences[nid]) * 100.0, 1)
                else:
                    row['Confidence'] = None
                rows.append(row)

            columns = [
                {'name': '#',               'id': '#',               'type': 'numeric'},
                {'name': 'Node ID',          'id': 'Node ID',         'type': 'numeric'},
                # Short headers: this panel is narrow
                {'name': 'True',             'id': 'True Class',      'type': 'numeric'},
                {'name': 'Pred.',            'id': 'Predicted Class', 'type': 'numeric'},
                {'name': 'Conf. %',          'id': 'Confidence',      'type': 'numeric'},
            ]
            footer_text = f'Showing {count} node{"s" if count != 1 else ""}'
            page_children, page_data = _render_cm_detail_page(rows, columns, badge_color, footer_text, 0)
            return html.Div([header, page_children], style={'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'minHeight': 0}), PANEL_TITLE, page_data

        # ─── LINK PREDICTION ─────────────────────────────────────────────
        print(f"[CM Click] Entering link_prediction branch. eval_data present: {bool(eval_data)}")
        if not eval_data or not isinstance(eval_data, dict):
            return (
                html.Div(
                    'No edge evaluation data available. Ensure a subgraph model is loaded.',
                    style={'color': '#999', 'fontStyle': 'italic',
                           'fontSize': '12px', 'padding': '10px', 'textAlign': 'center'}
                ),
                PANEL_TITLE,
                empty_paging,
            )

        # True if "Edge" appears but NOT "No Edge" (avoids substring collision)
        pred_is_edge   = 'Edge' in x_val and 'No Edge' not in x_val
        actual_is_edge = 'Edge' in y_val and 'No Edge' not in y_val

        if actual_is_edge and pred_is_edge:
            category    = 'TP';  label = 'True Positives'
            badge_color = '#27ae60'
            description = 'Edges that exist AND were correctly predicted'
        elif not actual_is_edge and pred_is_edge:
            category    = 'FP';  label = 'False Positives'
            badge_color = '#e74c3c'
            description = 'Non-edges that were wrongly predicted as edges'
        elif actual_is_edge and not pred_is_edge:
            category    = 'FN';  label = 'False Negatives'
            badge_color = '#e67e22'
            description = 'Real edges that the model missed'
        else:
            category    = 'TN';  label = 'True Negatives'
            badge_color = '#2980b9'
            description = 'Non-edges correctly predicted as absent'

        # eval_data stores lists of [score, u, v] (JSON-serialised tuples)
        edge_list = eval_data.get(category, [])
        count     = len(edge_list)
        print(f"[CM Click] Link prediction category='{category}' ({label}): {count} edges")

        header = _badge_div(label, count, 'edges', badge_color, description)

        if count == 0:
            return (
                html.Div([
                    header,
                    html.Div(f'No edges found for {label}.',
                             style={'color': '#999', 'fontStyle': 'italic',
                                    'fontSize': '13px', 'padding': '10px', 'textAlign': 'center'}),
                ]),
                label,
                empty_paging,
            )

        MAX_ROWS = 200
        rows = [
            {
                '#':           rank,
                'Source Node': int(item[1]),
                'Target Node': int(item[2]),
                'Confidence':  round(float(item[0]), 4),
            }
            for rank, item in enumerate(edge_list[:MAX_ROWS], start=1)
        ]
        columns = [
            {'name': '#',           'id': '#',           'type': 'numeric'},
            {'name': 'Source Node', 'id': 'Source Node', 'type': 'numeric'},
            {'name': 'Target Node', 'id': 'Target Node', 'type': 'numeric'},
            {'name': 'Confidence',  'id': 'Confidence',  'type': 'numeric'},
        ]
        footer_text = f'Showing top {min(count, MAX_ROWS)} of {count} edges  ·  sorted by confidence descending'
        page_children, page_data = _render_cm_detail_page(rows, columns, badge_color, footer_text, 0)
        return html.Div([header, page_children], style={'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'minHeight': 0}), label, page_data

    except Exception as e:
        print(f"Error in update_edge_list_from_cm: {e}")
        import traceback
        traceback.print_exc()
        return no_update, no_update, no_update


@callback(
    Output('edge-predictions-display', 'children', allow_duplicate=True),
    Output('cm-detail-paging-store', 'data', allow_duplicate=True),
    Input('cm-page-prev', 'n_clicks'),
    Input('cm-page-next', 'n_clicks'),
    State('cm-detail-paging-store', 'data'),
    prevent_initial_call=True
)
def update_cm_detail_page(prev_clicks, next_clicks, paging_data):
    if not paging_data or 'rows' not in paging_data:
        return no_update, no_update

    triggered_id = ctx.triggered_id
    if triggered_id not in {'cm-page-prev', 'cm-page-next'}:
        return no_update, no_update

    current_page = paging_data.get('page', 0)
    page_count = paging_data.get('page_count', 1)

    if triggered_id == 'cm-page-prev':
        new_page = max(0, current_page - 1)
    else:
        new_page = min(page_count - 1, current_page + 1)

    if new_page == current_page:
        return no_update, no_update

    page_children, updated_paging = _render_cm_detail_page(
        paging_data['rows'],
        paging_data['columns'],
        paging_data['badge_color'],
        paging_data['footer_text'],
        new_page,
    )

    return page_children, updated_paging


# Allow selecting a node by clicking a cell in the CM detail DataTable
@callback(
    Output('selected-node-store', 'data', allow_duplicate=True),
    Input('cm-detail-table', 'active_cell'),
    State('cm-detail-table', 'data'),
    prevent_initial_call=True,
)
def select_node_from_detail_table(active_cell, table_data):
    if not active_cell or not table_data:
        return no_update

    row_idx = active_cell.get('row')
    col_id = active_cell.get('column_id')
    try:
        row = table_data[row_idx]
    except Exception:
        return no_update

    # Prefer explicit node id columns
    for candidate in ('Node ID', 'Source Node', 'Target Node', 'Source', 'Target'):
        if candidate in row and row.get(candidate) is not None:
            val = str(row.get(candidate))
            print(f"[SELECTED_NODE] select_node_from_detail_table choosing {candidate}={val}")
            return {'id': val}

    # Fallback: if clicked column contains a numeric-looking value
    val = row.get(col_id)
    if val is not None:
        print(f"[SELECTED_NODE] select_node_from_detail_table fallback value={val}")
        return {'id': str(val)}

    return no_update


# If the details table row represents a link (has both source and target), select that link
@callback(
    Output('selected-link-store', 'data', allow_duplicate=True),
    Input('cm-detail-table', 'active_cell'),
    State('cm-detail-table', 'data'),
    prevent_initial_call=True,
)
def select_link_from_detail_table(active_cell, table_data):
    if not active_cell or not table_data:
        return no_update

    row_idx = active_cell.get('row')
    try:
        row = table_data[row_idx]
    except Exception:
        return no_update

    # Recognize common link column names
    src_keys = ['Source Node', 'Source', 'Source_ID']
    tgt_keys = ['Target Node', 'Target', 'Target_ID']
    src = None
    tgt = None
    for k in src_keys:
        if k in row and row.get(k) is not None:
            src = row.get(k)
            break
    for k in tgt_keys:
        if k in row and row.get(k) is not None:
            tgt = row.get(k)
            break

    if src is not None and tgt is not None:
        try:
            return {'source': int(src), 'target': int(tgt)}
        except Exception:
            return {'source': str(src), 'target': str(tgt)}

    return no_update


# Reset application callback

@callback(

    Output('url', 'href'),

    Input('reset-button', 'n_clicks'),

    prevent_initial_call=True

)

def reset_application(n_clicks):

    """Reload the application to reset state."""

    if n_clicks:

        return "/" 

    return no_update

@callback(
    Output('layer-2', 'style'),
    Output('layer-3', 'style'),
    Input('global-run-button', 'n_clicks'),
    Input('selected-link-store', 'data'),
    Input('selected-node-store', 'data'),
    Input('analysis-mode-store', 'data'),
    Input('reset-button', 'n_clicks'),
    Input('current-predictions-store', 'data'),
    State('layer3-tabs', 'value'),
    prevent_initial_call=False
)

def manage_layer_visibility(run_clicks, selected_link, selected_node, analysis_mode, reset_clicks, predictions_data, active_layer3_tab):
    """
    3-LAYER SEQUENTIAL FLOW:
      Step 1 — RUN clicked  : Layer 2 visible, Layer 3 hidden.
      Step 2 — Node or link selected in Layer 2 : Layer 3 becomes visible.

    Layer 3 is intentionally withheld until the user makes an explicit selection
    so the UI guides them through the workflow naturally.
    """
    l2_hidden  = {'display': 'none', 'borderTop': '2px solid #dee2e6', 'paddingTop': '15px', 'marginTop': '5px'}
    l3_hidden  = {'display': 'none', 'borderTop': '2px solid #dee2e6', 'paddingTop': '15px', 'marginTop': '10px'}
    l2_visible = {'display': 'block', 'borderTop': '2px solid #dee2e6', 'paddingTop': '15px', 'marginTop': '10px', 'height': '49.5vh', 'maxHeight': '49.5vh', 'overflow': 'hidden', 'boxSizing': 'border-box'}
    l3_visible = {'display': 'block', 'borderTop': '2px solid #dee2e6', 'paddingTop': '15px', 'marginTop': '10px', 'minHeight': '48vh', 'overflow': 'hidden'}

    # Show the overview as soon as a model is loaded; RUN is only needed to explain.
    if not predictions_data and not run_clicks:
        return l2_hidden, l3_hidden

    return l2_visible, l3_visible

# --- Client-side Callbacks ---

app.clientside_callback(

    """

    function(n_clicks) {

        if (n_clicks > 0) {

            return ["Processing...", true];

        }

        return [window.dash_clientside.no_update, window.dash_clientside.no_update];

    }

    """,

    Output('global-run-button', 'children'),

    Output('global-run-button', 'disabled'),

    Input('global-run-button', 'n_clicks')

)

# Immediate feedback for stability runs: the server reply replaces this text.
# Cached (sigma, model, dataset) combinations return at once; anything else
# runs GNNExplainer twice per node and can take minutes.
app.clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) { return window.dash_clientside.no_update; }
        return "⏳ Running… instant if precomputed; otherwise GNNExplainer runs per node (or on a 100-edge sample for links) and this can take a few minutes.";
    }
    """,
    Output('stability-status', 'children', allow_duplicate=True),
    Input('run-stability-button', 'n_clicks'),
    prevent_initial_call=True,
)

# --- Run Application ---






# Legacy callbacks removed to prevent duplication
# Causal Analysis is now handled by update_mid_layer and create_cpa_iv_panel

# --- Enhanced Link CPA-IV Callbacks ---









# Legacy link CPA summary callbacks removed.
# Causal explanation is now rendered directly inside the merged Explanation frame.

# --- Stability Analysis Callbacks ---

@callback(
    Output('stability-results-store', 'data'),
    Output('stability-status', 'children'),
    Input('run-stability-button', 'n_clicks'),
    State('stability-sigma-selector', 'value'),
    State('stability-sample-size', 'value'),
    State('precomputed-package-store', 'data'),
    State('full-dataset-store', 'data'),
    State('editable-graph-store', 'data'),
    State('current-predictions-store', 'data'),
    State('model-dropdown', 'value'),
    State('analysis-mode-store', 'data'),
    prevent_initial_call=True
)
def run_stability_callback(n_clicks, sigma, sample_size,
                            package, dataset, graph_data, predictions_data, model_type,
                            analysis_mode):
    """
    Global stability analysis — GNNExplainer Top-k Jaccard (paper-defined).
    Runs on every node (or a random sample when sample_size > 0).
    Independent of any selected node.
    """
    print(f"\n[STABILITY] run_stability_callback fired: n_clicks={n_clicks}")

    # Stability analysis is supported for GCN only; clear any earlier (GCN)
    # results so they are not shown under a GAT label.
    if n_clicks and not model_capabilities(model_type)['stability']:
        return [], stability_unsupported_message(model_type)

    if not n_clicks or not package or not dataset or not predictions_data:
        print("[STABILITY] Guard failed — missing data")
        return no_update, "Missing data — load a model first."

    try:
        sigma       = float(sigma)     if sigma       is not None else 0.05
        sample_size = int(sample_size) if sample_size is not None else 0  # 0 = ALL nodes
        print(f"[STABILITY] sigma={sigma}, sample_size={sample_size}, model={model_type}")

        # ── Cache check (only cache full-graph runs; sampled runs are not cached) ──
        dataset_name = dataset.get('name', 'unknown')
        # Only cache full-graph results on the *unedited* graph: the cache key has no
        # notion of user edge edits, so an edited graph must always be recomputed.
        _graph_edited = bool(graph_data) and graph_data.get('edge_index') != dataset.get('original_edge_index')
        _use_cache   = (sample_size == 0) and not _graph_edited
        if _graph_edited:
            print("[STABILITY] Graph has been edited — bypassing stability cache")

        # If analysis mode indicates link prediction, run link-stability branch
        mode = (analysis_mode or '').lower()
        if 'link' in mode or 'link_prediction' in mode:
            _cache_key = get_link_stability_cache_key(dataset_name, model_type or 'gcn', sigma)
            if _use_cache:
                _cached = load_link_stability_cache(_cache_key)
                if _cached is not None:
                    print(f"[LINK STABILITY] Loading from cache (key={_cache_key})")
                    safe_results = _cached
                    import math
                    def _safe(v):
                        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                            return None
                        return v
                    safe_results = [{k: _safe(v) for k, v in r.items()} for r in safe_results]
                    n_edges = len(safe_results)
                    avg_stab = sum(r.get('stability', 0) or 0 for r in safe_results) / max(1, n_edges)
                    scope_label = f"FULL GRAPH ({n_edges} edges) [from cache]"
                    status = (f"⚡ {scope_label} · σ={sigma} · avg S = {avg_stab:.3f}")
                    return safe_results, status
            else:
                print(f"[LINK STABILITY] Sampled run — skipping cache lookup")

            model = private_model(package, dataset['num_features'], dataset.get('num_classes', 2))
            if model is None:
                return [], "❌ Could not load model."

            pyg_data = prepare_torch_data(dataset, graph_data, include_labels=False)
            if pyg_data is None:
                return [], "❌ Could not construct graph data."

            ei = pyg_data.edge_index.cpu()
            n_total = int(torch.unique(torch.minimum(ei[0], ei[1]) * pyg_data.num_nodes
                                       + torch.maximum(ei[0], ei[1])).numel())
            live_sample = sample_size or LINK_STABILITY_LIVE_SAMPLE
            print(f"[LINK STABILITY] Computing from scratch (key={_cache_key}), "
                  f"{min(live_sample, n_total)} of {n_total} edges")
            results = run_link_stability_analysis(model, pyg_data, sigma, live_sample)
            print(f"[LINK STABILITY] raw results: {len(results)} entries")

            if not results:
                return [], "⚠️ No results — check console logs."

            # sanitize
            import math
            def _safe(v):
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    return None
                return v
            safe_results = [{k: _safe(v) for k, v in r.items()} for r in results]

            n_edges = len(safe_results)
            is_full = live_sample >= n_total
            # persist only complete runs, so the cache always means "full graph"
            if _use_cache and is_full:
                save_link_stability_cache(_cache_key, safe_results)

            avg_stab = sum(r.get('stability', 0) or 0 for r in safe_results) / max(1, n_edges)
            if is_full:
                status = f"✅ FULL GRAPH ({n_edges} edges) · σ={sigma} · avg S = {avg_stab:.3f}"
            else:
                status = (f"🔹 SAMPLE of {n_edges} / {n_total} edges · σ={sigma} · avg S = {avg_stab:.3f} · "
                          f"no precomputed file; run precompute_link_stability.py for the full graph")
            return safe_results, status

        # Default: node stability branch
        _cache_key   = get_stability_cache_key(dataset_name, model_type or 'gcn', sigma)
        if _use_cache:
            _cached = load_stability_cache(_cache_key)
            if _cached is not None:
                print(f"[STABILITY] Loading from cache (key={_cache_key})")
                safe_results = _cached
                import math
                def _safe(v):
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        return None
                    return v
                safe_results = [{k: _safe(v) for k, v in r.items()} for r in safe_results]
                n_correct = sum(1 for r in safe_results if r.get('correct'))
                n_wrong   = len(safe_results) - n_correct
                avg_stab  = sum(r.get('stability', 0) or 0 for r in safe_results) / len(safe_results)
                scope_label = f"FULL GRAPH ({len(safe_results)} nodes) [from cache]"
                status = (f"⚡ {scope_label} · σ={sigma} · "
                          f"avg S = {avg_stab:.3f} · correct={n_correct} / wrong={n_wrong}")
                return safe_results, status
        else:
            print(f"[STABILITY] Sampled run — skipping cache lookup")

        model = private_model(package, dataset['num_features'], dataset['num_classes'])
        if model is None:
            return [], "❌ Could not load model."

        pyg_data = prepare_torch_data(dataset, graph_data, include_labels=True)
        if pyg_data is None:
            return [], "❌ Could not construct graph data."
        preds_len = len(predictions_data.get('preds', []))
        print(f"[STABILITY] x={pyg_data.x.shape}, edge_index={pyg_data.edge_index.shape}, preds={preds_len}")
        print(f"[STABILITY] Computing from scratch (key={_cache_key})")

        results = run_stability_analysis(model, pyg_data, predictions_data, sigma, sample_size)
        print(f"[STABILITY] raw results: {len(results)} entries")

        if not results:
            return [], "⚠️ No results — check console logs."

        # ── Sanitize NaN/Inf → None so JSON serialization works ────────────
        import math
        def _safe(v):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v

        safe_results = [{k: _safe(v) for k, v in r.items()} for r in results]

        import json as _json
        try:
            _json.dumps(safe_results)
        except (TypeError, ValueError) as je:
            print(f"[STABILITY] JSON validation failed: {je}")
            return [], f"❌ Serialization error: {je}"

        # ── Persist to cache (full-graph runs only) ────────────────────────
        if _use_cache:
            save_stability_cache(_cache_key, safe_results)

        n_correct = sum(1 for r in safe_results if r.get('correct'))
        n_wrong   = len(safe_results) - n_correct
        avg_stab  = sum(r.get('stability', 0) or 0 for r in safe_results) / len(safe_results)
        print(f"[STABILITY] correct={n_correct}, wrong={n_wrong}, avg_stability={avg_stab:.3f}")
        print(f"[STABILITY] first result: {safe_results[0]}")

        is_full = (sample_size == 0 or len(safe_results) == dataset['num_nodes'])
        scope_label = f"FULL GRAPH ({len(safe_results)} nodes)" if is_full else f"{len(safe_results)} nodes (sampled)"
        status = (f"✅ {scope_label} · σ={sigma} · "
                  f"avg S = {avg_stab:.3f} · correct={n_correct} / wrong={n_wrong}")
        return safe_results, status

    except Exception as e:
        traceback.print_exc()
        return [], f"❌ Error: {str(e)}"


@callback(
    Output('main-stability-plot', 'figure'),
    Output('stability-stats-panel', 'children'),
    Input('stability-results-store', 'data'),
    Input('stability-plot-type', 'value'),
    Input('stability-metric-selector', 'value'),
)

def render_stability_plots(results, plot_type, metric_type):
    return build_stability_plots(results, plot_type, metric_type)


# ── Drill-down: capture click, update hint text, show/hide right panel ────────

_PANEL_HIDDEN = {'flex': '1', 'minWidth': '0', 'flexDirection': 'column', 'gap': '12px', 'display': 'none'}
_PANEL_SHOWN  = {'flex': '1', 'minWidth': '0', 'flexDirection': 'column', 'gap': '12px', 'display': 'flex'}

@callback(
    Output('drilldown-click-store',   'data'),
    Output('drilldown-hint',          'children'),
    Output('drilldown-right-panel',   'style'),
    Input('main-stability-plot',      'clickData'),
    Input('drilldown-reset-btn',      'n_clicks'),
    State('stability-plot-type',      'value'),
    State('stability-metric-selector','value'),
    State('stability-results-store',  'data'),
    prevent_initial_call=True,
)
def update_drilldown_store(main_click, _reset, plot_type, metric_type, results):
    # Mode-aware hint text shown in the controls bar
    _hints = {
        'hexbin':  'Click a hexbin cell → see individual nodes in that region',
        'kde':     'Click a KDE contour region → see individual nodes in that region',
        'scatter': 'Click a scatter point → see its nearest neighbors',
    }
    plot_type = (plot_type or 'scatter').lower()
    hint = _hints.get(plot_type, 'Click a plot region to drill down into nodes')

    triggered = ctx.triggered_id
    if triggered == 'drilldown-reset-btn':
        return None, hint, _PANEL_HIDDEN

    def _extract_xy(click_payload):
        """Return (x, y) floats from clickData, or (None, None) on failure.
        Handles list-format bin values from go.Histogram2d (takes midpoint)."""
        if not click_payload:
            return None, None
        pt = click_payload['points'][0]
        x_val = pt.get('x')
        y_val = pt.get('y')
        if x_val is None or y_val is None:
            return None, None
        def _to_float(v):
            if isinstance(v, (list, tuple)) and len(v) == 2:
                return float((v[0] + v[1]) / 2)
            return float(v)
        try:
            return _to_float(x_val), _to_float(y_val)
        except (TypeError, ValueError):
            return None, None

    def _snap_to_bin_center(x_val, y_val, x_col):
        """For hexbin mode: snap raw click coords to the nearest bin center."""
        if plot_type != 'hexbin' or not results:
            return x_val, y_val
        try:
            import pandas as _pd
            _df = _pd.DataFrame(results).dropna(subset=[x_col, 'stability'])
            _df[x_col]       = _df[x_col].astype(float)
            _df['stability'] = _df['stability'].astype(float)
            if _df.empty:
                return x_val, y_val
            NBINS = 28
            x_min, x_max = _df[x_col].min(), _df[x_col].max()
            y_min, y_max = _df['stability'].min(), _df['stability'].max()
            bw = (x_max - x_min) / NBINS if x_max > x_min else 1.0
            bh = (y_max - y_min) / NBINS if y_max > y_min else 1.0
            bi = int((x_val - x_min) / bw)
            bj = int((y_val - y_min) / bh)
            bi = max(0, min(NBINS - 1, bi))
            bj = max(0, min(NBINS - 1, bj))
            return x_min + (bi + 0.5) * bw, y_min + (bj + 0.5) * bh
        except Exception:
            return x_val, y_val

    def _kde_density_at(x_val, y_val, x_col):
        """For KDE mode: fit gaussian_kde on all nodes and return density at (x, y)."""
        if plot_type != 'kde' or not results:
            return None
        try:
            import pandas as _pd
            from scipy.stats import gaussian_kde as _gkde
            _df = _pd.DataFrame(results).dropna(subset=[x_col, 'stability'])
            _df[x_col]       = _df[x_col].astype(float)
            _df['stability'] = _df['stability'].astype(float)
            if len(_df) < 5:
                return None
            _x = _df[x_col].values
            _y = _df['stability'].values
            if float(np.std(_x)) < 1e-9 or float(np.std(_y)) < 1e-9:
                return None
            kde_fn = _gkde(np.vstack([_x, _y]), bw_method='silverman')
            return float(kde_fn(np.array([[x_val], [y_val]]))[0])
        except Exception:
            return None

    if triggered == 'main-stability-plot':
        if not main_click:
            return no_update, hint, no_update
        x_val, y_val = _extract_xy(main_click)
        if x_val is None:
            return no_update, hint, no_update

        source_col = metric_type if metric_type in ['confidence', 'degree', 'lipschitz'] else 'confidence'

        x_val, y_val = _snap_to_bin_center(x_val, y_val, source_col)
        clicked_density = _kde_density_at(x_val, y_val, source_col)
        return (
            {'x': x_val, 'y': y_val, 'source': source_col,
             'plot_type': plot_type, 'clicked_density': clicked_density},
            hint,
            _PANEL_SHOWN,
        )

    return no_update, hint, no_update


# ── Drill-down: render zoomed node-level scatter ──────────────────────────────

@callback(
    Output('drilldown-scatter',      'figure'),
    Output('drilldown-panel-header', 'children'),
    Input('drilldown-click-store',    'data'),
    Input('drilldown-epsilon-slider', 'value'),
    Input('stability-plot-type',      'value'),
    State('stability-results-store',  'data'),
    prevent_initial_call=True,
)
def render_drilldown_scatter(click_data, epsilon, plot_type, results):
    return build_drilldown_scatter(click_data, epsilon, plot_type, results)


# ── Highlight selected region on the main stability plots ─────────────────────
# Uses Patch() to inject / clear a rectangle shape without re-rendering the figure.

@callback(
    Output('main-stability-plot',   'figure', allow_duplicate=True),
    Input('drilldown-click-store',     'data'),
    Input('drilldown-epsilon-slider',  'value'),
    prevent_initial_call=True,
)
def highlight_region_on_main_plots(click_data, epsilon):
    return build_region_highlight(click_data, epsilon)



# --- Run Application ---
if __name__ == '__main__':
    print("Dash app starting...")
    _scan_link_stability_cache()
    # Disable the Werkzeug auto-reloader on Windows to avoid select() socket errors
    # (WinError 10038) caused by the reloader thread closing sockets unexpectedly.
    # Bind to localhost by default: debug mode exposes the Werkzeug debugger, which
    # allows code execution. Set GNN_VIS_HOST=0.0.0.0 / GNN_VIS_DEBUG=0 to change.
    _host = os.environ.get('GNN_VIS_HOST', '127.0.0.1')
    _debug = os.environ.get('GNN_VIS_DEBUG', '1') == '1'
    # dev_tools_hot_reload=False: with the code reloader off, hot reload only
    # made pages left open across a server restart reconnect and replay their
    # callbacks against the new server. Their browser-side (clientside)
    # functions are the old ones, so new clientside callbacks failed with
    # "Cannot read properties of undefined (reading 'apply')". Refresh instead.
    app.run(debug=_debug, host=_host, port=8050, use_reloader=False,
            dev_tools_hot_reload=False)

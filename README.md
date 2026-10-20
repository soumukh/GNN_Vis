## Introduction
InteractiveGNNExplainer, a visual analytics framework to enhance GNN explainability.  
Paper published in 29th International Conference Information Visualisation 2025 (https://ieeexplore.ieee.org/abstract/document/11216840)

## Offline Phase
To start first run the following codes:
1. preprocess_dataset.py
2. train_models.py
3. precompute_node_stability.py 
4. precompute_link_stability.py

## Online Phase
Run the dash application using app.py. By default it will start at http://localhost:8050/

The server binds to 127.0.0.1 with debug mode on. To serve on your network, set
`GNN_VIS_HOST=0.0.0.0` and turn debug off with `GNN_VIS_DEBUG=0` (the debugger allows code execution).

Quick check after the offline phase: `python test_smoke.py`

## Code layout
- `gnnvis/models/`: the only definition of the GNN models (`GCNNet`, `GATNet`), the
  model registry, and loading of saved model packages. Training, the precompute scripts
  and the app all use it.
- `gnnvis/ui/`: the view layer. `theme.py` (class names, colours, empty figures),
  `components.py` (tiles, cards, placeholders) and `figures/` (embeddings, confusion
  matrices, subgraph, importance, CPA-IV, attention, full graph, stability): plain
  functions from data to Plotly figures / Dash components, testable without the app.
- `gnnvis/ui/layout/`: the page layout, one module per area (header, controls, overview,
  explanation / full-graph / stability tabs, stores); `build_layout()` assembles it.
- `gnnvis/config.py`: datasets, defaults and the model list (from the registry).
- `gnnvis/tasks.py`: the two tasks (node classification, link prediction) and what differs
  between them; callbacks use `get_task(mode)` instead of comparing mode strings.
- `gnnvis/explainers/`: explainers registered per task and kind (`attribution`:
  GNNExplainer for nodes, a structural heuristic for links; `causal_paths`: CPA-IV).
  The app and both stability analyses build their GNNExplainer here.
- `gnnvis/ui/callbacks/`: the Dash callbacks, one module per feature (session, selection,
  explanation, overview, link, full graph, stability); importing the package registers them.
- `gnnvis/runtime.py`: device, paths, caches of model packages / datasets / models, locks.
- `gnnvis/jobs.py`: background jobs with progress and cancel. A stability run that is not
  precomputed runs as a job; the Stability tab shows its progress and remaining time and
  has a Cancel button.
- `gnnvis/stability_cache.py`: the precomputed stability files in `stability_cache/`. Each
  has a `.meta.json` sidecar with a fingerprint of the model weights, so results computed
  for an older model (e.g. before retraining) are detected as out of date and not shown.
  Files from before the sidecars existed are checked against the current model on first
  use and stamped if they match.
- `gnnvis/ui/ids.py`: every component id as a constant; layout and callbacks use these,
  and `tests/test_ids.py` fails if a callback references a component that never exists.
- `app.py`: the entry point (creates the app, attaches the layout, registers callbacks); `app_stability.py`, `app_expl.py`,
  `cpa_iv_link_prediction.py`, `link_prediction_utils.py`: analysis code.
- `train_models.py`, `precompute_node_stability.py`, `precompute_link_stability.py`: offline scripts.
- `tests/`: `python -m pytest` (a few seconds; synthetic graphs, plus checks on the saved
  models when they are present).

### Adding a model
1. Create `gnnvis/models/<name>.py` with a `GNNBase` subclass decorated with
   `@register_model("NAME", attention=..., stability=...)` and import it in
   `gnnvis/models/__init__.py`.
2. Add its hyperparameters to `MODEL_CONFIGS` in `train_models.py` and train it.

The app's model dropdown, the attention panel and the stability tab follow the
registry and the model's capabilities; no other code needs to know its name.

### Stability caches after retraining
After retraining a model, run `python precompute_node_stability.py` (and
`python precompute_link_stability.py` for link stability). They recompute only the
files that are missing or out of date for the current models; `--force` recomputes all.

### Adding an explainer
Write `fn(model, data, target, **options)` (`target` is a node index, or a
`(source, target)` pair for links) returning the same kind of result as the built-in
one, and register it in a module imported by `gnnvis/explainers/__init__.py`:

    @register_explainer('MyExplainer', NODE_CLASSIFICATION, ATTRIBUTION, default=True)
    def my_explainer(model, data, node_idx, predicted_class=None): ...

With `default=True` the app uses it instead of the built-in one.


For better performance GPU should be utilized

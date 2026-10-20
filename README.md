## Introduction
InteractiveGNNExplainer, a visual analytics framework to enhance GNN explainability.  
Paper published in 29th International Conference Information Visualisation 2025 (https://ieeexplore.ieee.org/abstract/document/11216840)

## Offline Phase
To start first run the following codes:
1. preprocess_dataset.py
2. train_models.py
3. precompute_stability.py 
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
- `app.py`: the Dash layout and callbacks; `app_stability.py`, `app_expl.py`,
  `cpa_iv_link_prediction.py`, `link_prediction_utils.py`: analysis code.
- `train_models.py`, `precompute_stability.py`, `precompute_link_stability.py`: offline scripts.
- `tests/`: `python -m pytest` (a few seconds; synthetic graphs, plus checks on the saved
  models when they are present).

### Adding a model
1. Create `gnnvis/models/<name>.py` with a `GNNBase` subclass decorated with
   `@register_model("NAME", attention=..., stability=...)` and import it in
   `gnnvis/models/__init__.py`.
2. Add its hyperparameters to `MODEL_CONFIGS` in `train_models.py` and train it.

The app's model dropdown, the attention panel and the stability tab follow the
registry and the model's capabilities; no other code needs to know its name.


For better performance GPU should be utilized

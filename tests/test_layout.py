"""The page layout builds on its own, with unique component ids."""
import json
from collections import Counter

import plotly

from gnnvis.config import AVAILABLE_DATASETS, AVAILABLE_MODELS
from gnnvis.ui.layout import build_layout


def _components(layout):
    """Every component in the tree as a serialized dict."""
    data = json.loads(json.dumps(layout, cls=plotly.utils.PlotlyJSONEncoder))
    out = []

    def walk(c):
        if isinstance(c, dict) and 'props' in c:
            out.append(c)
            for v in c['props'].values():
                walk(v)
        elif isinstance(c, list):
            for x in c:
                walk(x)
    walk(data)
    return out


def _ids(layout):
    return [c['props']['id'] for c in _components(layout) if isinstance(c['props'].get('id'), str)]


def test_component_ids_are_unique():
    dupes = [i for i, n in Counter(_ids(build_layout())).items() if n > 1]
    assert dupes == []


def test_core_components_present():
    ids = set(_ids(build_layout()))
    for required in ('status-bar', 'layer-1', 'layer-2', 'layer-3', 'global-run-button', 'reset-button',
                     'dataset-dropdown', 'model-dropdown', 'analysis-mode-switch', 'embeddings-view',
                     'confusion-matrix-plot', 'edge-predictions-display', 'graph-view', 'full-graph-view',
                     'main-stability-plot', 'run-stability-button', 'selected-node-store', 'selected-link-store'):
        assert required in ids, required


def test_tabs_and_dropdown_options():
    comps = _components(build_layout())
    tabs = [c['props']['value'] for c in comps if c['type'] == 'Tab']
    assert tabs == ['tab-gnn', 'tab-full-graph', 'tab-stability']
    dropdowns = {c['props']['id']: c['props']['options'] for c in comps
                 if c['type'] == 'Dropdown' and c['props'].get('id') in ('dataset-dropdown', 'model-dropdown')}
    assert dropdowns['dataset-dropdown'] == AVAILABLE_DATASETS
    assert dropdowns['model-dropdown'] == AVAILABLE_MODELS   # follows the model registry


def test_layout_is_rebuilt_fresh_each_call():
    # build_layout() returns a new tree each time (no shared mutable components)
    assert build_layout() is not build_layout()

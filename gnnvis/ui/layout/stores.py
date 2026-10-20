"""Browser-side stores and hidden placeholder components."""

from dash import html, dcc


def stores():
    """Shared browser-side state (dcc.Store) and hidden components that
    callbacks still reference."""
    return [
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

        ])
    ]

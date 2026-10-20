"""Browser-side stores and hidden placeholder components."""

from dash import html, dcc
from gnnvis.ui import ids


def stores():
    """Shared browser-side state (dcc.Store) and hidden components that
    callbacks still reference."""
    return [
        dcc.Store(id=ids.PRECOMPUTED_PACKAGE_STORE),

        dcc.Store(id=ids.FULL_DATASET_STORE),

        dcc.Store(id=ids.EDITABLE_GRAPH_STORE),

        dcc.Store(id=ids.CURRENT_PREDICTIONS_STORE),

        dcc.Store(id=ids.SELECTED_NODE_STORE),

        dcc.Store(id=ids.EXPLANATION_STORE),

        dcc.Store(id=ids.CPA_IV_STORE),

        dcc.Store(id=ids.LINK_CPA_IV_STORE),

        dcc.Store(id=ids.SELECTED_PATH_STORE),

        # Placeholder removed: actual header component is defined in Layer 3

        dcc.Store(id=ids.SELECTED_LINK_STORE),

        dcc.Store(id=ids.ANALYSIS_MODE_STORE, data='node_classification'),

        dcc.Store(id=ids.GRAPH_TYPE_STORE, data='full'),

        dcc.Store(id=ids.REMOVED_EDGES_STORE, data=None),

        dcc.Store(id=ids.EDGE_STATUS_STORE, data=None),

        dcc.Store(id=ids.LINK_STATUS_STORE, data=None),

        dcc.Store(id=ids.CPA_STATUS_STORE, data=None),

        dcc.Store(id=ids.SUBGRAPH_EVAL_STORE, data=None),
        dcc.Store(id=ids.CM_DETAIL_PAGING_STORE, data=None),

        dcc.Store(id=ids.STABILITY_RESULTS_STORE, data=None),

        dcc.Store(id=ids.DRILLDOWN_CLICK_STORE, data=None),

        dcc.Store(id=ids.LINK_PREDICTION_STORE),

        # Hidden placeholder components

        html.Div(style={'display': 'none'}, children=[

            dcc.Input(id=ids.LINK_SOURCE_INPUT, value=None),

            dcc.Input(id=ids.LINK_TARGET_INPUT, value=None),

            html.Button(id=ids.ANALYZE_LINK_BUTTON, n_clicks=0),

            html.Button(id=ids.PREDICT_LINKS_BUTTON, n_clicks=0),

            dcc.Input(id=ids.SELECT_NODE_INPUT_NC, value=None),

            html.Button(id=ids.RUN_CPA_IV_BUTTON, n_clicks=0),

            html.Button(id=ids.ADD_EDGE_BUTTON_LP, n_clicks=0),

            html.Button(id=ids.REMOVE_EDGE_BUTTON_LP, n_clicks=0),

        ])
    ]

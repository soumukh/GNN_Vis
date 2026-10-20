"""Explanation tab: local subgraph, neighbour and feature importance, CPA-IV / attention cards."""

import dash_cytoscape as cyto
from dash import html, dcc

from gnnvis.ui.figures.subgraph import STYLESHEET
from gnnvis.ui import ids


def explanation_tab():
    return (
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
                                    id=ids.SUBGRAPH_COLOR_MODE,
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
                                id=ids.GRAPH_VIEW,
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
                            html.Div(id=ids.SUBGRAPH_LEGEND_INLINE, style={
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
                                id=ids.LAYER3_RIGHT_HEADER,
                                style={'fontWeight': 'bold', 'fontSize': '12px',
                                       'color': '#0056b3', 'margin': '0 0 6px 0',
                                       'textTransform': 'uppercase', 'letterSpacing': '0.3px'}),
                            dcc.Loading(type='dot', color='#2563eb', children=dcc.Graph(
                                id=ids.NEIGHBOR_IMPORTANCE_VIEW,
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
                                id=ids.FEATURE_IMPORTANCE_VIEW,
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
                        html.Div(id=ids.GNN_CAUSAL_EXPLANATION_VIEW, style={'display': 'none'})
                    ]),

                    # ── A6: Subgraph Legend (moved inline into the Local Subgraph panel)

                ]),
            ],
        )
    )

"""Stability Analysis tab: noise / plot / metric controls, main plot, statistics, drill-down."""

import plotly.graph_objects as go
from dash import html, dcc
from gnnvis.ui import ids


def stability_tab():
    return (
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

                    html.Div(id=ids.STABILITY_SECTION, children=[

                        # ── Mode toggle: Node / Link ──────────────────────────────────
                        # Node Stability button removed; show node stability controls directly

                        # ── NODE stability section ────────────────────────────────────
                        html.Div(id=ids.NODE_STABILITY_SECTION, children=[

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
                                    id=ids.STABILITY_SIGMA_SELECTOR,
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
                                        id=ids.STABILITY_PLOT_TYPE,
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
                                        id=ids.STABILITY_METRIC_SELECTOR,
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
                                    id=ids.STABILITY_SAMPLE_SIZE, type="hidden",
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
                            dcc.Input(id=ids.STABILITY_METHOD_SELECTOR,
                                      type='hidden', value='gnn_explainer'),
                            dcc.Input(id=ids.STABILITY_TRIALS,
                                      type='hidden', value=20),

                            html.Button(
                                "Run Stability Analysis",
                                id=ids.RUN_STABILITY_BUTTON, n_clicks=0,
                                style={
                                    'backgroundColor': '#17a2b8', 'color': 'white',
                                    'border': 'none', 'padding': '8px 20px',
                                    'borderRadius': '4px', 'cursor': 'pointer',
                                    'fontWeight': 'bold', 'fontSize': '12px',
                                    'whiteSpace': 'nowrap',
                                },
                            ),

                            html.Div(id=ids.STABILITY_STATUS,
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
                                        id=ids.MAIN_STABILITY_PLOT, 
                                        figure=go.Figure(layout={'title': 'Select a metric to view', 'height': 360, 'margin': dict(l=40, r=15, t=40, b=40), 'paper_bgcolor': 'rgba(0,0,0,0)', 'plot_bgcolor': 'rgba(0,0,0,0)'}),
                                        style={'height': '360px'}, 
                                        config={'displayModeBar': False}
                                    ))
                                ]),

                                # Stats Panel
                                html.Div(
                                    id=ids.STABILITY_STATS_PANEL,
                                    style={'border': '1px solid #ccc', 'padding': '10px', 'backgroundColor': '#fff', 'borderRadius': '4px', 'fontSize': '12px'},
                                    children=html.Div("Run analysis to see statistics.", style={'color': '#999', 'textAlign': 'center', 'paddingTop': '10px'})
                                ),
                            ]),

                            # --- RIGHT COLUMN (50%) ---
                            html.Div(id=ids.DRILLDOWN_RIGHT_PANEL, style={'flex': '1', 'minWidth': '0', 'display': 'none', 'flexDirection': 'column', 'gap': '12px'}, children=[
                                # Drill-down controls
                                html.Div(style={
                                    'display': 'flex', 'alignItems': 'center', 'gap': '10px',
                                    'padding': '8px 12px', 'backgroundColor': '#f0f7ff', 'borderRadius': '6px',
                                    'border': '1px solid #b8d4f0', 'flexWrap': 'wrap'
                                }, children=[
                                    html.Span("Region Size (ε):", style={'fontSize': '12px', 'fontWeight': 'bold', 'color': '#1a5276', 'whiteSpace': 'nowrap'}),
                                    html.Div(style={'flex': '1', 'minWidth': '120px'}, children=[
                                        dcc.Slider(
                                            id=ids.DRILLDOWN_EPSILON_SLIDER,
                                            min=0.01, max=0.30, step=0.01, value=0.08,
                                            marks={v: f'{v:.2f}' for v in [0.01, 0.05, 0.10, 0.15, 0.20, 0.30]},
                                            tooltip={'placement': 'bottom', 'always_visible': False},
                                        ),
                                    ]),
                                    html.Button(
                                        "Reset Selection", id=ids.DRILLDOWN_RESET_BTN, n_clicks=0,
                                        style={'fontSize': '12px', 'padding': '4px 12px', 'backgroundColor': '#dc3545', 'color': '#fff', 'border': 'none', 'borderRadius': '4px', 'cursor': 'pointer', 'whiteSpace': 'nowrap'}
                                    ),
                                ]),
                                html.Div(
                                    id=ids.DRILLDOWN_HINT, 
                                    style={'fontSize': '11px', 'color': '#666', 'fontStyle': 'italic', 'textAlign': 'center'}, 
                                    children="Click any region on the left plot to drill down"
                                ),

                                # Drill-down scatter panel
                                html.Div(style={'border': '1px solid #dee2e6', 'borderRadius': '6px', 'backgroundColor': '#fafafa', 'padding': '8px'}, children=[
                                    html.Div(
                                        id=ids.DRILLDOWN_PANEL_HEADER,
                                        children="Click on a region to explore node-level details",
                                        style={'fontSize': '12px', 'fontWeight': 'bold', 'color': '#555', 'marginBottom': '6px', 'padding': '4px 8px', 'backgroundColor': '#e9ecef', 'borderRadius': '4px'}
                                    ),
                                    dcc.Graph(
                                        id=ids.DRILLDOWN_SCATTER,
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
        )
    )

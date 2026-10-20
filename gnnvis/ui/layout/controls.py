"""Layer 1: task / dataset / model / selection controls, metric tiles, RESET and RUN."""

from dash import html, dcc

from gnnvis.config import AVAILABLE_DATASETS, DEFAULT_DATASET, DEFAULT_MODEL, AVAILABLE_MODELS
from gnnvis.ui import ids


def controls():
    return (
        html.Div(id=ids.LAYER_1, style={'borderBottom': 'none', 'paddingTop': '1px', 'paddingBottom': '15px', 'marginBottom': '20px'}, children=[

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
                        id=ids.ANALYSIS_MODE_SWITCH,
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
                        id=ids.DATASET_DROPDOWN,
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
                        id=ids.MODEL_DROPDOWN,
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
                        id=ids.SELECTION_INTERFACE,
                        style={'minWidth': '150px'}
                    )
                ]
            )
        ]
        ),

                # Compact performance metrics centered between left controls and right buttons
                html.Div(id=ids.PERFORMANCE_METRICS, style={'flex': '0 0 auto', 'alignSelf': 'center', 'display': 'none', 'gap': '6px', 'alignItems': 'center', 'fontSize': '18px', 'fontWeight': 'bold'}),

                # Right Control (Buttons),

                html.Div(style={'display': 'flex', 'gap': '10px'}, children=[

                    html.Button(

                        "RESET", 

                        id=ids.RESET_BUTTON, 

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

                        id=ids.GLOBAL_RUN_BUTTON, 

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
            html.Div(id=ids.DYNAMIC_ANALYSIS_TABS, style={'display': 'none'}),

        ])
    )

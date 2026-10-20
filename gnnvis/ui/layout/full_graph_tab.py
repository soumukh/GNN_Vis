"""Full Graph View tab: colour / filter controls and the full-graph figure."""

import plotly.graph_objects as go
from dash import html, dcc
from gnnvis.ui import ids


def full_graph_tab():
    return (
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
                                id=ids.FULL_GRAPH_COLOR_MODE,
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
                                id=ids.FULL_GRAPH_FILTER_METRIC,
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
                                id=ids.FULL_GRAPH_FILTER_RANGE,
                                min=0, max=1, step=0.01, value=[0, 1],
                                marks={0: '0', 0.5: '0.5', 1: '1'},
                                allowCross=False,
                                tooltip={'placement': 'bottom', 'always_visible': False}
                            )
                        ]),
                    ]),
                    dcc.Loading(type='dot', color='#2563eb', children=dcc.Graph(
                        id=ids.FULL_GRAPH_VIEW,
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
        )
    )

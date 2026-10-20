"""Layer 2: node embeddings, confusion matrix and cell details."""

from dash import html, dcc

from gnnvis.ui.components import cell_details_placeholder, confusion_matrix_style


def overview():
    return (
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

            ])
    )

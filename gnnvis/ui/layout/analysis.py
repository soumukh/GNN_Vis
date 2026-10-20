"""Layer 3: the tabbed explanation / full-graph / stability section."""

from dash import html, dcc

from gnnvis.ui.layout.explanation_tab import explanation_tab
from gnnvis.ui.layout.full_graph_tab import full_graph_tab
from gnnvis.ui.layout.stability_tab import stability_tab


def analysis_section():
    return (
        html.Div(id='layer-3', style={'display': 'none', 'borderTop': '2px solid #dee2e6', 'paddingTop': '0', 'marginTop': '0', 'minHeight': '0', 'overflow': 'visible'}, children=[
            html.Div(id='loading-layer-3', children=[

                dcc.Tabs(
                    id='layer3-tabs',
                    value='tab-gnn',
                    style={'borderBottom': '2px solid #dee2e6', 'minHeight': '100%'},
                    colors={'border': '#dee2e6', 'primary': '#007bff', 'background': '#f8f9fa'},
                    children=[

                        # ══════════════════════════════════════════════════
                        # TAB A: EXPLANATION
                        # ══════════════════════════════════════════════════
                        explanation_tab(),

                        # ══════════════════════════════════════════════════
                        # TAB B: FULL GRAPH VIEW
                        # ══════════════════════════════════════════════════
                        full_graph_tab(),

                        # ══════════════════════════════════════════════════
                        # TAB C: STABILITY ANALYSIS
                        # ══════════════════════════════════════════════════
                        stability_tab(),

                    ],
                ),

            ])

        ])
    )

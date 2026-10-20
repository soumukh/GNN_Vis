"""Page layout, one module per area; build_layout() assembles the page."""

from dash import html, dcc

from gnnvis.ui.layout.analysis import analysis_section
from gnnvis.ui.layout.overview import overview
from gnnvis.ui.layout.controls import controls
from gnnvis.ui.layout.header import header
from gnnvis.ui.layout.stores import stores


def build_layout():
    """The whole page: stores, header, controls, overview, tabbed analysis."""
    return (
        html.Div(style={

        'fontFamily': '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',

        'backgroundColor': '#f5f7fb',

        'minHeight': '100vh',

        'margin': 0,

        'padding': 0,

        'color': '#333'

        }, children=[

        dcc.Location(id='url', refresh=True),

        html.Div(style={'maxWidth': '100%', 'margin': '0 auto', 'padding': '20px'}, children=[

        *stores(),

        # --- HEADER: title + live status / next-step guidance ---
        header(),

        # --- LAYER 1: TOP (USER CONTROLS ONLY) ---

        controls(),

        # --- LAYER 2: MID (GLOBAL VIEW ONLY) ---

        overview(),

        ]),

        # --- LAYER 3: BOTTOM — THREE EXPLANATION FRAMES ---

        analysis_section(),

        # Legacy sections (HIDDEN),

        html.Div(id='node-classification-section', style={'display': 'none'}),

        ])
    )

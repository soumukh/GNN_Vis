"""Header: title, subtitle and the live status line."""

from dash import html


def header():
    return (
        html.Div(className='app-header', children=[
            html.Div(className='app-brand', children=[
                html.H1("Interactive GNN Explainer", className='app-title'),
                html.Span("Visual analytics for GCN / GAT predictions, explanations and stability",
                          className='app-subtitle'),
            ]),
            html.Div(id='status-bar', className='status-bar'),
        ])
    )

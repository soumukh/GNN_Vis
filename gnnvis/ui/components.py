"""Small reusable Dash components: metric tiles, cards, placeholders, keys, messages."""

from dash import html
from gnnvis.models import available_models
from gnnvis.models import model_capabilities
from gnnvis.ui.theme import CORRECT_COLOR, INCORRECT_COLOR, class_color_map, class_name


def stability_unsupported_message(model_type):
    supported = [m for m in available_models() if model_capabilities(m)['stability']]
    return (f"⚠️ Stability analysis is not available for {model_type} models. "
            f"Switch the model to {' or '.join(supported)} to run it.")


def metric_tiles(items):
    """Compact stat tiles: items = [(label, value, tooltip), ...]."""
    return [
        html.Div([
            html.Div(f"{value:.3f}" if isinstance(value, float) else str(value), className='metric-value'),
            html.Div(label, className='metric-label'),
        ], className='metric-tile', title=tip)
        for label, value, tip in items
    ]


_EXPLAIN_CARD = {'backgroundColor': '#f8f9ff', 'border': '1px solid #dde', 'borderRadius': '6px',
                 'padding': '12px', 'boxSizing': 'border-box'}


def explanation_cards(cpa_content, attention_content):
    """CPA-IV card at exactly the Local Subgraph column width; the GAT attention
    card takes the remaining space to its right (instead of both sharing the
    subgraph width). Width maths: see the explain-row note on .cpa-frame."""
    cards = []
    if cpa_content:
        cards.append(html.Div(cpa_content, className='cpa-card',
                              style={**_EXPLAIN_CARD, 'width': 'calc((100% - 50px) / 2)', 'flex': '0 0 auto'}))
    if attention_content:
        cards.append(html.Div(attention_content, className='attention-card',
                              style={**_EXPLAIN_CARD, 'flex': '1 1 0', 'minWidth': '280px'}))
    return html.Div(cards, style={'display': 'flex', 'gap': '14px', 'alignItems': 'flex-start', 'flexWrap': 'wrap'})


def confusion_matrix_style(analysis_mode):
    """Graph style for the confusion matrix: fill the panel, but never shrink
    below a readable size (7x7 node matrix needs more room than 2x2 links).
    The panel has overflow:auto, so a scroll bar appears only if it is shorter."""
    min_h = '240px' if analysis_mode == 'link_prediction' else '380px'
    return {'height': '100%', 'minHeight': min_h, 'minWidth': '300px', 'width': '100%',
            'cursor': 'pointer', 'margin': '0', 'padding': '0'}


def cell_details_placeholder(analysis_mode='node_classification'):
    """Cell-details panel before any confusion-matrix cell is clicked."""
    what = ("its edges (TP / FP / FN / TN); click a row to inspect that link"
            if analysis_mode == 'link_prediction'
            else "the nodes in it; click a row to select that node")
    return html.Div([
        html.Div("Cell details", className='legend-title'),
        html.Div(f"Click a cell of the confusion matrix to list {what}.", className='empty-hint'),
    ], className='legend-card', style={'padding': '8px'})


def subgraph_color_key(color_mode, dataset_name, num_classes):
    """Small swatch legend matching the node colours in the Cytoscape subgraph."""
    if color_mode == 'correctness':
        items = [('Correct', CORRECT_COLOR), ('Incorrect', INCORRECT_COLOR)]
    else:
        cmap = class_color_map(num_classes)
        items = [(class_name(dataset_name, i), cmap[i]) for i in range(num_classes)]
    return html.Div(className='color-key', children=[
        html.Span([html.Span(className='swatch', style={'backgroundColor': c}), name], className='key-item')
        for name, c in items
    ])

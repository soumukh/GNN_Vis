"""GAT attention panel: per-layer summary and neighbour attention bar chart."""

import numpy as np
import pandas as pd
import plotly.express as px
from dash import html
from dash import dcc


def _node_attention_scores(attention_weights, layer_name, node_idx):
    """Head-averaged attention scores on the edges touching `node_idx`
    (the same selection _build_attention_neighbor_bar plots)."""
    try:
        layer_data = attention_weights.get(layer_name)
        if not layer_data or not isinstance(layer_data, (list, tuple)) or len(layer_data) < 2:
            return None
        edge_arr, att_arr = np.array(layer_data[0]), np.array(layer_data[1], dtype=float)
        if edge_arr.ndim == 2 and edge_arr.shape[0] == 2:
            srcs, dsts = edge_arr[0], edge_arr[1]
        elif edge_arr.ndim == 2 and edge_arr.shape[1] == 2:
            srcs, dsts = edge_arr[:, 0], edge_arr[:, 1]
        else:
            return None
        scores = att_arr.mean(axis=1) if att_arr.ndim == 2 else att_arr
        mask = (srcs == int(node_idx)) | (dsts == int(node_idx))
        return scores[mask]
    except Exception:
        return None


def _build_attention_neighbor_bar(attention_weights, node_idx, layer_name='conv1', top_k=None):
    """Build a bar chart of 1-hop attention scores for the selected node and layer."""
    try:
        layer_data = attention_weights.get(layer_name)
        if not layer_data or not isinstance(layer_data, (list, tuple)) or len(layer_data) < 2:
            return None

        edge_index, att_values = layer_data
        if edge_index is None or att_values is None:
            return None

        edge_arr = np.array(edge_index)
        att_arr = np.array(att_values)

        # Support both shapes: (2, E) or (E, 2)
        if edge_arr.ndim == 2 and edge_arr.shape[0] == 2:
            srcs, dsts = edge_arr[0], edge_arr[1]
        elif edge_arr.ndim == 2 and edge_arr.shape[1] == 2:
            srcs, dsts = edge_arr[:, 0], edge_arr[:, 1]
        else:
            return None

        # Collapse per-head scores if present
        if att_arr.ndim == 2:
            att_scores = np.mean(att_arr, axis=1)
        else:
            att_scores = att_arr.astype(float)

        mask = (srcs == int(node_idx)) | (dsts == int(node_idx))
        if not np.any(mask):
            return None

        neighbors = np.where(srcs[mask] == int(node_idx), dsts[mask], srcs[mask])
        if neighbors.size == 0:
            return None

        df = pd.DataFrame({
            'Neighbor': neighbors.astype(int),
            'Attention': att_scores[mask].astype(float)
        })
        df = df.groupby('Neighbor', as_index=False).mean()
        df = df.sort_values('Attention', ascending=False)

        if df.empty:
            return None

        if top_k is not None:
            df = df.head(int(top_k))

        fig = px.bar(
            df,
            x='Neighbor',
            y='Attention',
            title=f'1-Hop {layer_name.upper()} Attention for Node {node_idx}',
            text_auto='.3f',
            labels={'Neighbor': 'Neighbor Node', 'Attention': 'Attention Score'},
            color='Attention',
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=40, b=10),
            height=320,
        )
        fig.update_xaxes(type='category')
        return dcc.Graph(figure=fig, config={'displayModeBar': False})
    except Exception:
        return None


def create_attention_panel(attention_weights, node_idx):

    """Create a panel displaying attention weights for GAT models."""

    if not attention_weights:

        return html.Div("No attention data available (only for GAT models).", 

                       style={'color': '#999', 'fontStyle': 'italic', 'fontSize': '13px'})

    if isinstance(attention_weights, dict):
        summary_items = []
        for layer_name in ('conv1', 'conv2'):
            # Stored as (edge_index, attention): summarise the attention values only,
            # for the selected node's edges (same rule as the bar chart below).
            # Flattening the whole tuple mixed node ids into the average (e.g. Max 2707).
            node_scores = _node_attention_scores(attention_weights, layer_name, node_idx)
            if node_scores is None or node_scores.size == 0:
                continue
            avg_weight = float(np.mean(node_scores))
            max_weight = float(np.max(node_scores))
            summary_items.append(
                html.Div([
                    html.Div(layer_name.upper(), style={'fontWeight': 'bold', 'fontSize': '12px', 'color': '#555', 'marginBottom': '4px'}),
                    html.Div([
                        html.Span(f"Avg: {avg_weight:.3f}", style={'fontSize': '12px', 'color': '#007bff', 'fontWeight': 'bold'}),
                        html.Span(f" | Max: {max_weight:.3f}", style={'fontSize': '12px', 'color': '#666', 'marginLeft': '8px'}),
                        html.Span(f" | {node_scores.size} edges", style={'fontSize': '11px', 'color': '#999', 'marginLeft': '8px'})
                    ])
                ], style={'padding': '6px 0', 'borderBottom': '1px solid #eee'})
            )

        # Build a compact summary row for conv1 and conv2 and show side-by-side
        panel_children = [
            html.H5(f"GAT Attention for Node {node_idx}",
                   style={'marginBottom': '10px', 'color': '#333', 'fontSize': '14px', 'fontWeight': 'bold'}),
        ]

        if summary_items:
            # Ensure exactly two summary boxes (conv1, conv2) are shown side-by-side
            summary_boxes = []
            for item in summary_items:
                summary_boxes.append(html.Div(item, style={'flex': '1', 'minWidth': '160px', 'padding': '6px', 'border': '1px solid #eee', 'borderRadius': '4px', 'backgroundColor': '#fafafa'}))
            # If only one summary exists, add a placeholder for the other
            if len(summary_boxes) == 1:
                summary_boxes.append(html.Div(html.Div("No data"), style={'flex': '1', 'minWidth': '160px', 'padding': '6px'}))
            panel_children.append(html.Div(summary_boxes, style={'display': 'flex', 'gap': '12px', 'marginBottom': '12px'}))

        # Keep a single bar graph below. Prefer conv1, else conv2.
        graph = _build_attention_neighbor_bar(attention_weights, node_idx, layer_name='conv1')
        if graph is None:
            graph = _build_attention_neighbor_bar(attention_weights, node_idx, layer_name='conv2')

        if graph is not None:
            panel_children.append(graph)
        else:
            panel_children.append(html.Div("No 1-hop attention weights available for this node.", style={'color': '#999', 'fontStyle': 'italic', 'fontSize': '13px'}))

        return html.Div(panel_children)

    return html.Div("Attention data format not recognized.", 

                   style={'color': '#dc3545', 'fontStyle': 'italic'})

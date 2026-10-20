"""Full-graph view: sampled, filtered spring layout coloured by class or correctness."""

import traceback
import numpy as np
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
from dash import no_update
from gnnvis.ui.theme import CORRECT_COLOR, INCORRECT_COLOR, class_name, empty_figure


_FULL_GRAPH_LAYOUT_CACHE = {}  # hash(nodes, edges) -> spring_layout positions


def build_full_graph_figure(active_tab, color_mode, filter_metric, filter_range, selected_node, predictions_data, dataset, graph_data, stability_results):
    """
    Build a full-graph Plotly figure for the embedded frame.

    Data source  : full-dataset-store + editable-graph-store
    Coloring     : predicted class → px.colors.qualitative.Plotly palette
    Layout       : NetworkX spring_layout
    Performance  : capped at MAX_MODAL_NODES; selected node highlighted if any
    """
    MAX_MODAL_NODES = 800  # cap for the embedded graph

    empty_fig = go.Figure(layout={
        'title': {'text': ''},
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': '#f9fbff',
    })

    # Only build the (expensive) figure while its tab is visible
    if active_tab != 'tab-full-graph':
        return no_update

    if not dataset or not graph_data or not predictions_data:
        return empty_figure("Load a model to see the full graph")

    try:
        num_nodes   = dataset['num_nodes']
        preds       = predictions_data.get('preds', [])
        edge_index  = graph_data.get('edge_index', [[], []])
        num_classes = dataset.get('num_classes', 7)
        dataset_name = dataset.get('name', 'Graph')

        if not preds:
            return empty_fig

        preds_arr = np.array(preds)
        true_labels = np.asarray(dataset.get('y')) if dataset.get('y') is not None else None
        has_true_labels = true_labels is not None and len(true_labels) == num_nodes and len(preds_arr) >= num_nodes

        def safe_int(arr, idx, default=-1):
            try:
                return int(arr[idx]) if arr is not None and 0 <= idx < len(arr) else default
            except Exception:
                return default

        # ── Apply confidence / stability filters ─────────────────────────
        filter_metric = (filter_metric or 'confidence').lower() if filter_metric else 'confidence'
        filter_range = filter_range if isinstance(filter_range, (list, tuple)) and len(filter_range) == 2 else [0.0, 1.0]
        try:
            filter_min, filter_max = float(filter_range[0]), float(filter_range[1])
        except Exception:
            filter_min, filter_max = 0.0, 1.0
        filter_min = max(0.0, min(filter_min, 1.0))
        filter_max = max(0.0, min(filter_max, 1.0))

        filtered_nodes = list(range(num_nodes))
        if filter_metric == 'confidence':
            confidences = predictions_data.get('confidences') if isinstance(predictions_data, dict) else None
            if confidences and len(confidences) >= num_nodes:
                filtered_nodes = [
                    n for n in filtered_nodes
                    if 0 <= n < len(confidences) and filter_min <= float(confidences[n]) <= filter_max
                ]
        elif filter_metric == 'stability' and stability_results:
            stability_map = {}
            for row in stability_results:
                try:
                    idx = int(row.get('node_idx'))
                    stability_map[idx] = float(row.get('stability'))
                except Exception:
                    continue
            if stability_map:
                filtered_nodes = [
                    n for n in filtered_nodes
                    if n in stability_map and filter_min <= stability_map[n] <= filter_max
                ]

        if len(filtered_nodes) == 0:
            return go.Figure(layout={
                'title': {'text': 'No nodes match the selected filter range.'},
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': '#f9fbff',
            })

        # ── Build NetworkX graph ──────────────────────────────────────
        G = nx.Graph()
        G.add_nodes_from(filtered_nodes)
        src_list, tgt_list = edge_index[0], edge_index[1]
        seen = set()
        node_set = set(filtered_nodes)
        for u, v in zip(src_list, tgt_list):
            if u in node_set and v in node_set:
                key = (min(u, v), max(u, v))
                if key not in seen:
                    seen.add(key)
                    G.add_edge(u, v)

        # ── Sample if too large ──────────────────────────────────────
        sampled = False
        display_nodes = filtered_nodes
        extra_node = None   # selected node added on top of a fixed sample
        G_filtered = G
        if len(filtered_nodes) > MAX_MODAL_NODES:
            sampled = True
            rng = np.random.default_rng(42)
            # Fixed sample (independent of the selection) so its layout can be cached
            display_nodes = rng.choice(filtered_nodes, MAX_MODAL_NODES, replace=False).tolist()
            sel_id = None
            if selected_node and selected_node.get('id') is not None:
                try:
                    sel_id = int(selected_node['id'])
                except (ValueError, TypeError):
                    pass
            if sel_id is not None and sel_id in node_set and sel_id not in display_nodes:
                extra_node = sel_id
            G = G.subgraph(display_nodes).copy()

        # ── Spring layout (cached: colour/filter/selection changes reuse positions) ────
        layout_key = (dataset_name, tuple(sorted(G.nodes())), tuple(sorted(G.edges())))
        layout_key = hash(layout_key)
        pos = _FULL_GRAPH_LAYOUT_CACHE.get(layout_key)
        if pos is None:
            pos = nx.spring_layout(G, seed=42, k=1.4 / max(1, len(G) ** 0.5))
            if len(_FULL_GRAPH_LAYOUT_CACHE) > 32:
                _FULL_GRAPH_LAYOUT_CACHE.clear()
            _FULL_GRAPH_LAYOUT_CACHE[layout_key] = pos

        # Keep the selected node visible: place it at the centroid of its
        # sampled neighbours (no re-layout, so selection changes stay fast)
        if extra_node is not None:
            pos = dict(pos)
            nbrs = [n for n in G_filtered.neighbors(extra_node) if n in pos]
            if nbrs:
                pos[extra_node] = np.mean([pos[n] for n in nbrs], axis=0)
            else:
                pos[extra_node] = np.array([0.0, 0.0])
            G.add_node(extra_node)
            G.add_edges_from((extra_node, n) for n in nbrs)
            display_nodes = list(display_nodes) + [extra_node]

        # ── Color palette ─────────────────────────────────────────────
        color_palette   = px.colors.qualitative.Plotly
        legend_title = 'Color Key'
        if color_mode == 'actual' and has_true_labels:
            node_value_map = {n: safe_int(true_labels, n, 0) for n in display_nodes}
            class_colors = [color_palette[c % len(color_palette)] for c in range(num_classes)]
            legend_name_fn = lambda cls: class_name(dataset_name, cls)
            legend_title = 'Actual class'
        elif color_mode == 'correctness' and has_true_labels:
            node_value_map = {n: 0 if safe_int(preds_arr, n, -1) == safe_int(true_labels, n, -2) else 1 for n in display_nodes}
            class_colors = [CORRECT_COLOR, INCORRECT_COLOR]
            num_classes = 2
            legend_name_fn = lambda cls: 'Correct' if cls == 0 else 'Incorrect'
            legend_title = 'Prediction'
        else:
            node_value_map = {n: safe_int(preds_arr, n, 0) for n in display_nodes}
            class_colors = [color_palette[c % len(color_palette)] for c in range(num_classes)]
            legend_name_fn = lambda cls: class_name(dataset_name, cls)
            legend_title = 'Predicted class'

        # ── Edge trace ────────────────────────────────────────────────
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            mode='lines',
            line=dict(width=0.5, color='#bbb'),
            hoverinfo='none',
            showlegend=False,
        )

        # ── Node traces — one per class ───────────────────────────────
        sel_node_id = None
        if selected_node and selected_node.get('id') is not None:
            try:
                sel_node_id = int(selected_node['id'])
            except (ValueError, TypeError):
                pass

        node_values = np.array([int(node_value_map.get(n, 0)) for n in G.nodes()])
        node_traces = []
        for cls in range(num_classes):
            cls_nodes = [n for n, val in zip(G.nodes(), node_values) if val == cls and n != sel_node_id]
            if not cls_nodes:
                continue

            # Mark the selected node with a different symbol
            symbols = [
                'star' if n == sel_node_id else 'circle'
                for n in cls_nodes
            ]
            sizes = [
                16 if n == sel_node_id else 7
                for n in cls_nodes
            ]
            border_colors = [
                '#FFD700' if n == sel_node_id else 'white'
                for n in cls_nodes
            ]
            border_widths = [
                3 if n == sel_node_id else 0.5
                for n in cls_nodes
            ]

            hover_templates = []
            for n in cls_nodes:
                pred_class = safe_int(preds_arr, n, -1)
                actual_class = None
                if true_labels is not None and 0 <= n < len(true_labels):
                    try:
                        actual_class = int(true_labels[n])
                    except Exception:
                        actual_class = None

                if actual_class is not None:
                    correct = pred_class == actual_class
                    hover_templates.append(
                        f"Node {n}<br>Predicted: {class_name(dataset_name, pred_class)}<br>Actual: {class_name(dataset_name, actual_class)}<br>{'Correct' if correct else 'Incorrect'}<extra></extra>"
                    )
                else:
                    hover_templates.append(
                        f'Node {n}<br>Predicted: {class_name(dataset_name, pred_class)}<extra></extra>'
                    )

            node_traces.append(go.Scatter(
                x=[pos[n][0] for n in cls_nodes],
                y=[pos[n][1] for n in cls_nodes],
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=class_colors[cls],
                    symbol=symbols,
                    line=dict(width=border_widths, color=border_colors),
                    opacity=0.85,
                ),
                hovertemplate=hover_templates,
                customdata=cls_nodes,
                name=legend_name_fn(cls),
            ))

        if sel_node_id is not None and sel_node_id in pos:
            node_traces.append(go.Scatter(
                x=[pos[sel_node_id][0]], y=[pos[sel_node_id][1]], mode='markers',
                marker=dict(symbol='star', size=18, color='#FFD700', line=dict(width=1.5, color='#333')),
                customdata=[sel_node_id], name=f'Selected (node {sel_node_id})',
                hovertemplate=f'Selected: node {sel_node_id}<extra></extra>',
            ))

        title_text = (
            f'{dataset_name} — Full Graph'
            + (f'  (sampled {len(display_nodes):,} / {num_nodes:,} nodes)' if sampled else
               f'  ({num_nodes:,} nodes, {len(seen):,} edges)')
        )

        fig = go.Figure(data=[edge_trace] + node_traces)
        fig.update_layout(
            title=dict(text=title_text, font=dict(size=15, color='#222')),
            showlegend=True,
            legend=dict(
                title=dict(text=legend_title, font=dict(size=12)),
                orientation='v',
                x=1.01, y=1,
                bgcolor='rgba(255,255,255,0.92)',
                bordercolor='#ccc',
                borderwidth=1,
                font=dict(size=12),
            ),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='#f9fbff',
            margin=dict(l=10, r=160, t=50, b=10),
            hovermode='closest',
        )
        return fig

    except Exception as exc:
        traceback.print_exc()
        return go.Figure(layout={
            'title': f'Error building full graph: {exc}',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': '#f9fbff',
        })

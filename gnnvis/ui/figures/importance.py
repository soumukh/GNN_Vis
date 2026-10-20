"""Neighbour-importance table and feature-importance bar chart."""

import pandas as pd
import torch
import plotly.express as px
import plotly.graph_objects as go
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph
from gnnvis.ui.theme import class_name, empty_figure


def get_one_hop_neighbors(pyg_data, node_idx):
    """Return the 1-hop neighbors of a node from the graph edge list."""
    try:
        if not hasattr(pyg_data, 'edge_index') or pyg_data.edge_index is None:
            return []
        neighbors = set()
        for src, dst in pyg_data.edge_index.t().tolist():
            if int(src) == int(node_idx):
                neighbors.add(int(dst))
            elif int(dst) == int(node_idx):
                neighbors.add(int(src))
        return sorted(neighbors)
    except Exception:
        return []


def plot_neighbor_importance(node_importance, node_idx, preds=None, y_labels=None, one_hop_neighbors=None, dataset_name=None):
    """Show a table of 1-hop neighbors with their true class, predicted class, and importance."""
    if one_hop_neighbors is None:
        one_hop_neighbors = sorted(node_importance.keys())

    if not one_hop_neighbors:
        return empty_figure(f"Node {node_idx} has no neighbours")

    neighbor_list = []
    for neighbor in one_hop_neighbors:
        if neighbor not in node_importance:
            importance = 0.0
        else:
            importance = float(node_importance.get(neighbor, 0.0))

        # Get true class if labels are available
        true_class = 'N/A'
        if y_labels is not None and int(neighbor) < len(y_labels):
            try:
                true_class = class_name(dataset_name, int(y_labels[int(neighbor)]))
            except Exception:
                true_class = 'N/A'

        # Get predicted class if predictions are available
        predicted_class = 'N/A'
        if preds is not None and int(neighbor) < len(preds):
            try:
                predicted_class = class_name(dataset_name, int(preds[int(neighbor)]))
            except Exception:
                predicted_class = 'N/A'

        neighbor_list.append({
            'Neighbor': int(neighbor),
            'True Class': true_class,
            'Predicted Class': predicted_class,
            'Importance': importance,
        })

    # Sort by importance descending for readability
    neighbor_list = sorted(neighbor_list, key=lambda item: item['Importance'], reverse=True)

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Neighbor', 'True Class', 'Predicted Class', 'Importance'],
            fill_color='#2c3e50',
            font=dict(color='white', size=13),
            align='left'
        ),
        cells=dict(
            values=[
                [str(item['Neighbor']) for item in neighbor_list],
                [str(item['True Class']) for item in neighbor_list],
                [str(item['Predicted Class']) for item in neighbor_list],
                [f"{item['Importance']:.3f}" for item in neighbor_list],
            ],
            fill_color='white',
            align='left',
            font=dict(color='#2c3e50', size=12),
            height=30,
        )
    )])

    fig.update_layout(
        title=f'1-Hop Neighbor Details for Node {node_idx}',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=40, b=10),
        height=320,
        autosize=False,
    )

    fig.update_traces(
        columnwidth=[0.22, 0.26, 0.26, 0.26]
    )

    return fig


def plot_feature_importance(feature_importance, node_idx, top_k=20, title=None, value_label='Importance'):
    """Plot per-feature importance for a node.

    `feature_importance` can be a dict {feat_idx:score} or a list/array.
    """
    if not feature_importance:
        return empty_figure(f"No feature importance for Node {node_idx}")

    # Normalize into dataframe
    if isinstance(feature_importance, dict):
        items = list(feature_importance.items())
    else:
        # assume sequence
        items = [(i, float(v)) for i, v in enumerate(feature_importance)]

    df = pd.DataFrame(items, columns=['Feature', 'Importance'])
    df = df.sort_values('Importance', ascending=False).head(top_k)
    # Ensure feature labels are strings so Plotly treats them as categorical
    df['Feature'] = df['Feature'].astype(str)

    fig = px.bar(
        df,
        x='Feature',
        y='Importance',
        title=title or f'Feature importance, node {node_idx}',
        hover_data={'Importance': ':.4f'},
    )
    # Single colour: a colour bar would only repeat the bar heights and takes
    # ~40% of this narrow panel; per-bar text labels were unreadably small.
    fig.update_traces(marker_color='#2563eb')

    fig.update_layout(xaxis_type='category', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False,
                      # Narrow panel: small, left-aligned title (may wrap onto a second line)
                      title=dict(font=dict(size=13), x=0.02, xanchor='left', y=0.97, yanchor='top'),
                      yaxis_title=value_label, xaxis_title='Feature index',
                      margin=dict(l=48, r=10, t=56, b=44))

    return fig


def compute_neighbor_importance_from_pyg(pyg_data, node_idx, hops=2):
    """Compute a heuristic neighbor importance using degree centrality within a k-hop subgraph."""
    try:
        subset, sub_edge_index, _, _ = k_hop_subgraph(
            torch.tensor([node_idx]), hops, pyg_data.edge_index, relabel_nodes=False, num_nodes=pyg_data.num_nodes
        )
        subgraph_nodes = subset.cpu().tolist()
        # Build networkx graph for centrality
        from torch_geometric.utils import to_networkx
        nx_graph = to_networkx(Data(edge_index=sub_edge_index, num_nodes=pyg_data.num_nodes), to_undirected=True)
        sub_G = nx_graph.subgraph(subgraph_nodes)
        import networkx as nx
        centrality = nx.degree_centrality(sub_G)
        node_importance = {int(n): float(s) for n, s in centrality.items() if int(n) != node_idx}
        return node_importance
    except Exception as e:
        print(f"Error computing heuristic neighbor importance: {e}")
        return {}

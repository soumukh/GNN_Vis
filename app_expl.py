"""
Explanation helpers extracted from app.py.
Contains: run_inference, explain_link_prediction, run_cpa_iv, and path utilities.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx, remove_self_loops
import networkx as nx

# Device local to this module
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def run_inference(model, data):
    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device))
    try:
        probs = torch.softmax(out, dim=-1)
    except Exception:
        probs = out
    probs_np = probs.cpu().numpy()
    preds = probs_np.argmax(axis=-1)
    confidences = probs_np.max(axis=-1)
    return {'preds': preds.tolist(), 'confidences': confidences.tolist()}


@torch.no_grad()
def explain_link_prediction(model, data, source_node, target_node):
    """Explain why a link between two nodes is predicted.
    Returns a serializable dict with link score, subgraph, and node importance heuristics.
    """
    model.eval()
    data = data.to(device)

    if not (0 <= source_node < data.num_nodes and 0 <= target_node < data.num_nodes):
        return {
            'error': 'Invalid node ID(s) for link explanation.',
            'source_node': source_node,
            'target_node': target_node,
            'link_score': 0,
            'subgraph_nodes': [],
            'subgraph_edges': [],
            'node_importance': {},
            'explanation_type': 'link_prediction'
        }

    # Get embeddings
    _, embeddings = model(data.x, data.edge_index, return_embeddings=True)

    # Create edge tensor for the specific link
    edge_tensor = torch.tensor([[source_node], [target_node]], device=device)

    # Get link prediction score
    link_score = model.predict_links(embeddings, edge_tensor)
    link_probability = torch.sigmoid(link_score).item()

    # Get k-hop subgraph around both nodes
    combined_nodes = torch.tensor([source_node, target_node])
    subset, sub_edge_index, _, _ = k_hop_subgraph(
        combined_nodes, 2, data.edge_index,
        relabel_nodes=False, num_nodes=data.num_nodes
    )

    # Create small Data object for the subgraph and convert to NetworkX
    sub_data = Data(edge_index=sub_edge_index, num_nodes=data.num_nodes)
    nx_graph = to_networkx(sub_data, to_undirected=True)

    subgraph_nodes = subset.cpu().tolist()
    node_importance = {}
    try:
        sub_G = nx_graph.subgraph(subgraph_nodes)
        centrality = nx.degree_centrality(sub_G)
        for node, score in centrality.items():
            if node != source_node and node != target_node:
                node_importance[node] = score
    except Exception as e:
        node_importance = {}

    return {
        'source_node': source_node,
        'target_node': target_node,
        'link_score': link_probability,
        'subgraph_nodes': subgraph_nodes,
        'subgraph_edges': sub_edge_index.cpu().tolist(),
        'node_importance': node_importance,
        'explanation_type': 'link_prediction'
    }


@torch.no_grad()
def run_cpa_iv(model, data, node_idx, top_k=5, max_path_len=2):
    model.eval()
    data = data.to(device)

    if not (0 <= node_idx < data.num_nodes):
        return []

    subset, sub_edge_index, _, _ = k_hop_subgraph(node_idx, max_path_len, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
    sub_node_idx = torch.where(subset == node_idx)[0].item()
    sub_data = Data(x=data.x[subset], edge_index=sub_edge_index).to(device)

    log_probs = model(sub_data.x, sub_data.edge_index)
    original_probs = torch.exp(log_probs)
    predicted_class = original_probs[sub_node_idx].argmax()
    baseline_score = original_probs[sub_node_idx, predicted_class].item()

    nx_graph = to_networkx(Data(edge_index=remove_self_loops(sub_data.edge_index)[0], num_nodes=sub_data.num_nodes), to_undirected=False)

    paths = [path for source in nx_graph.nodes() if source != sub_node_idx for path in nx.all_simple_paths(nx_graph, source=source, target=sub_node_idx, cutoff=max_path_len) if len(path) > 1]

    path_effects = []
    for path in paths[:100]:
        perturbed_edge_index = sub_data.edge_index.clone()
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            mask = ~((perturbed_edge_index[0] == u) & (perturbed_edge_index[1] == v))
            perturbed_edge_index = perturbed_edge_index[:, mask]
        perturbed_log_probs = model(sub_data.x, perturbed_edge_index)
        perturbed_score = torch.exp(perturbed_log_probs)[sub_node_idx, predicted_class].item()
        path_effects.append({'path': [subset[node].item() for node in path], 'score': baseline_score - perturbed_score})

    return sorted(path_effects, key=lambda x: x['score'], reverse=True)[:top_k]


def _path_fingerprints(paths):
    fps = set()
    for p in paths:
        nodes = p.get('nodes', [])
        if len(nodes) >= 2:
            edges = frozenset(
                tuple(sorted((nodes[i], nodes[i + 1])))
                for i in range(len(nodes) - 1)
            )
            fps.add(edges)
    return fps

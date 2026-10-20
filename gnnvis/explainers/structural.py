"""Structural explanation of a predicted link: the model's link probability
plus degree centrality of the nodes in the 2-hop neighbourhood of both ends."""

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx

from gnnvis.explainers.gnnexplainer import model_device
from gnnvis.explainers.registry import ATTRIBUTION, register_explainer
from gnnvis.tasks import LINK_PREDICTION


@torch.no_grad()
def explain_link_prediction(model, data, source_node, target_node):
    """Explain why a link between two nodes is predicted.
    Returns a serializable dict with link score, subgraph, and node importance heuristics.
    """
    model.eval()
    device = model_device(model)
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
            'explanation_type': LINK_PREDICTION
        }

    _, embeddings = model(data.x, data.edge_index, return_embeddings=True)
    edge_tensor = torch.tensor([[source_node], [target_node]], device=device)
    link_probability = torch.sigmoid(model.predict_links(embeddings, edge_tensor)).item()

    subset, sub_edge_index, _, _ = k_hop_subgraph(
        torch.tensor([source_node, target_node]), 2, data.edge_index,
        relabel_nodes=False, num_nodes=data.num_nodes
    )
    nx_graph = to_networkx(Data(edge_index=sub_edge_index, num_nodes=data.num_nodes), to_undirected=True)

    subgraph_nodes = subset.cpu().tolist()
    node_importance = {}
    try:
        centrality = nx.degree_centrality(nx_graph.subgraph(subgraph_nodes))
        for node, score in centrality.items():
            if node != source_node and node != target_node:
                node_importance[node] = score
    except Exception:
        node_importance = {}

    return {
        'source_node': source_node,
        'target_node': target_node,
        'link_score': link_probability,
        'subgraph_nodes': subgraph_nodes,
        'subgraph_edges': sub_edge_index.cpu().tolist(),
        'node_importance': node_importance,
        'explanation_type': LINK_PREDICTION
    }


@register_explainer('Structural', LINK_PREDICTION, ATTRIBUTION,
                    description='Link probability and degree centrality of the 2-hop neighbourhood.')
def explain_link(model, data, link):
    source, target = link
    return explain_link_prediction(model, data, source, target)

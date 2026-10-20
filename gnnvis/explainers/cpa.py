"""CPA-IV causal path analysis: which paths to a node (or between a link's
ends) change the prediction when they are removed. The link version, with
instrumental-variable regression, is in cpa_iv_link_prediction.py."""

import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, remove_self_loops, to_networkx

from cpa_iv_link_prediction import run_cpa_iv_link
from gnnvis.explainers.gnnexplainer import model_device
from gnnvis.explainers.registry import CAUSAL_PATHS, register_explainer
from gnnvis.tasks import LINK_PREDICTION, NODE_CLASSIFICATION


@torch.no_grad()
def run_cpa_iv(model, data, node_idx, top_k=5, max_path_len=2):
    model.eval()
    device = model_device(model)
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


register_explainer('CPA-IV', NODE_CLASSIFICATION, CAUSAL_PATHS,
                   description='Top paths into the node by drop in predicted-class probability '
                               'when the path is removed.')(run_cpa_iv)


@register_explainer('CPA-IV', LINK_PREDICTION, CAUSAL_PATHS,
                    description='Causal paths between the link ends, with IV regression.')
def explain_link_paths(model, data, link):
    source, target = link
    return run_cpa_iv_link(model, data, source, target)

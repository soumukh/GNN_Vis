"""Cytoscape elements and stylesheet for the local subgraph (node and link views)."""

import torch
from torch_geometric.utils import k_hop_subgraph
from gnnvis.ui.theme import CORRECT_COLOR, INCORRECT_COLOR


def create_subgraph_cytoscape(data, center_node_idx, predictions, class_to_color_map, explanation, cpa_data, link_predictions=None, analysis_mode='node_classification', selected_path_idx=None, color_mode='predicted', true_labels=None):

    # STRICT ENGINEERING PATCH: Ensure Link Subgraph strictly reflects selected link (u, v)

    nodes_in_subgraph = set()

    sub_edge_index = torch.empty((2, 0), dtype=torch.long)

    # 1. Determine Subgraph Nodes (Strict Isolation for Link Prediction)

    if analysis_mode == 'link_prediction':

        # Case A: Explanation-driven (if explanation matches selected link)

        if explanation and explanation.get('explanation_type') == 'link_prediction':

            nodes_in_subgraph = set(explanation['subgraph_nodes'])

            sub_edge_index = torch.tensor(explanation['subgraph_edges'])

            center_node_idx = explanation['source_node']  # Focus on source node for visualization

        # Case B: Selection-driven (Fallback or Manual Selection)

        # MUST include Source (u) and Target (v) + Neighbors

        elif link_predictions and len(link_predictions) == 1:

            # Extract source and target

            # Note: center_node_idx is passed as the source node from update_all_visuals

            u = center_node_idx 

            # Extract target from the single link prediction entry

            pred = link_predictions[0]

            if isinstance(pred, dict):

                v = pred.get('target')

            else:

                # Tuple format (edge, score) -> edge is (u, v)

                edge, _ = pred

                v = edge[1] if edge[0] == u else edge[0]

            if v is not None:

                # STRICT REQUIREMENT: Subgraph of Source (u) AND Target (v)

                # Ensure u and v are treated as integers for tensor creation

                combined_nodes = torch.tensor([int(u), int(v)], dtype=torch.long)

                # USE 1-HOP NEIGHBORHOOD to satisfy "Small Graph" constraint

                # 2-hop can be too large (hundreds of nodes). 1-hop is strictly local.

                subset, sub_edge_index, _, _ = k_hop_subgraph(

                    combined_nodes, 

                    1, # 1-hop neighborhood (Strictly Local)

                    data.edge_index, 

                    relabel_nodes=False, 

                    num_nodes=data.num_nodes

                )

                nodes_in_subgraph = set(subset.cpu().numpy())

                # Expand with CPA nodes if available

                if cpa_data and isinstance(cpa_data, dict) and 'paths' in cpa_data:

                    cpa_nodes = set()

                    for p_info in cpa_data['paths'][:10]: # Up to 10 paths

                        cpa_nodes.update(p_info['nodes'])

                    nodes_in_subgraph.update(cpa_nodes)

                    # Re-extract edges for the expanded node set

                    final_subset = torch.tensor(list(nodes_in_subgraph), dtype=torch.long)

                    mask = torch.isin(data.edge_index[0], final_subset) & torch.isin(data.edge_index[1], final_subset)

                    sub_edge_index = data.edge_index[:, mask]

            else:

                # Fallback to source only (should not happen if logic is correct)

                if 0 <= center_node_idx < data.num_nodes:

                    subset, sub_edge_index, _, _ = k_hop_subgraph(center_node_idx, 1, data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes)

                    nodes_in_subgraph = set(subset.cpu().numpy())

        else:

            # Fallback (e.g. no selection yet, just source context)

            if 0 <= center_node_idx < data.num_nodes:

                 subset, sub_edge_index, _, _ = k_hop_subgraph(center_node_idx, 1, data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes)

                 nodes_in_subgraph = set(subset.cpu().numpy())

    # 2. Node Classification Mode (k-hop subgraph around selected node)

    else:

        if center_node_idx is not None and 0 <= center_node_idx < data.num_nodes:

            subset, sub_edge_index, _, _ = k_hop_subgraph(center_node_idx, 2, data.edge_index, relabel_nodes=False, num_nodes=data.num_nodes)

            nodes_in_subgraph = set(subset.cpu().numpy())

        else:

            # No valid node selected - return empty

            return [], set()

    if not nodes_in_subgraph:

        return [], set()

    explained_edges = set()
    edge_weight = {}  # undirected edge -> max explainer mask value (drives edge width)

    if explanation and explanation.get('edge_mask'):

        edge_mask = torch.tensor(explanation['edge_mask'])

        if edge_mask.numel() == data.edge_index.shape[1]:
            ei_cpu = data.edge_index.cpu()
            for idx in torch.nonzero(edge_mask > 0).flatten().tolist():
                key = tuple(sorted((int(ei_cpu[0, idx]), int(ei_cpu[1, idx]))))
                edge_weight[key] = max(edge_weight.get(key, 0.0), float(edge_mask[idx]))

        important_indices = torch.where(edge_mask > 0.1)[0]

        if len(important_indices) > 0 and important_indices.max() < data.edge_index.shape[1]:

            for idx in important_indices:

                u, v = data.edge_index[0, idx].item(), data.edge_index[1, idx].item()

                explained_edges.add(tuple(sorted((u, v))))

    causal_edges = set()
    selected_path_edges = set()

    if cpa_data:
        # Handle Link Prediction CPA (Dict format)
        if isinstance(cpa_data, dict) and 'paths' in cpa_data:
            # All paths (general causal edges)
            for i, p_info in enumerate(cpa_data['paths']):
                path = p_info['nodes']
                is_selected = (selected_path_idx is not None and i == selected_path_idx)
                for j in range(len(path) - 1):
                    edge = tuple(sorted((path[j], path[j+1])))
                    causal_edges.add(edge)
                    if is_selected:
                        selected_path_edges.add(edge)

        # Handle Node Classification CPA (List format)
        elif isinstance(cpa_data, list):
            for i, p_info in enumerate(cpa_data):
                path = p_info['path']
                is_selected = (selected_path_idx is not None and i == selected_path_idx)
                for j in range(len(path) - 1):
                    edge = tuple(sorted((path[j], path[j+1])))
                    causal_edges.add(edge)
                    if is_selected:
                        selected_path_edges.add(edge)

    predicted_edges = set()

    if link_predictions:

        for pred in link_predictions:

            if isinstance(pred, dict):

                source, target = pred.get('source'), pred.get('target')

                score = pred.get('score', 0)

            else:

                # Handle tuple format (edge, score)

                edge, score = pred

                source, target = edge

            if score > 0.5:  # Only show high-confidence predictions

                predicted_edges.add(tuple(sorted((source, target))))

    # Prepare Counterfactual Heatmap Data
    node_cf_map = {}
    max_cf = 0.0
    if cpa_data and isinstance(cpa_data, dict):
        node_cf_map = cpa_data.get('node_counterfactuals', {})
        if node_cf_map:
            max_cf = max(node_cf_map.values()) if node_cf_map else 0.0

    cyto_elements = []

    for node_id in nodes_in_subgraph:

        pred_class_idx = int(predictions[node_id])

        classes = []

        # Highlight center node or link nodes

        if analysis_mode == 'link_prediction':

             # Case A: Explanation driven

            if explanation and explanation.get('explanation_type') == 'link_prediction':

                 if node_id == explanation['source_node'] or node_id == explanation['target_node']:

                     classes.append('center-node')

            # Case B: Selection driven

            elif link_predictions and len(link_predictions) == 1:

                # Highlight source (center_node_idx)

                if node_id == center_node_idx:

                     classes.append('center-node')

                # Highlight target

                pred = link_predictions[0]

                target_cand = None

                if isinstance(pred, dict):

                    target_cand = pred.get('target')

                else:

                    edge, _ = pred

                    target_cand = edge[1] if edge[0] == center_node_idx else edge[0]

                if target_cand is not None and node_id == target_cand:

                    classes.append('center-node')

        elif node_id == center_node_idx:

            classes.append('center-node')

        # Determine Color (Class-based OR Heatmap-based)
        node_color = class_to_color_map.get(pred_class_idx, '#808080')

        if color_mode == 'actual' and true_labels is not None and node_id < len(true_labels):
            try:
                actual_idx = int(true_labels[node_id])
                node_color = class_to_color_map.get(actual_idx, node_color)
            except Exception:
                pass
        elif color_mode == 'correctness':
            if true_labels is not None and node_id < len(true_labels):
                try:
                    actual_idx = int(true_labels[node_id])
                    is_correct = int(predictions[node_id]) == actual_idx
                    node_color = CORRECT_COLOR if is_correct else INCORRECT_COLOR
                except Exception:
                    pass
            else:
                node_color = class_to_color_map.get(pred_class_idx, node_color)
        
        # Heatmap Override (Causal Visualization)
        if node_id in node_cf_map:
            val = node_cf_map[node_id]
            if max_cf > 1e-6:
                intensity = min(1.0, max(0.0, val / max_cf))
                # Interpolate Grey (#bdc3c7 / 189,195,199) -> Red (#e74c3c / 231,76,60)
                r = int(189 + (231 - 189) * intensity)
                g = int(195 + (76 - 195) * intensity)
                b = int(199 + (60 - 199) * intensity)
                node_color = f'rgb({r},{g},{b})'
            
        cyto_elements.append({

            'data': {'id': str(node_id), 'label': str(node_id), 'color': node_color},

            'classes': ' '.join(classes)

        })

    processed_edges = set()

    for i in range(sub_edge_index.shape[1]):

        u, v = sub_edge_index[0, i].item(), sub_edge_index[1, i].item()

        edge_tuple = tuple(sorted((u, v)))

        if edge_tuple in processed_edges: continue

        processed_edges.add(edge_tuple)

        classes = []

        if edge_tuple in explained_edges: classes.append('explained-edge')

        if edge_tuple in causal_edges: classes.append('causal-path-edge')
        
        if edge_tuple in selected_path_edges: classes.append('selected-path-edge')

        if edge_tuple in predicted_edges: classes.append('predicted-edge')

        edge_data = {'source': str(u), 'target': str(v)}
        if edge_tuple in edge_weight:
            edge_data['weight'] = round(edge_weight[edge_tuple], 4)
        cyto_elements.append({'data': edge_data, 'classes': ' '.join(classes)})

    # Add predicted edges that don't exist in the graph

    if link_predictions:

        for pred in link_predictions:

            if isinstance(pred, dict):

                u, v = pred.get('source'), pred.get('target')

                score = pred.get('score', 0)

            else:

                # Handle tuple format (edge, score)

                edge, score = pred

                u, v = edge

            if score > 0.5:  # Only show high-confidence predictions

                # CRITICAL: Only add edge if BOTH nodes exist in the subgraph

                if u in nodes_in_subgraph and v in nodes_in_subgraph:

                    edge_tuple = tuple(sorted((u, v)))

                    # Check if this edge already exists

                    existing = False

                    for i in range(sub_edge_index.shape[1]):

                        existing_u, existing_v = sub_edge_index[0, i].item(), sub_edge_index[1, i].item()

                        if tuple(sorted((existing_u, existing_v))) == edge_tuple:

                            existing = True

                            break

                    if not existing:

                        cyto_elements.append({

                            'data': {'source': str(u), 'target': str(v)}, 

                            'classes': 'predicted-edge'

                        })

    return cyto_elements, nodes_in_subgraph


STYLESHEET = [

    {'selector': 'node', 'style': {'background-color': 'data(color)', 'label': 'data(label)', 'font-size': '14px', 'text-valign': 'center', 'color': 'white', 'text-outline-color': '#2c3e50', 'text-outline-width': 2, 'shape': 'ellipse', 'width': 35, 'height': 35}},

    {'selector': '.center-node', 'style': {'shape': 'star', 'width': 50, 'height': 50, 'border-width': 3, 'border-color': '#1f2937'}},

    {'selector': 'edge', 'style': {'width': 2, 'line-color': '#bdc3c7', 'curve-style': 'bezier'}},

    {'selector': '.explained-edge', 'style': {'line-color': '#1a9850', 'width': 'mapData(weight, 0, 1, 2, 10)', 'opacity': 0.9}},

    {'selector': '.causal-path-edge', 'style': {'line-color': '#e74c3c', 'width': 5, 'line-style': 'dashed'}},
    
    {'selector': '.selected-path-edge', 'style': {'line-color': '#e74c3c', 'width': 8, 'line-style': 'solid', 'z-index': 9999}},

    {'selector': '.predicted-edge', 'style': {'line-color': '#9b59b6', 'width': 4, 'line-style': 'dotted', 'opacity': 0.8}},

]

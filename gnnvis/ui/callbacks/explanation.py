"""Explanation callbacks: GNNExplainer, link explanation, CPA-IV, and the Explanation tab (subgraph, importance, CPA-IV / attention cards)."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from dash import ALL, Input, Output, State, callback, ctx, html, no_update
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph

from app_expl import explain_link_prediction, run_cpa_iv
from cpa_iv_link_prediction import run_cpa_iv_link

from gnnvis.models import model_capabilities
from gnnvis.runtime import device, get_attention_weights, get_cached_dataset_if_missing, get_cached_model, get_cached_package_if_missing, prepare_torch_data, serialize_model_use
from gnnvis.ui.components import explanation_cards, subgraph_color_key
from gnnvis.ui.figures.attention import create_attention_panel
from gnnvis.ui.figures.cpa import create_cpa_iv_panel
from gnnvis.ui.figures.importance import compute_neighbor_importance_from_pyg, get_one_hop_neighbors, plot_feature_importance, plot_neighbor_importance
from gnnvis.ui.figures.subgraph import create_subgraph_cytoscape
from gnnvis.ui.theme import CORRECT_COLOR, INCORRECT_COLOR, class_color_map, class_name, empty_figure
from gnnvis.ui import ids


@callback(

    Output(ids.EXPLANATION_STORE, 'data', allow_duplicate=True),

    Output(ids.LINK_PREDICTION_STORE, 'data', allow_duplicate=True),

    Input(ids.SELECTED_LINK_STORE, 'data'),

    State(ids.GLOBAL_RUN_BUTTON, 'n_clicks'),

    State(ids.EDITABLE_GRAPH_STORE, 'data'),

    State(ids.PRECOMPUTED_PACKAGE_STORE, 'data'), 

    State(ids.FULL_DATASET_STORE, 'data'),

    State(ids.MODEL_DROPDOWN, 'value'),

    State(ids.ANALYSIS_MODE_STORE, 'data'),

    prevent_initial_call=True

)
@serialize_model_use
def generate_link_explanation_on_select(selected_link, run_clicks, graph_data, package_meta, dataset_meta, model_type, analysis_mode):

    """Generates a link explanation automatically when a link is selected."""

    # DEBUG: log select-triggered generation attempts
    try:
        with open('link_debug.log', 'a') as _log:
            _log.write(f"GEN_LINK_SELECT run_clicks={run_clicks} selected_link_present={bool(selected_link)} analysis_mode={analysis_mode}\n")
    except Exception:
        pass

    if not run_clicks or run_clicks == 0 or not selected_link or analysis_mode != 'link_prediction':

        return no_update, no_update

    try:

        dataset = get_cached_dataset_if_missing(dataset_meta)
        package = get_cached_package_if_missing(package_meta)
        if not all([package, dataset, graph_data]):
            return no_update, no_update

        model = get_cached_model(package_meta, dataset_meta['num_features'], dataset_meta['num_classes'])
        if model is None or not getattr(model, 'enable_link_prediction', False):
            return no_update, no_update

        source_node = selected_link.get('source')

        target_node = selected_link.get('target')

        if source_node is not None and target_node is not None:

            pyg_data = prepare_torch_data(dataset_meta, graph_data, include_labels=False)

            model.eval()

            link_explanation = explain_link_prediction(model, pyg_data, source_node, target_node)

            # Create a single-item list for link prediction store to enforce (u,v) display in graph

            single_prediction = [{

                'source': source_node,

                'target': target_node,

                'score': link_explanation.get('link_score', 0.0)

            }]

            print(f"SUCCESS (on-select): Link explanation completed for ({source_node}, {target_node})")

            return link_explanation, single_prediction

        else:

            return no_update, no_update

    except Exception as e:

        print(f"ERROR in generate_link_explanation_on_select: {e}")

        return {'error': f'On-select explanation failed: {str(e)}'}, no_update


@callback(

    Output(ids.EXPLANATION_STORE, 'data', allow_duplicate=True),

    Output(ids.GLOBAL_RUN_BUTTON, 'children', allow_duplicate=True),

    Output(ids.GLOBAL_RUN_BUTTON, 'disabled', allow_duplicate=True),

    Input(ids.GLOBAL_RUN_BUTTON, 'n_clicks'),

    State(ids.SELECTED_NODE_STORE, 'data'),

    State(ids.SELECTED_LINK_STORE, 'data'),

    State({'type': ids.LP_INPUT_TYPE, 'index': ALL}, 'value'), # Capture manual link inputs

    State(ids.EDITABLE_GRAPH_STORE, 'data'),

    State(ids.PRECOMPUTED_PACKAGE_STORE, 'data'), State(ids.FULL_DATASET_STORE, 'data'),

    State(ids.CURRENT_PREDICTIONS_STORE, 'data'), State(ids.MODEL_DROPDOWN, 'value'),

    State(ids.ANALYSIS_MODE_STORE, 'data'),

    State(ids.LAYER3_TABS, 'value'),

    prevent_initial_call=True

)
@serialize_model_use
def run_explainer(n_clicks, selected_node, selected_link, lp_inputs, graph_data, package_meta, dataset_meta, predictions_data, model_type, analysis_mode, active_layer3_tab):

    try:

        if n_clicks == 0:
            return no_update, no_update, no_update

        # Link mode: run_link_prediction_unified owns the run-button reset
        if analysis_mode != 'node_classification':
            return no_update, no_update, no_update

        # From here on this callback must always re-enable the run button,
        # otherwise the clientside "Processing..." state never clears.
        if not all([package_meta, dataset_meta, graph_data, predictions_data]):
            return no_update, "RUN ANALYSIS", False

        if active_layer3_tab == 'tab-stability':
            return no_update, "RUN ANALYSIS", False

        dataset = get_cached_dataset_if_missing(dataset_meta)
        package = get_cached_package_if_missing(package_meta)
        if not dataset or not package:
            return no_update, "RUN ANALYSIS", False

        print(f"Running explainer with {model_type} model in {analysis_mode} mode...")

        model = get_cached_model(package_meta, dataset_meta['num_features'], dataset_meta['num_classes'])
        if model is None:
            print(f"ERROR: Could not instantiate {model_type} model for explanation")
            return no_update, "RUN ANALYSIS", False

        if analysis_mode == 'node_classification' and selected_node and selected_node.get('id') is not None:
            try:
                node_idx = int(selected_node['id'])
            except (TypeError, ValueError):
                return {'error': f"Invalid node ID: {selected_node['id']}"}, "RUN ANALYSIS", False
            num_nodes = dataset_meta.get('num_nodes') or len(predictions_data.get('preds', []))
            if not (0 <= node_idx < num_nodes):
                return {'error': f"Node {node_idx} is out of range (0-{num_nodes - 1})."}, "RUN ANALYSIS", False
            pyg_data = prepare_torch_data(dataset_meta, graph_data, include_labels=False)
            if pyg_data is None:
                return no_update, "RUN ANALYSIS", False
            target = torch.tensor([predictions_data['preds'][node_idx]], device=device)
            explainer = Explainer(model=model, algorithm=GNNExplainer(epochs=100), explanation_type='model', edge_mask_type='object', model_config=dict(mode='multiclass_classification', task_level='node', return_type='log_probs'))
            explanation = explainer(x=pyg_data.x, edge_index=pyg_data.edge_index, target=target, index=node_idx)

            # Calculate node importance from edge mask

            node_importance = {}

            if explanation.get('edge_mask') is not None:

                edge_mask = explanation.get('edge_mask').cpu()

                edge_index_cpu = pyg_data.edge_index.cpu()

                # Aggregate edge importance for each neighbor

                for i in range(edge_index_cpu.size(1)):

                    src, dst = edge_index_cpu[0, i].item(), edge_index_cpu[1, i].item()

                    # Find edges connected to the target node

                    if dst == node_idx or src == node_idx:

                        if i < len(edge_mask):

                            importance = abs(edge_mask[i].item())

                            neighbor = src if dst == node_idx else dst

                            if neighbor in node_importance:

                                node_importance[neighbor] = max(node_importance[neighbor], importance)

                            else:

                                node_importance[neighbor] = importance

            print(f"SUCCESS: Node explanation completed with {model_type}")

            # Try to compute feature importance via gradient*input as a fallback
            feature_imp = None
            try:
                model.eval()
                x_in = pyg_data.x.clone().detach().to(device)
                x_in.requires_grad = True
                edge_index_dev = pyg_data.edge_index.to(device)
                # Forward
                out = model(x_in, edge_index_dev)
                # predicted class for node
                pred_class = int(predictions_data['preds'][node_idx]) if predictions_data and 'preds' in predictions_data else None
                if pred_class is None:
                    pred_class = int(torch.argmax(out[node_idx]).item())
                score = out[node_idx, pred_class]
                # Backward
                model.zero_grad()
                if x_in.grad is not None:
                    x_in.grad.zero_()
                score.backward(retain_graph=False)
                grads = x_in.grad[node_idx].cpu().abs().numpy()
                vals = (grads * x_in.detach().cpu().numpy()[node_idx]).astype(float)
                # If multiplication yields zeros, fall back to abs gradients
                if not vals.any():
                    vals = grads
                feature_imp = {i: float(vals[i]) for i in range(len(vals))}
            except Exception:
                feature_imp = None

            return {
                'edge_mask': explanation.get('edge_mask').cpu().tolist() if explanation.get('edge_mask') is not None else None,
                'node_importance': node_importance,
                'feature_importance': feature_imp,
                'explanation_type': 'node_classification',
                'node_idx': node_idx,
            }, "RUN ANALYSIS", False

        return no_update, "RUN ANALYSIS", False

    except Exception as e:

        print(f"ERROR in run_explainer: {e}")

        import traceback

        traceback.print_exc()

        return {'error': f'Explanation failed with {model_type}: {str(e)}'}, "RUN ANALYSIS", False


@callback(

    Output(ids.CPA_IV_STORE, 'data'),

    Output(ids.CPA_STATUS_STORE, 'data'),

    Input(ids.RUN_CPA_IV_BUTTON, 'n_clicks'),

    State(ids.SELECTED_NODE_STORE, 'data'),

    State(ids.PRECOMPUTED_PACKAGE_STORE, 'data'),

    State(ids.FULL_DATASET_STORE, 'data'),

    State(ids.EDITABLE_GRAPH_STORE, 'data'),

    State(ids.MODEL_DROPDOWN, 'value'),

    prevent_initial_call=True

)
@serialize_model_use
def run_cpa_iv_callback(n_clicks, selected_node, package_meta, dataset_meta, graph_data, model_type):

    try:

        print(f"[CPA-IV-SIMPLE] Triggered run_cpa_iv_callback; n_clicks={n_clicks}, selected_node={selected_node}")
        if n_clicks == 0 or not selected_node:
            return no_update, no_update

        dataset = get_cached_dataset_if_missing(dataset_meta)
        package = get_cached_package_if_missing(package_meta)
        if not all([package, dataset, graph_data]):
            return no_update, no_update

        try:
            node_idx = int(selected_node['id'])
        except Exception as e:
            print(f"[CPA-IV-SIMPLE] Failed to parse selected_node id: {selected_node} -> {e}")
            return no_update, {'text': "CPA-IV failed: invalid selected node.", "type": "error"}
        print(f"[CPA-IV-SIMPLE] Resolved node_idx={node_idx}")

        model = get_cached_model(package_meta, dataset_meta['num_features'], dataset_meta['num_classes'])
        if model is None:
            return no_update, {'text': "CPA-IV failed: Could not load model.", "type": "error"}

        pyg_data = prepare_torch_data(dataset_meta, graph_data, include_labels=False)
        causal_paths = run_cpa_iv(model, pyg_data, node_idx)

        status = {'text': f"CPA-IV found {len(causal_paths)} influential paths for Node {node_idx}.", "type": "success"}
        return causal_paths, status

    except Exception as e:

        print(f"CPA-IV callback error: {e}")
        return no_update, {'text': f"CPA-IV failed: {str(e)}", "type": "error"}


@callback(
    Output(ids.CPA_IV_STORE, 'data', allow_duplicate=True),
    Output(ids.LINK_CPA_IV_STORE, 'data', allow_duplicate=True),
    Output(ids.CPA_STATUS_STORE, 'data', allow_duplicate=True),
    Input(ids.GLOBAL_RUN_BUTTON, 'n_clicks'),
    Input(ids.RUN_CPA_IV_BUTTON, 'n_clicks'),
    Input(ids.SELECTED_LINK_STORE, 'data'),
    Input(ids.SELECTED_NODE_STORE, 'data'),
    State(ids.SELECTED_LINK_STORE, 'data'),
    State(ids.PRECOMPUTED_PACKAGE_STORE, 'data'),
    State(ids.FULL_DATASET_STORE, 'data'),
    State(ids.EDITABLE_GRAPH_STORE, 'data'),
    State(ids.MODEL_DROPDOWN, 'value'),
    State(ids.ANALYSIS_MODE_STORE, 'data'),
    State(ids.LAYER3_TABS, 'value'),
    State({'type': ids.LP_INPUT_TYPE, 'index': ALL}, 'value'),
    State({'type': ids.LP_INPUT_TYPE, 'index': ALL}, 'id'),
    State(ids.LINK_SOURCE_INPUT, 'value'),
    State(ids.LINK_TARGET_INPUT, 'value'),
    prevent_initial_call=True
)
@serialize_model_use
def run_cpa_iv_callback_active(global_clicks, layer3_clicks, selected_link_trigger,
                               selected_node, selected_link_state, package_meta, dataset_meta, graph_data, model_type,
                               analysis_mode, active_layer3_tab,
                               lp_inputs, lp_input_ids,
                               layer3_source, layer3_target):
    triggered = ctx.triggered_id
    # A row click in the cell-details table sets BOTH selected-link-store and
    # selected-node-store; ctx.triggered_id may then name the node store, whose
    # branch bails out in link mode. Resolve against every input that changed.
    _changed = {t['prop_id'].split('.')[0] for t in (ctx.triggered or [])}
    if analysis_mode == 'link_prediction' and ids.SELECTED_LINK_STORE in _changed:
        triggered = ids.SELECTED_LINK_STORE
    elif analysis_mode == 'node_classification' and ids.SELECTED_NODE_STORE in _changed:
        triggered = ids.SELECTED_NODE_STORE

    if triggered == ids.GLOBAL_RUN_BUTTON:
        if (global_clicks or 0) == 0:
            return no_update, no_update, no_update
        if active_layer3_tab == 'tab-stability':
            return no_update, no_update, no_update
    elif triggered == ids.RUN_CPA_IV_BUTTON:
        if (layer3_clicks or 0) == 0:
            return no_update, no_update, no_update
    # If the selected-link-store changed, attempt to run CPA for the selected link
    elif triggered == ids.SELECTED_LINK_STORE:
        # Only run for link prediction mode and when a valid selection exists
        if analysis_mode != 'link_prediction':
            return no_update, no_update, no_update
        if not selected_link_trigger:
            return no_update, no_update, no_update
        # Proceed — treat selection as the trigger
    # If the selected-node-store changed, attempt to run CPA for the selected node
    elif triggered == ids.SELECTED_NODE_STORE:
        # Only run for node classification mode and when a valid selection exists
        if analysis_mode != 'node_classification':
            return no_update, no_update, no_update
        if not selected_node:
            return no_update, no_update, no_update
        # Proceed — treat selection as the trigger
    else:
        return no_update, no_update, no_update

    dataset = get_cached_dataset_if_missing(dataset_meta)
    package = get_cached_package_if_missing(package_meta)
    if not all([package, dataset, graph_data]):
        return no_update, no_update, {'text': "Load a dataset and model first.", "type": "error"}

    print(f"[CPA-IV] Triggered by {triggered}. Mode: {analysis_mode}")
    print(f"[CPA-IV] layer3_source={layer3_source}, selected_node={selected_node}")
    # selected_link may come from either State or the selected-link input; prefer trigger
    sel_link = selected_link_trigger if triggered == ids.SELECTED_LINK_STORE else selected_link_state
    print(f"[CPA-IV] selected_link (raw): {sel_link}")

    model = get_cached_model(package_meta, dataset_meta['num_features'], dataset_meta['num_classes'])
    if model is None:
        return no_update, no_update, {'text': "CPA-IV failed: could not load model.", "type": "error"}

    pyg_data = prepare_torch_data(dataset_meta, graph_data, include_labels=False)

    # ── Mode A: Node Classification ──────────────────────────────────────────
    if analysis_mode == 'node_classification':
        # Layer 3 source input overrides selected-node-store when present
        node_idx = None
        if layer3_source is not None:
            try: node_idx = int(layer3_source)
            except (ValueError, TypeError): pass
        if node_idx is None and selected_node:
            try: node_idx = int(selected_node['id'])
            except (ValueError, TypeError, KeyError): pass
        if node_idx is None:
            # Silent no-op when no node is selected in node-classification mode
            return no_update, no_update, no_update
        try:
            if int(node_idx) < 0:
                # Treat negative indices (e.g. -1 used for 'no selection') as no selection
                return no_update, no_update, no_update
        except Exception:
            # If we can't interpret node_idx as int, fall through to error handling below
            pass

        causal_paths = run_cpa_iv(model, pyg_data, node_idx)
        status = {'text': f"CPA-IV: {len(causal_paths)} influential paths for Node {node_idx}.", "type": "success"}
        return causal_paths, no_update, status


    

    # ── Mode B: Link Prediction ───────────────────────────────────────────────
    elif analysis_mode == 'link_prediction':
        u, v = None, None

        # If the callback was triggered by a selected-link change, prefer that selection
        if triggered == ids.SELECTED_LINK_STORE and sel_link and isinstance(sel_link, dict):
            try:
                u = int(sel_link.get('source'))
                v = int(sel_link.get('target'))
                print(f"[CPA-IV] Triggered selection used: {u} → {v}")
            except Exception as e:
                print(f"[CPA-IV] Failed to parse triggered selection: {sel_link} -> {e}")

        # Priority 1: Layer 3 dedicated inputs (causal tab) — only use if no triggered selection
        if (u is None or v is None) and layer3_source is not None and layer3_target is not None:
            try:
                u, v = int(layer3_source), int(layer3_target)
                print(f"[CPA-IV] Layer 3 inputs: {u} → {v}")
            except (ValueError, TypeError):
                u, v = None, None

        # Priority 2: Layer 1 lp-input pattern components (GNN/non-causal LP mode)
        if (u is None or v is None) and lp_inputs and lp_input_ids:
            manual_source, manual_target = None, None
            for val, id_dict in zip(lp_inputs, lp_input_ids):
                if id_dict['index'] == 'source-active':   manual_source = val
                elif id_dict['index'] == 'target-active': manual_target = val
            if manual_source is not None and manual_target is not None:
                try:
                    u, v = int(manual_source), int(manual_target)
                    print(f"[CPA-IV] Layer 1 lp-inputs: {u} → {v}")
                except (ValueError, TypeError):
                    u, v = None, None

        # Priority 3: Graph-click selected link
        sel = sel_link if sel_link is not None else None
        if (u is None or v is None) and sel and isinstance(sel, dict):
            s = sel.get('source')
            t = sel.get('target')
            try:
                if s is not None and t is not None:
                    u = int(s)
                    v = int(t)
                    print(f"[CPA-IV] Selected link parsed: {u} → {v}")
            except Exception as e:
                print(f"[CPA-IV] Failed to parse selected_link source/target: {sel} -> {e}")
                u, v = None, None

        if u is None or v is None:
            return no_update, no_update, {
                'text': "CPA-IV: enter Source & Target node IDs in the Explanation panel.",
                "type": "error"
            }
        if not (0 <= u < dataset['num_nodes'] and 0 <= v < dataset['num_nodes']):
            return no_update, no_update, {'text': f"Node IDs {u}, {v} out of range (max {dataset['num_nodes']-1}).", "type": "error"}

        print(f"[CPA-IV] Running link analysis {u} → {v}…")
        cpa_result = run_cpa_iv_link(model, pyg_data, u, v)

        if 'error' in cpa_result:
            return no_update, no_update, {'text': f"CPA-IV Error: {cpa_result['error']}", "type": "error"}

        num_paths = len(cpa_result.get('paths', []))
        print(f"[CPA-IV] Link success: {num_paths} paths, verdict: {cpa_result.get('verdict')}")
        status = {'text': f"CPA-IV: {num_paths} causal paths for link {u}→{v}.", "type": "success"}
        return no_update, cpa_result, status

    return no_update, no_update, no_update


@callback(

    Output(ids.GRAPH_VIEW, 'elements'),

    Output(ids.NEIGHBOR_IMPORTANCE_VIEW, 'figure'),

    Output(ids.FEATURE_IMPORTANCE_VIEW, 'figure'),

    Output(ids.SUBGRAPH_LEGEND_INLINE, 'children'),

    Output(ids.LAYER3_RIGHT_HEADER, 'children'),

    Output(ids.GNN_CAUSAL_EXPLANATION_VIEW, 'children'),

    Output(ids.GNN_CAUSAL_EXPLANATION_VIEW, 'style'),

    Input(ids.SELECTED_LINK_STORE, 'data'),

    Input(ids.SELECTED_NODE_STORE, 'data'),

    Input(ids.EXPLANATION_STORE, 'data'),

    Input(ids.CPA_IV_STORE, 'data'),
    
    Input(ids.LINK_CPA_IV_STORE, 'data'),
    
    Input(ids.SELECTED_PATH_STORE, 'data'),

    Input(ids.ANALYSIS_MODE_STORE, 'data'),

    Input(ids.SUBGRAPH_COLOR_MODE, 'value'),

    State(ids.CURRENT_PREDICTIONS_STORE, 'data'),

    State(ids.FULL_DATASET_STORE, 'data'),

    State(ids.EDITABLE_GRAPH_STORE, 'data'),

    State(ids.PRECOMPUTED_PACKAGE_STORE, 'data'),

    State(ids.MODEL_DROPDOWN, 'value'),

    State(ids.LAYER3_TABS, 'value'),

    prevent_initial_call=True

)

def update_bottom_layer(selected_link, selected_node, explanation, cpa_data, link_cpa_data, selected_path, analysis_mode, subgraph_color_mode, predictions_data, dataset, graph_data, package_meta, model_type, active_layer3_tab):
    try:
        empty_fig = empty_figure()
        subgraph_color_mode = subgraph_color_mode or 'actual'

        # Visibility Rule: Only show if specific selection exists
        visible = False
        target_node_idx = None

        if analysis_mode == 'link_prediction':
            if selected_link and isinstance(selected_link, dict):
                visible = True
                target_node_idx = selected_link['source']  # Focus on source

        elif analysis_mode == 'node_classification':
            # Automatically use FULL graph for Node Task (always visible)
            visible = True
            if selected_node and isinstance(selected_node, dict):
                target_node_idx = int(selected_node['id'])
            else:
                target_node_idx = -1  # Indicate no selection

        # Ensure we have the full cached dataset (may be metadata passed in state)
        dataset = get_cached_dataset_if_missing(dataset)
        if not visible or not dataset or not graph_data or not predictions_data:
            hint = ("Pick a predicted link from the list above to inspect it."
                    if analysis_mode == 'link_prediction' else "Load a model to begin.")
            return [], empty_fig, empty_fig, html.Div(hint, className='empty-hint'), "Details", None, {'display': 'none'}

        preds = np.array(predictions_data['preds'])
        num_classes = dataset['num_classes']
        dataset_name = dataset.get('name')
        class_to_color_map = class_color_map(num_classes)
        # Class colours are already keyed in the node-embeddings view; only the
        # two-colour correctness key is shown here
        color_key = (subgraph_color_key(subgraph_color_mode, dataset_name, num_classes)
                     if subgraph_color_mode == 'correctness' else None)
        true_labels = dataset.get('y') if isinstance(dataset, dict) else None
        yvals = true_labels

        pyg_data = prepare_torch_data(dataset, graph_data, include_labels=False)
        if pyg_data is None:
            return [], empty_fig, empty_fig, "Unable to construct graph data", "Details", None, {'display': 'none'}

        # Path Selection
        selected_path_idx = selected_path.get('index') if selected_path else None

        # --- LOGIC BRANCHING ---
        # A. LINK PREDICTION MODE (STRICT LOCAL VIEW)
        if analysis_mode == 'link_prediction':
            u = selected_link['source']
            v = selected_link['target']
            # Use link_cpa_data for path info
            active_cpa_data = link_cpa_data

            # 1. CALCULATE INDUCED SUBGRAPH (Single Source of Truth)
            subset, sub_edge_index, _, _ = k_hop_subgraph(
                torch.tensor([u, v]),
                1,  # 1-hop neighborhood
                pyg_data.edge_index,
                relabel_nodes=False,
                num_nodes=dataset['num_nodes']
            )

            subgraph_nodes = subset.tolist()
            subgraph_edges = sub_edge_index.tolist()

            # 2. PREPARE DATA OBJECT FOR VIZ
            strict_explanation = {
                'explanation_type': 'link_prediction',
                'source_node': u,
                'target_node': v,
                'subgraph_nodes': subgraph_nodes,
                'subgraph_edges': subgraph_edges,
                'link_score': 0,  # Placeholder
                'node_importance': {}
            }

            # Check if we have a real explanation to augment our strict data
            real_score = 0.0
            if explanation and explanation.get('explanation_type') == 'link_prediction':
                if explanation.get('source_node') == u and explanation.get('target_node') == v:
                    strict_explanation['node_importance'] = explanation.get('node_importance', {})
                    real_score = explanation.get('link_score', 0.0)
                    strict_explanation['link_score'] = real_score

            # 3. GENERATE NEIGHBORHOOD DETAILS (Right Panel)
            # Hide the right-panel header when in link-detail view (we show the list elsewhere)
            header_text = ""

            # Determine prediction class (Edge vs No Edge) based on threshold
            predicted_label = 1 if real_score > 0.5 else 0
            pred_text = "Edge Predicted" if predicted_label == 1 else "No Edge"
            pred_color = CORRECT_COLOR if predicted_label == 1 else INCORRECT_COLOR

            # Ground Truth Check
            is_existing_link = False
            if 'original_edge_index' in dataset:
                edges = torch.tensor(dataset['original_edge_index'], dtype=torch.long)
            else:
                edges = pyg_data.edge_index

            src_edges, dst_edges = edges[0], edges[1]
            mask = ((src_edges == u) & (dst_edges == v)) | ((src_edges == v) & (dst_edges == u))
            if mask.any():
                is_existing_link = True

            # Override from CPA if available
            if active_cpa_data and isinstance(active_cpa_data, dict):
                is_existing_link = active_cpa_data.get('is_existing_link', is_existing_link)

            ground_truth_label = 1 if is_existing_link else 0
            is_correct = (predicted_label == ground_truth_label)
            correctness_label = "(True)" if is_correct else "(False)"
            badge_color = CORRECT_COLOR if is_correct else INCORRECT_COLOR

            # Additional Path Info
            path_info_div = html.Div()
            if selected_path_idx is not None and active_cpa_data and 'paths' in active_cpa_data:
                try:
                    p = active_cpa_data['paths'][selected_path_idx]
                    path_str = ' → '.join(map(str, p.get('nodes', [])))
                    path_info_div = html.Div([
                        html.Span("Highlighted Path:", style={'fontWeight': 'bold', 'color': '#e74c3c'}),
                        html.Div(path_str, style={'fontFamily': 'monospace', 'fontSize': '12px', 'marginTop': '3px'})
                    ], style={'marginTop': '10px', 'padding': '5px', 'border': '1px dashed #e74c3c', 'backgroundColor': '#fdedec'})
                except Exception:
                    pass

            mode_label = 'Predicted Class'
            if subgraph_color_mode == 'actual':
                mode_label = 'Actual Class'
            elif subgraph_color_mode == 'correctness':
                mode_label = 'Correctness'

            legend_content = html.Div([
                html.Div([
                    html.Span("Color Mode:", style={'fontWeight': 'bold', 'color': '#333'}),
                    html.Span(mode_label, style={'marginLeft': '5px', 'fontFamily': 'monospace', 'fontSize': '14px'})
                ], style={'marginBottom': '10px'}),
                html.Div([
                    html.Span("Selected Link:", style={'fontWeight': 'bold', 'color': '#333'}),
                    html.Span(f" {u} → {v}", style={'marginLeft': '5px', 'fontFamily': 'monospace', 'fontSize': '14px'})
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Span("Subgraph Size:", style={'fontWeight': 'bold', 'color': '#333'}),
                    html.Span(f" {len(subgraph_nodes)} nodes", style={'marginLeft': '5px'})
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Span("Link Score:", style={'fontWeight': 'bold', 'color': '#333'}),
                    html.Span(f" {real_score:.4f}", style={'marginLeft': '5px', 'fontWeight': 'bold'})
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Span("Prediction:", style={'fontWeight': 'bold', 'color': '#333'}),
                    html.Span(f" {pred_text}", style={'marginLeft': '5px', 'color': pred_color, 'fontWeight': 'bold'}),
                    html.Span(correctness_label, style={
                        'marginLeft': '8px',
                        'backgroundColor': badge_color,
                        'color': 'white',
                        'padding': '2px 8px',
                        'borderRadius': '10px',
                        'fontSize': '11px',
                        'fontWeight': 'bold',
                        'display': 'inline-block'
                    })
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Span("View Radius:", style={'fontWeight': 'bold', 'color': '#333'}),
                    html.Span(f" 1-hop", style={'marginLeft': '5px', 'color': '#666'})
                ], style={'marginBottom': '8px'}),
                path_info_div,
                color_key,
            ], style={'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px', 'border': '1px solid #dee2e6'})

            # 4. RENDER GRAPH
            layer3_link_predictions = [{'source': u, 'target': v, 'score': real_score}]
            graph_elements, _ = create_subgraph_cytoscape(
                pyg_data, u, preds, class_to_color_map,
                strict_explanation, active_cpa_data, layer3_link_predictions, analysis_mode, selected_path_idx,
                color_mode=subgraph_color_mode,
                true_labels=true_labels
            )

            # 5. IMPORTANCE PLOT - show neighbor importance and feature importance in link-detail view
            # Compute neighbor importance for the induced subgraph (exclude source/target)
            try:
                node_imp = {}
                from torch_geometric.utils import to_networkx
                import networkx as nx
                # sub_edge_index is already a torch tensor from k_hop_subgraph
                nx_graph = to_networkx(Data(edge_index=sub_edge_index, num_nodes=pyg_data.num_nodes), to_undirected=True)
                sub_G = nx_graph.subgraph(subgraph_nodes)
                centrality = nx.degree_centrality(sub_G)
                # Exclude the source and target nodes from neighbor importance
                for n, s in centrality.items():
                    if int(n) not in (int(u), int(v)):
                        node_imp[int(n)] = float(s)
                if node_imp:
                    neighbor_fig = plot_neighbor_importance(node_imp, u, preds=preds, y_labels=true_labels,
                                                            dataset_name=dataset_name)
                    try:
                        neighbor_fig.update_layout(title=f'Top Neighbor Importance')
                    except Exception:
                        pass
                else:
                    neighbor_fig = empty_fig
            except Exception:
                neighbor_fig = empty_fig

            # Feature importance for link-prediction: prefer CPA feature_analysis, else fallback to raw features
            try:
                feature_fig = empty_fig
                print(f"[LINK-FEAT] link {u}->{v} active_cpa_data? {bool(active_cpa_data)}")
                print(f"[LINK-FEAT] pyg_data.x present? {hasattr(pyg_data,'x') and pyg_data.x is not None}")
                # 1) CPA-driven feature analysis
                if active_cpa_data and isinstance(active_cpa_data, dict) and 'feature_analysis' in active_cpa_data:
                    fa = active_cpa_data['feature_analysis']
                    print(f"[LINK-FEAT] CPA feature_analysis keys: {list(fa.keys()) if isinstance(fa, dict) else 'n/a'}")
                    rows = []
                    for item in fa.get('source_importance', []) or []:
                        rows.append({'Feature': f"{item.get('index')}", 'Score': float(item.get('score', 0)), 'Type': 'Source (u)'} )
                    for item in fa.get('target_importance', []) or []:
                        rows.append({'Feature': f"{item.get('index')}", 'Score': float(item.get('score', 0)), 'Type': 'Target (v)'} )
                    df_fa = pd.DataFrame(rows)
                    print(f"[LINK-FEAT] CPA df rows: {len(df_fa)}")
                    if not df_fa.empty:
                        # keep top features by absolute score across both
                        df_agg = df_fa.groupby('Feature', as_index=False).agg({'Score': lambda s: float(np.abs(s).sum())})
                        top_feats = df_agg.sort_values('Score', ascending=False).head(10)['Feature'].tolist()
                        df_plot = df_fa[df_fa['Feature'].isin(top_feats)]
                        print(f"[LINK-FEAT] CPA df_plot rows: {len(df_plot)}; head: {df_plot.head().to_dict('list')}")
                        feature_fig = px.bar(df_plot, x='Feature', y='Score', color='Type', barmode='group',
                                            title=f'Top Feature Importance for Link {u} → {v}')
                        feature_fig.update_layout(xaxis_type='category', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        print(f"[LINK-FEAT] created CPA feature_fig with {len(feature_fig.data)} traces")
                        for ti, tr in enumerate(feature_fig.data):
                            print(f"[LINK-FEAT] trace {ti}: name={getattr(tr,'name',None)}, x_sample={tr.x[:5] if hasattr(tr,'x') else None}, y_sample={tr.y[:5] if hasattr(tr,'y') else None}")
                else:
                    # 2) Fallback: use raw node features
                    if hasattr(pyg_data, 'x') and pyg_data.x is not None:
                        xu = pyg_data.x[int(u)]
                        xv = pyg_data.x[int(v)]
                        print(f"[LINK-FEAT] xu shape: {getattr(xu,'shape', None)}, xv shape: {getattr(xv,'shape', None)}")
                        if hasattr(xu, 'detach'):
                            xu_arr = xu.detach().cpu().numpy()
                            xv_arr = xv.detach().cpu().numpy()
                        else:
                            xu_arr = np.array(xu)
                            xv_arr = np.array(xv)
                        print(f"[LINK-FEAT] xu_arr size: {xu_arr.size}, xv_arr size: {xv_arr.size}")
                        if xu_arr.size > 0 and xv_arr.size > 0:
                            sum_abs = np.abs(xu_arr) + np.abs(xv_arr)
                            idxs = np.argsort(sum_abs)[-10:][::-1]
                            print(f"[LINK-FEAT] selected feature idxs: {idxs}")
                            rows = []
                            for i in idxs:
                                rows.append({'Feature': str(int(i)), 'Score': float(xu_arr[int(i)]), 'Type': 'Source (u)'} )
                                rows.append({'Feature': str(int(i)), 'Score': float(xv_arr[int(i)]), 'Type': 'Target (v)'} )
                            df_plot = pd.DataFrame(rows)
                            print(f"[LINK-FEAT] fallback df_plot rows: {len(df_plot)}; head: {df_plot.head().to_dict('list')}")
                            feature_fig = px.bar(df_plot, x='Feature', y='Score', color='Type', barmode='group',
                                                title=f'Top Features for Link {u} → {v}')
                            feature_fig.update_layout(xaxis_type='category', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                            print(f"[LINK-FEAT] created fallback feature_fig with {len(feature_fig.data)} traces")
                            for ti, tr in enumerate(feature_fig.data):
                                print(f"[LINK-FEAT] trace {ti}: name={getattr(tr,'name',None)}, x_sample={tr.x[:5] if hasattr(tr,'x') else None}, y_sample={tr.y[:5] if hasattr(tr,'y') else None}")
                # If still empty, keep empty_fig
            except Exception:
                feature_fig = empty_fig
            causal_content = None
            attention_content = None
            causal_style = {'display': 'none'}
            if active_cpa_data:
                causal_content = create_cpa_iv_panel(active_cpa_data, f"{u} \u2192 {v}")
            if model_capabilities(model_type)['attention']:
                attention_content = create_attention_panel(
                    get_attention_weights(package_meta),
                    f"{u} \u2192 {v}"
                )
            if causal_content or attention_content:
                causal_content = explanation_cards(causal_content, attention_content)
                causal_style = {'display': 'block', 'marginTop': '16px'}
            return graph_elements, neighbor_fig, feature_fig, legend_content, header_text, causal_content, causal_style

        # B. NODE CLASSIFICATION MODE (Standard)
        else:
            header_text = "Node Neighborhood & Importance"
            # Never show one node's explanation on another node's subgraph
            if isinstance(explanation, dict) and explanation.get('node_idx') is not None \
                    and int(explanation['node_idx']) != int(target_node_idx):
                explanation = None
            if target_node_idx != -1:
                # Determine actual class if available in dataset
                actual_class = None
                try:
                    if yvals is None:
                        if isinstance(dataset, dict) and 'y' in dataset:
                            yvals = dataset.get('y')
                        elif hasattr(dataset, 'y'):
                            yvals = dataset.y
                    if yvals is not None and len(yvals) > target_node_idx:
                        val = yvals[target_node_idx]
                        if hasattr(val, 'item'):
                            actual_class = int(val.item())
                        else:
                            actual_class = int(val)
                except Exception:
                    actual_class = None
                    yvals = None

                pred_cls = int(preds[target_node_idx])
                is_correct = actual_class is not None and pred_cls == actual_class
                conf = None
                try:
                    conf = float(predictions_data.get('confidences', [])[target_node_idx])
                except Exception:
                    pass
                conf_text = f"  ({conf:.0%})" if conf is not None else ""
                legend_content = html.Div([
                    html.Div([
                        html.Span(f"Node {target_node_idx}", className='legend-title'),
                        html.Span("correct" if is_correct else "misclassified",
                                  className='pill', style={'backgroundColor': CORRECT_COLOR if is_correct else INCORRECT_COLOR})
                        if actual_class is not None else None,
                    ], className='legend-row'),
                    html.Div([html.Span("Predicted", className='legend-label'),
                              html.Span(class_name(dataset_name, pred_cls) + conf_text)], className='legend-row'),
                    html.Div([html.Span("Actual", className='legend-label'),
                              html.Span(class_name(dataset_name, actual_class) if actual_class is not None else 'N/A')],
                             className='legend-row'),
                    color_key,
                ], className='legend-card')
            else:
                legend_content = html.Div([
                    html.Div("No node selected", className='legend-title'),
                    html.Div("Click a point in the embedding plot or the full graph, or type a node ID "
                             "under Selection, then press RUN ANALYSIS to explain it.", className='empty-hint'),
                ], className='legend-card')

            # Standard Graph Rendering
            graph_elements, _ = create_subgraph_cytoscape(
                pyg_data, target_node_idx, preds, class_to_color_map,
                explanation, cpa_data, None, analysis_mode, selected_path_idx,
                color_mode=subgraph_color_mode,
                true_labels=yvals
            )

            # Importance Plot - only when a node is selected
            if target_node_idx == -1:
                neighbor_fig = empty_figure("Select a node to list its neighbours")
            else:
                node_imp = None
                if explanation and isinstance(explanation, dict) and 'node_importance' in explanation:
                    # Store round-trips through JSON, so keys arrive as strings
                    node_imp = {int(k): float(v) for k, v in (explanation.get('node_importance') or {}).items()}

                if not node_imp or len(node_imp) == 0:
                    try:
                        node_imp = compute_neighbor_importance_from_pyg(pyg_data, target_node_idx, hops=2)
                    except Exception:
                        node_imp = {}

                one_hop_neighbors = get_one_hop_neighbors(pyg_data, target_node_idx)
                if node_imp and len(node_imp) > 0 and one_hop_neighbors:
                    neighbor_imp = {n: node_imp.get(n, 0.0) for n in one_hop_neighbors}
                    neighbor_fig = plot_neighbor_importance(
                        neighbor_imp,
                        target_node_idx,
                        preds=preds,
                        y_labels=yvals,
                        one_hop_neighbors=one_hop_neighbors,
                        dataset_name=dataset_name,
                    )
                else:
                    neighbor_fig = plot_neighbor_importance(
                        {},
                        target_node_idx,
                        preds=preds,
                        y_labels=yvals,
                        one_hop_neighbors=one_hop_neighbors,
                        dataset_name=dataset_name,
                    ) if one_hop_neighbors else empty_figure(f"Node {target_node_idx} has no neighbours")

            # CPA Panel is shown in the merged Explanation frame.
            cpa_panel = create_cpa_iv_panel(cpa_data, target_node_idx) if cpa_data else None
            # Feature importance (from explanation if present)
            feature_fig = empty_fig
            feature_imp = None
            if explanation and isinstance(explanation, dict) and 'feature_importance' in explanation:
                feature_imp = explanation.get('feature_importance') or {}

            if feature_imp and len(feature_imp) > 0:
                try:
                    feature_fig = plot_feature_importance(feature_imp, target_node_idx)
                except Exception:
                    feature_fig = empty_fig
            else:
                # Fallback: show top-5 features from the node's raw feature vector
                try:
                    if target_node_idx != -1 and hasattr(pyg_data, 'x') and pyg_data.x is not None:
                        node_feat = pyg_data.x[target_node_idx]
                        # convert to numpy
                        if hasattr(node_feat, 'detach'):
                            arr = node_feat.detach().cpu().numpy()
                        else:
                            arr = np.array(node_feat)
                        if arr.size > 0:
                            # use absolute magnitude to pick top features
                            idxs = np.argsort(np.abs(arr))[-10:][::-1]
                            feat_dict = {int(i): float(arr[int(i)]) for i in idxs}
                            # Not an explanation yet: label honestly as raw input values
                            feature_fig = plot_feature_importance(
                                feat_dict, target_node_idx, top_k=10, value_label='Raw value',
                                title=f'Raw feature values, node {target_node_idx}')
                        else:
                            feature_fig = empty_fig
                    else:
                        feature_fig = empty_fig
                except Exception:
                    feature_fig = empty_fig

            causal_content = None
            attention_content = None
            causal_style = {'display': 'none'}
            if cpa_data:
                causal_content = create_cpa_iv_panel(cpa_data, target_node_idx)
            if model_capabilities(model_type)['attention']:
                attention_content = create_attention_panel(
                    get_attention_weights(package_meta),
                    target_node_idx if target_node_idx != -1 else 'selected node'
                )
            if causal_content or attention_content:
                causal_content = explanation_cards(causal_content, attention_content)
                causal_style = {'display': 'block', 'marginTop': '16px'}
            return graph_elements, neighbor_fig, feature_fig, legend_content, header_text, causal_content, causal_style

    except Exception as e:
        print(f"Error in update_bottom_layer: {e}")
        import traceback
        traceback.print_exc()
        return [], go.Figure(), go.Figure(), "Error", "Error", None, {'display': 'none'}

"""Link-prediction callbacks: typed links, link list, and the held-out link evaluation."""

import plotly.graph_objects as go
import torch
from dash import ALL, Input, Output, State, callback, ctx, html, no_update

from link_prediction_utils import create_link_prediction_candidates, get_top_k_predictions

from gnnvis.runtime import device, get_cached_dataset, get_cached_dataset_if_missing, get_cached_model, get_cached_package_if_missing, link_test_split, prepare_torch_data, serialize_model_use
from gnnvis.ui.components import cell_details_placeholder, metric_tiles
from gnnvis.ui.figures.confusion import link_confusion_figure
from gnnvis.ui import ids
from gnnvis.tasks import LINK_PREDICTION, get_task


@callback(

    Output(ids.LINK_PREDICTION_STORE, 'data', allow_duplicate=True),

    Output(ids.LINK_STATUS_STORE, 'data'),

    Output(ids.GLOBAL_RUN_BUTTON, 'children', allow_duplicate=True),

    Output(ids.GLOBAL_RUN_BUTTON, 'disabled', allow_duplicate=True),

    Input(ids.PREDICT_LINKS_BUTTON, 'n_clicks'),

    Input(ids.GLOBAL_RUN_BUTTON, 'n_clicks'),

    State(ids.SELECTED_NODE_STORE, 'data'),

    State(ids.SELECTED_LINK_STORE, 'data'),

    State({'type': ids.LP_INPUT_TYPE, 'index': ALL}, 'value'),

    State({'type': ids.LP_INPUT_TYPE, 'index': ALL}, 'id'),

    State(ids.LINK_SOURCE_INPUT, 'value'),

    State(ids.PRECOMPUTED_PACKAGE_STORE, 'data'),

    State(ids.FULL_DATASET_STORE, 'data'),

    State(ids.EDITABLE_GRAPH_STORE, 'data'),

    State(ids.MODEL_DROPDOWN, 'value'),

    State(ids.ANALYSIS_MODE_STORE, 'data'),

    prevent_initial_call=True

)
@serialize_model_use
def run_link_prediction_unified(predict_clicks, global_clicks,

                                selected_node, selected_link, lp_inputs, lp_input_ids, link_source_input,

                                package_meta, dataset_meta, graph_data, model_type, analysis_mode):

    try:
        # DEBUG: record entry and basic state for reproduction tracing
        try:
            with open('link_debug.log', 'a') as _log:
                _log.write(f"RUN_LINK_UNIFIED triggered_id={ctx.triggered_id} selected_node={selected_node} selected_link_present={bool(selected_link)} link_source_input={link_source_input} analysis_mode={analysis_mode}\n")
        except Exception:
            pass

        triggered_id = ctx.triggered_id

        if not triggered_id:

            return no_update, no_update, no_update, no_update

        dataset = get_cached_dataset_if_missing(dataset_meta)
        package = get_cached_package_if_missing(package_meta)
        if not dataset:

            return no_update, {'text': 'Dataset not yet loaded.', 'type': 'error'}, "RUN ANALYSIS", False

        n_clicks = global_clicks if triggered_id == ids.GLOBAL_RUN_BUTTON else predict_clicks

        if not n_clicks or n_clicks == 0:

            return no_update, no_update, no_update, no_update

        manual_source = None

        manual_target = None

        if lp_inputs and lp_input_ids:

            for val, id_dict in zip(lp_inputs, lp_input_ids):

                if id_dict['index'] == 'source-active':

                    manual_source = val

                elif id_dict['index'] == 'target-active':

                    manual_target = val

                elif id_dict['index'] == 'pred-node' and manual_source is None:

                    manual_source = val

        if triggered_id == ids.GLOBAL_RUN_BUTTON:

            if not get_task(analysis_mode).is_link:

                return no_update, no_update, no_update, no_update

            if manual_source is not None and manual_target is not None:

                try:

                    u, v = int(manual_source), int(manual_target)

                    if not (0 <= u < dataset['num_nodes'] and 0 <= v < dataset['num_nodes']):
                         return no_update, {'text': f"Node IDs must be between 0 and {dataset['num_nodes'] - 1} (got {u}, {v}).", "type": "error"}, "RUN ANALYSIS", False
                    if u == v:
                         return no_update, {'text': "Source and target must be different nodes.", "type": "error"}, "RUN ANALYSIS", False

                    model = get_cached_model(package_meta, dataset_meta['num_features'], dataset_meta['num_classes'])

                    if model is None:
                        return no_update, {'text': "Model load failed.", "type": "error"}, "RUN ANALYSIS", False

                    pyg_data = prepare_torch_data(dataset_meta, graph_data, include_labels=False)

                    with torch.no_grad():

                        _, embeddings = model(pyg_data.x, pyg_data.edge_index, return_embeddings=True)
                        edge_tensor = torch.tensor([[u], [v]], device=device)
                        link_score = model.predict_links(embeddings, edge_tensor)
                        prob = torch.sigmoid(link_score).item()

                    prediction_result = [{'source': u, 'target': v, 'score': prob}]

                    status = {'text': f"Predicted link {u} → {v} with score {prob:.4f}", "type": "success"}

                    return prediction_result, status, "RUN ANALYSIS", False

                except ValueError:
                    return no_update, {'text': "Invalid node input (must be integers).", "type": "error"}, "RUN ANALYSIS", False

            # Link mode needs an explicit pair: both source and target are required
            return no_update, {'text': "Enter both a source and a target node ID, then press RUN ANALYSIS.", "type": "error"}, "RUN ANALYSIS", False

        node_idx = None

        if manual_source is not None:

            node_idx = int(manual_source)

        elif selected_node and selected_node.get('id') is not None:

            node_idx = int(selected_node['id'])

        elif selected_link and selected_link.get('source') is not None:

            node_idx = int(selected_link['source'])

        elif link_source_input is not None:

            node_idx = int(link_source_input)

        if node_idx is None:
            # If in link prediction mode and Run pressed without explicit source,
            # fallback to node 0 and continue so the UI shows the Link CM instead
            # of silently doing nothing.
            try:
                if get_task(analysis_mode).is_link:
                    node_idx = 0
                    print("[LINK-PRED] No source provided on Run — defaulting to node 0")
                else:
                    return no_update, {'text': "Please select a source node for link prediction.", "type": "error"}, "RUN ANALYSIS", False
            except Exception:
                return no_update, {'text': "Please select a source node for link prediction.", "type": "error"}, "RUN ANALYSIS", False

        if not isinstance(node_idx, int) or node_idx < 0 or node_idx >= dataset['num_nodes']:

            return no_update, {'text': f"Node {node_idx} is out of range for the current graph.", "type": "error"}, "RUN ANALYSIS", False

        model = get_cached_model(package_meta, dataset_meta['num_features'], dataset_meta['num_classes'])

        if model is None:
            return no_update, {'text': "Link prediction failed: Could not load model.", "type": "error"}, "RUN ANALYSIS", False

        if not hasattr(model, 'enable_link_prediction') or not model.enable_link_prediction:

            return no_update, {'text': "Link prediction not enabled for this model.", "type": "error"}, "RUN ANALYSIS", False

        pyg_data = prepare_torch_data(dataset_meta, graph_data, include_labels=False)

        with torch.no_grad():

            _, embeddings = model(pyg_data.x, pyg_data.edge_index, return_embeddings=True)

        candidates = create_link_prediction_candidates(pyg_data, target_node=node_idx, exclude_existing=True)

        if candidates.size(1) == 0:

            return [], {'text': f"No candidate links found for Node {node_idx}.", "type": "info"}, "RUN ANALYSIS", False

        top_predictions = get_top_k_predictions(embeddings, candidates.to(device), model, k=50)

        status = {'text': f"Found {len(top_predictions)} link predictions for Node {node_idx}.", "type": "success"}

        return top_predictions, status, "RUN ANALYSIS", False

    except Exception as e:

        print(f"Link prediction callback error: {e}")

        import traceback

        traceback.print_exc()

        return no_update, {'text': f"Link prediction failed: {str(e)}", "type": "error"}, "RUN ANALYSIS", False


@callback(

    Output(ids.LINK_PREDICTION_OUTPUT_ACTIVE, 'children'),

    Output(ids.LINK_CONFIDENCE_PLOT_ACTIVE, 'figure'),

    Output(ids.LINK_CONFIDENCE_PLOT_ACTIVE, 'style'),

    Input(ids.LINK_PREDICTION_STORE, 'data')

)

def update_link_list(link_predictions):
    """The 'top predicted links for the source node' list is not shown: links
    are opened from the confusion-matrix cell list instead (click a cell, then
    a row). Without a source node the list defaulted to node 0, which was noise."""
    return html.Div([], style={'minHeight': '0'}), go.Figure(), {'display': 'none', 'height': '0px'}


@callback(

    Output(ids.SUBGRAPH_EVAL_STATUS, 'children'),

    Output(ids.CONFUSION_MATRIX_PLOT, 'figure', allow_duplicate=True),

    Output(ids.EDGE_PREDICTIONS_DISPLAY, 'children', allow_duplicate=True),

    Output(ids.PERFORMANCE_METRICS, 'children'),
    Output(ids.PERFORMANCE_METRICS, 'style'),
    Output(ids.SUBGRAPH_EVAL_STORE, 'data'),
    Output(ids.GLOBAL_LINK_ANALYSIS_TITLE, 'children', allow_duplicate=True),

    Input(ids.LINK_PREDICTION_STORE, 'data'),
    Input(ids.ANALYSIS_MODE_STORE, 'data'),
    Input(ids.PRECOMPUTED_PACKAGE_STORE, 'data'),
    State(ids.FULL_DATASET_STORE, 'data'),
    State(ids.MODEL_DROPDOWN, 'value'),
    State(ids.DATASET_DROPDOWN, 'value'),
    State(ids.GRAPH_TYPE_STORE, 'data'),
    prevent_initial_call=True
)
def evaluate_subgraph_predictions(link_pred_data, analysis_mode, package, dataset, model_type, dataset_name, graph_type):
    """Link-view confusion matrix. It is a global evaluation of the model on the
    held-out edges (independent of the typed or selected link), so it is built
    when Link view is entered or the model changes, and cached per
    (model, dataset): RUN and link selections reuse it instantly instead of
    waiting behind other model work."""
    if not package or not dataset or not get_task(analysis_mode).is_link:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update
    key = (model_type, dataset_name)
    result = _LINK_EVAL_CACHE.get(key)
    if result is None:
        result = _evaluate_link_predictions(link_pred_data, analysis_mode, package, dataset,
                                            model_type, dataset_name, graph_type)
        if isinstance(result[0], str) and result[0].startswith('✅'):
            _LINK_EVAL_CACHE[key] = result
    return result


_LINK_EVAL_CACHE = {}   # (model_type, dataset_name) -> evaluate_subgraph_predictions outputs


@serialize_model_use
def _evaluate_link_predictions(link_pred_data, analysis_mode, package, dataset, model_type, dataset_name, graph_type):

    """Evaluate edge predictions on subgraph and show confusion matrix."""

    try:
        # DEBUG: log evaluation entry and presence of inputs
        try:
            with open('link_debug.log', 'a') as _log:
                _log.write(f"EVAL_SUBGRAPH link_pred_data_present={bool(link_pred_data)} analysis_mode={analysis_mode} package_present={bool(package)} dataset_present={bool(dataset)}\n")
        except Exception:
            pass

        if not package or not dataset or not get_task(analysis_mode).is_link:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update

        # Evaluate on the models' OWN held-out test split. train_models.py builds
        # it with create_edge_splits(..., random_state=42), so it is reproducible
        # here; scoring uses the training graph (test edges removed), exactly as
        # in training. (The old subgraph_models/ edge list came from a different
        # split: ~80% of its "held-out" edges were training edges of the current
        # models, and they were also left in the graph while being scored.)
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            model = get_cached_model(package, dataset.get('num_features'), dataset.get('num_classes'))
            if model is None or not getattr(model, 'enable_link_prediction', False):
                # never show simulated numbers as if they were an evaluation
                return ("❌ This model has no link-prediction head to evaluate", go.Figure(),
                        [cell_details_placeholder(LINK_PREDICTION)], no_update, {'display': 'none'}, no_update, no_update)

            split = link_test_split(dataset_name)
            positives = [tuple(e) for e in split['test_pos']]
            negatives = [tuple(e) for e in split['test_neg']]
            removed_edges = positives   # held-out positives (name used below)
            full = get_cached_dataset({'name': dataset_name})
            with torch.no_grad():
                _, emb = model(full['x'].to(device), split['train_edge_index'].to(device), return_embeddings=True)
                pairs = torch.tensor(positives + negatives, dtype=torch.long).t().to(emb.device)
                scores = torch.sigmoid(model.predict_links(emb, pairs).reshape(-1)).cpu().tolist()

            pos_data, neg_data = [], []
            for i, ((u, v), score) in enumerate(zip(positives + negatives, scores)):
                rec = {'u': u, 'v': v, 'score': float(score), 'pred': 1 if score > 0.5 else 0,
                       'true': 1 if i < len(positives) else 0}
                (pos_data if i < len(positives) else neg_data).append(rec)

            all_data = pos_data + neg_data

            true_labels = [d['true'] for d in all_data]

            predicted_labels = [d['pred'] for d in all_data]

            # --- CALCULATE METRICS ---

            accuracy = accuracy_score(true_labels, predicted_labels)

            precision = precision_score(true_labels, predicted_labels, zero_division=0)

            recall = recall_score(true_labels, predicted_labels, zero_division=0)

            f1 = f1_score(true_labels, predicted_labels, zero_division=0)

            # --- PREPARE HOVER DATA (Buckets) ---

            tp_list = []

            fp_list = []

            fn_list = []

            tn_list = []

            for d in all_data:

                item = (d['score'], d['u'], d['v'])

                if d['true'] == 1 and d['pred'] == 1:

                    tp_list.append(item)

                elif d['true'] == 0 and d['pred'] == 1:

                    fp_list.append(item)

                elif d['true'] == 1 and d['pred'] == 0:

                    fn_list.append(item)

                elif d['true'] == 0 and d['pred'] == 0:

                    tn_list.append(item)

            # Sort Top K

            # TP: Highest score first

            tp_list.sort(key=lambda x: x[0], reverse=True)

            # FP: Highest score first

            fp_list.sort(key=lambda x: x[0], reverse=True)

            # FN: Lowest score first (closest to 0)

            fn_list.sort(key=lambda x: x[0])

            # TN: Lowest score first

            tn_list.sort(key=lambda x: x[0])

            fig = link_confusion_figure(true_labels, predicted_labels, tp_list, fp_list, fn_list, tn_list)

            # Leave the cell-details panel alone: it is filled only when a matrix
            # cell is clicked (update_edge_list_from_cm) and cleared by
            # reset_cell_details. This callback also re-runs when a link is picked,
            # which would otherwise wipe the list the user just opened.
            edge_predictions = no_update

            # Performance metrics on the held-out (removed) edges
            metrics = metric_tiles([
                ("Accuracy", float(accuracy), "Link accuracy on the removed (held-out) edges and sampled non-edges"),
                ("Precision", float(precision), "Of predicted links, the share that are real"),
                ("Recall", float(recall), "Of real removed edges, the share recovered"),
                ("F1", float(f1), "Harmonic mean of precision and recall"),
            ])

            status_msg = f"✅ Evaluated {len(removed_edges)} held-out test edges"

            perf_style = {'display': 'flex'}

            return status_msg, fig, edge_predictions, metrics, perf_style, {'TP': tp_list, 'FP': fp_list, 'FN': fn_list, 'TN': tn_list}, "Confusion Matrix Details"

        except Exception as e:

            return f"❌ Error loading subgraph model: {str(e)}", go.Figure(), no_update, no_update, no_update, no_update, no_update

    except Exception as e:

        print(f"Subgraph evaluation error: {e}")

        return f"❌ Evaluation failed: {str(e)}", go.Figure(), no_update, no_update, no_update, no_update, no_update

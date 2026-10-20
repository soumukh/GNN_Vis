"""Overview callbacks: embeddings, confusion matrix, metric tiles and the confusion-matrix cell details."""

import numpy as np
import traceback
from dash import Input, Output, State, callback, ctx, html, no_update

from gnnvis.runtime import get_cached_dataset_if_missing, get_cached_package_if_missing
from gnnvis.ui.components import cell_details_placeholder, confusion_matrix_style, metric_tiles
from gnnvis.ui.figures.confusion import _render_cm_detail_page, node_confusion_figure
from gnnvis.ui.figures.embeddings import get_2d_embeddings, plot_embeddings
from gnnvis.ui.theme import CORRECT_COLOR, INCORRECT_COLOR, class_color_map, class_label, empty_figure
from gnnvis.ui import ids
from gnnvis.tasks import get_task


@callback(
    Output(ids.EDGE_PREDICTIONS_DISPLAY, 'children', allow_duplicate=True),
    Output(ids.CM_DETAIL_PAGING_STORE, 'data', allow_duplicate=True),
    Input(ids.ANALYSIS_MODE_STORE, 'data'),
    Input(ids.DATASET_DROPDOWN, 'value'),
    Input(ids.MODEL_DROPDOWN, 'value'),
    prevent_initial_call=True,
)
def reset_cell_details(analysis_mode, _dataset, _model):
    """A clicked cell belongs to one matrix; clear it when the matrix changes."""
    return cell_details_placeholder(analysis_mode), None


@callback(
    Output(ids.CONFUSION_MATRIX_PLOT, 'style'),
    Input(ids.ANALYSIS_MODE_STORE, 'data'),
)
def update_confusion_matrix_style(analysis_mode):
    return confusion_matrix_style(analysis_mode)


@callback(
    Output(ids.EMBEDDINGS_VIEW, 'figure'),
    Output(ids.CONFUSION_MATRIX_PLOT, 'figure', allow_duplicate=True),
    Output(ids.GNN_MID_LAYER_CONTENT, 'style'),
    Input(ids.GLOBAL_RUN_BUTTON, 'n_clicks'),
    Input(ids.CURRENT_PREDICTIONS_STORE, 'data'),
    Input(ids.CPA_IV_STORE, 'data'),
    Input(ids.LINK_CPA_IV_STORE, 'data'),
    Input(ids.EMBEDDING_COLOR_MODE, 'value'),
    Input(ids.SELECTED_NODE_STORE, 'data'),
    Input(ids.SELECTED_LINK_STORE, 'data'),
    Input(ids.ANALYSIS_MODE_STORE, 'data'),
    State(ids.PRECOMPUTED_PACKAGE_STORE, 'data'),
    State(ids.FULL_DATASET_STORE, 'data'),
    State(ids.LAYER3_TABS, 'value'),
    State(ids.LINK_PREDICTION_STORE, 'data'),
    prevent_initial_call=True
)
def update_mid_layer(n_clicks, predictions_data, cpa_data, link_cpa_data, embedding_color_mode, selected_node, selected_link, analysis_mode, package, dataset, active_layer3_tab, link_prediction_store):
    try:
        empty_fig = empty_figure("Loading model…")
        gnn_style = {'flex': '0.8', 'minWidth': '260px', 'minHeight': '0', 'height': '100%', 'maxHeight': '100%',
                     'margin': '0', 'overflow': 'hidden', 'boxSizing': 'border-box',
                     'display': 'flex', 'flexDirection': 'column'}

        # Ensure we have the full cached dataset (callbacks may pass metadata only)
        dataset_full = get_cached_dataset_if_missing(dataset)
        package_meta = package
        package = get_cached_package_if_missing(package_meta)
        if not predictions_data or not package or not dataset_full:
            return empty_fig, empty_fig, gnn_style

        preds = np.array(predictions_data['preds'])
        true_labels = np.array(dataset_full['y'])
        num_classes = dataset_full['num_classes']
        dataset_name = dataset_full.get('name')
        class_to_color_map = class_color_map(num_classes)

        selected_node_id_str = selected_node['id'] if selected_node else None
        target_node_id_str = None
        if get_task(analysis_mode).is_link and selected_link and isinstance(selected_link, dict):
            target = selected_link.get('target')
            if target is not None:
                target_node_id_str = str(target)
            # mark the link's source with the star (not a node selected in node view)
            if selected_link.get('source') is not None:
                selected_node_id_str = str(selected_link['source'])

        embeddings_2d = get_2d_embeddings(package)
        if embeddings_2d is None:
            embeddings_fig = empty_figure("Embeddings not available")
        else:
            try:
                embeddings_fig = plot_embeddings(np.array(embeddings_2d), preds, true_labels, selected_node_id_str, class_to_color_map, num_classes,
                                                 target_node_id=target_node_id_str, dataset_name=dataset_name,
                                                 color_mode=embedding_color_mode or 'actual',
                                                 confidences=predictions_data.get('confidences'))
            except Exception:
                # Keep going: a failed embedding plot must not blank the confusion matrix
                traceback.print_exc()
                embeddings_fig = empty_figure("Could not draw embeddings - change colour mode to retry")


        # In Link Prediction mode the confusion matrix is produced by the
        # subgraph/link-evaluation callback. Avoid overwriting that figure
        # by returning `no_update` for the CM output so other callbacks can
        # set it (prevents brief flash + overwrite behavior).
        if get_task(analysis_mode).is_link:
            return embeddings_fig, no_update, gnn_style

        cm_fig = node_confusion_figure(true_labels, preds, num_classes, dataset_name)

        return embeddings_fig, cm_fig, gnn_style

    except Exception as e:
        print(f"Error in update_mid_layer: {e}")
        import traceback
        traceback.print_exc()
        return (empty_figure("Could not draw embeddings"), empty_figure("Could not draw confusion matrix"),
                {'flex': '0.8', 'minWidth': '260px', 'minHeight': '0', 'height': '100%', 'maxHeight': '100%',
                 'margin': '0', 'overflow': 'hidden', 'boxSizing': 'border-box',
                 'display': 'flex', 'flexDirection': 'column'})


# Node classification performance metrics — populate the same `performance-metrics` area
@callback(
    Output(ids.PERFORMANCE_METRICS, 'children', allow_duplicate=True),
    Output(ids.PERFORMANCE_METRICS, 'style', allow_duplicate=True),
    Input(ids.GLOBAL_RUN_BUTTON, 'n_clicks'),
    Input(ids.CURRENT_PREDICTIONS_STORE, 'data'),
    Input(ids.ANALYSIS_MODE_STORE, 'data'),
    State(ids.FULL_DATASET_STORE, 'data'),
    prevent_initial_call=True,
)
def update_node_performance_metrics(run_clicks, predictions_data, analysis_mode, dataset_meta):
    try:
        # Node-classification metrics, shown as soon as predictions exist
        if get_task(analysis_mode).is_link:
            return no_update, no_update

        if not predictions_data or not dataset_meta:
            return no_update, no_update

        dataset_full = get_cached_dataset_if_missing({'name': dataset_meta.get('name')}) if isinstance(dataset_meta, dict) else None
        if not dataset_full or 'y' not in dataset_full:
            return no_update, no_update

        preds = np.array(predictions_data.get('preds', []))
        true = np.asarray(dataset_full['y'])
        if preds.size == 0 or true.size == 0:
            return no_update, no_update

        from sklearn.metrics import accuracy_score, f1_score

        items = []
        test_mask = dataset_full.get('test_mask')
        if test_mask is not None:
            tm = np.asarray(test_mask, dtype=bool)
            items.append(("Test acc", float(accuracy_score(true[tm], preds[tm])),
                          f"Accuracy on the {int(tm.sum())} standard Planetoid test nodes"))
        items.append(("All-node acc", float(accuracy_score(true, preds)),
                      "Accuracy over every node, including the training nodes"))
        items.append(("Macro F1", float(f1_score(true, preds, average='macro', zero_division=0)),
                      "Macro-averaged F1 over every node"))
        metrics = metric_tiles(items)

        perf_style = {'display': 'flex'}

        return metrics, perf_style

    except Exception as e:
        print(f"Error in update_node_performance_metrics: {e}")
        return no_update, no_update


@callback(
    Output(ids.EDGE_PREDICTIONS_DISPLAY, 'children', allow_duplicate=True),
    Output(ids.GLOBAL_LINK_ANALYSIS_TITLE, 'children', allow_duplicate=True),
    Output(ids.CM_DETAIL_PAGING_STORE, 'data', allow_duplicate=True),
    Input(ids.CONFUSION_MATRIX_PLOT, 'clickData'),
    State(ids.SUBGRAPH_EVAL_STORE, 'data'),
    State(ids.ANALYSIS_MODE_STORE, 'data'),
    State(ids.CURRENT_PREDICTIONS_STORE, 'data'),
    State(ids.FULL_DATASET_STORE, 'data'),
    prevent_initial_call=True
)
def update_edge_list_from_cm(clickData, eval_data, analysis_mode, predictions_data, dataset):
    """
    Handles clicks on either confusion matrix variant:

    NODE CLASSIFICATION  (multi-class CM, x/y = class indices as strings):
        Shows the list of nodes that fall in the clicked (actual_class, pred_class) cell.
        Diagonal cells = correctly classified; off-diagonal = misclassified.

    LINK PREDICTION  (binary CM, x/y = "Predicted: Edge" / "Actual: No Edge" etc.):
        Maps the cell to TP / FP / FN / TN and shows the corresponding edge list
        from subgraph-eval-store, sorted by score.

        Layout of the link-prediction confusion matrix (y = Actual, x = Predicted):

        The plot uses sklearn's confusion_matrix layout and the heatmap axes are:
            x = ['Predicted: No Edge', 'Predicted: Edge']
            y = ['Actual: No Edge',    'Actual: Edge']

        Corresponding cell mapping (sklearn 2x2 matrix):
            [0,0] = TN (Actual: No Edge, Predicted: No Edge)
            [0,1] = FP (Actual: No Edge, Predicted: Edge)
            [1,0] = FN (Actual: Edge,    Predicted: No Edge)
            [1,1] = TP (Actual: Edge,    Predicted: Edge)
    """
    if not clickData:
        return no_update, no_update, no_update

    print(f"[CM Click] clickData received: {clickData}")
    print(f"[CM Click] analysis_mode={analysis_mode}")

    empty_paging = {'rows': [], 'columns': [], 'badge_color': '', 'footer_text': '', 'page': 0, 'page_count': 1}
    try:
        point = clickData['points'][0]

        # Axis values are numeric class indices: 0 is valid, so test for None, not truthiness
        def _first(*keys):
            for k in keys:
                v = point.get(k)
                if v is not None and v != '':
                    return str(v)
            return ''
        x_val = _first('x', 'xaxis', 'xlabel')
        y_val = _first('y', 'yaxis', 'ylabel')
        if not x_val or not y_val:
            print(f"[CM Click] Point keys: {list(point.keys())}")
            return (
                html.Div(
                    'Clicked cell information could not be interpreted. Try clicking a different cell.',
                    style={'color': '#999', 'fontStyle': 'italic', 'fontSize': '12px', 'padding': '10px', 'textAlign': 'center'}
                ),
                'Confusion Matrix Details',
                empty_paging,
            )

        print(f"[CM Click] x_val (predicted)='{x_val}'  y_val (actual)='{y_val}'")

        PANEL_TITLE = 'Confusion Matrix Details'

        # Ensure node-classification uses the full cached dataset so y labels are available.
        if not get_task(analysis_mode).is_link and dataset and isinstance(dataset, dict) and 'name' in dataset:
            dataset = get_cached_dataset_if_missing(dataset)

        def _badge_div(label, count, unit, badge_color, description):
            return html.Div(style={'marginBottom': '8px'}, children=[
                html.Span(
                    f'{label}  ({count} {unit})',
                    style={
                        'backgroundColor': badge_color, 'color': 'white',
                        'padding': '3px 10px', 'borderRadius': '12px',
                        'fontSize': '12px', 'fontWeight': 'bold', 'marginRight': '8px',
                    }
                ),
                html.Span(description, style={'fontSize': '11px', 'color': '#666'}),
            ])

        empty_paging = {
            'rows': [],
            'columns': [],
            'badge_color': '',
            'footer_text': '',
            'page': 0,
            'page_count': 1,
        }

        # ─── NODE CLASSIFICATION ─────────────────────────────────────────
        if not get_task(analysis_mode).is_link:
            print(f"[CM Click] Entering node_classification branch")
            if not predictions_data or not dataset:
                return (
                    html.Div(
                        'No prediction data available. Run analysis first.',
                        style={'color': '#999', 'fontStyle': 'italic',
                               'fontSize': '12px', 'padding': '10px', 'textAlign': 'center'}
                    ),
                    PANEL_TITLE,
                    empty_paging,
                )

            try:
                pred_class   = int(x_val)
                actual_class = int(y_val)
            except (ValueError, TypeError):
                return no_update, no_update, no_update

            preds       = predictions_data.get('preds', [])
            true_labels = dataset.get('y', [])
            if len(preds) == 0 or len(true_labels) == 0:
                print(f"[CM Click] Missing preds/true_labels for node classification: len(preds)={len(preds)}, len(true_labels)={len(true_labels)}")
                return (
                    html.Div(
                        'No prediction or label data available for node classification. Reload the dataset and try again.',
                        style={'color': '#999', 'fontStyle': 'italic', 'fontSize': '12px', 'padding': '10px', 'textAlign': 'center'}
                    ),
                    PANEL_TITLE,
                    empty_paging,
                )

            matching = [
                node_id
                for node_id, (p, t) in enumerate(zip(preds, true_labels))
                if int(p) == pred_class and int(t) == actual_class
            ]
            count = len(matching)
            print(f"[CM Click] Node class cell ({actual_class}→{pred_class}): {count} nodes found")

            is_diagonal = (pred_class == actual_class)
            ds_name = dataset.get('name') if isinstance(dataset, dict) else None
            pred_name, actual_name = class_label(ds_name, pred_class), class_label(ds_name, actual_class)
            if is_diagonal:
                badge_color = CORRECT_COLOR
                label       = f'Correctly classified as {pred_name}'
                description = (f'{count} node{"s" if count != 1 else ""} of class '
                               f'{actual_name} predicted correctly')
            else:
                badge_color = INCORRECT_COLOR
                label       = f'{actual_name} → predicted {pred_name}'
                description = (f'{count} node{"s" if count != 1 else ""} of class '
                               f'{actual_name} wrongly predicted as {pred_name}')

            header = _badge_div(label, count, 'nodes', badge_color, description)

            if count == 0:
                return (
                    html.Div([
                        header,
                        html.Div('No nodes in this cell.',
                                 style={'color': '#999', 'fontStyle': 'italic',
                                        'fontSize': '13px', 'padding': '10px', 'textAlign': 'center'}),
                    ]),
                    PANEL_TITLE,
                    empty_paging,
                )

            # Include confidence if available in predictions_data
            confidences = predictions_data.get('confidences') if predictions_data else None
            rows = []
            for i, nid in enumerate(matching):
                row = {'#': i + 1, 'Node ID': nid, 'True Class': actual_class, 'Predicted Class': pred_class}
                if confidences and len(confidences) > nid:
                    # store as percentage (0-100) with one decimal
                    row['Confidence'] = round(float(confidences[nid]) * 100.0, 1)
                else:
                    row['Confidence'] = None
                rows.append(row)

            columns = [
                {'name': '#',               'id': '#',               'type': 'numeric'},
                {'name': 'Node ID',          'id': 'Node ID',         'type': 'numeric'},
                # Short headers: this panel is narrow
                {'name': 'True',             'id': 'True Class',      'type': 'numeric'},
                {'name': 'Pred.',            'id': 'Predicted Class', 'type': 'numeric'},
                {'name': 'Conf. %',          'id': 'Confidence',      'type': 'numeric'},
            ]
            footer_text = f'Showing {count} node{"s" if count != 1 else ""}'
            page_children, page_data = _render_cm_detail_page(rows, columns, badge_color, footer_text, 0)
            return html.Div([header, page_children], style={'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'minHeight': 0}), PANEL_TITLE, page_data

        # ─── LINK PREDICTION ─────────────────────────────────────────────
        print(f"[CM Click] Entering link_prediction branch. eval_data present: {bool(eval_data)}")
        if not eval_data or not isinstance(eval_data, dict):
            return (
                html.Div(
                    'No edge evaluation data available. Ensure a subgraph model is loaded.',
                    style={'color': '#999', 'fontStyle': 'italic',
                           'fontSize': '12px', 'padding': '10px', 'textAlign': 'center'}
                ),
                PANEL_TITLE,
                empty_paging,
            )

        # True if "Edge" appears but NOT "No Edge" (avoids substring collision)
        pred_is_edge   = 'Edge' in x_val and 'No Edge' not in x_val
        actual_is_edge = 'Edge' in y_val and 'No Edge' not in y_val

        if actual_is_edge and pred_is_edge:
            category    = 'TP';  label = 'True Positives'
            badge_color = '#27ae60'
            description = 'Edges that exist AND were correctly predicted'
        elif not actual_is_edge and pred_is_edge:
            category    = 'FP';  label = 'False Positives'
            badge_color = '#e74c3c'
            description = 'Non-edges that were wrongly predicted as edges'
        elif actual_is_edge and not pred_is_edge:
            category    = 'FN';  label = 'False Negatives'
            badge_color = '#e67e22'
            description = 'Real edges that the model missed'
        else:
            category    = 'TN';  label = 'True Negatives'
            badge_color = '#2980b9'
            description = 'Non-edges correctly predicted as absent'

        # eval_data stores lists of [score, u, v] (JSON-serialised tuples)
        edge_list = eval_data.get(category, [])
        count     = len(edge_list)
        print(f"[CM Click] Link prediction category='{category}' ({label}): {count} edges")

        header = _badge_div(label, count, 'edges', badge_color, description)

        if count == 0:
            return (
                html.Div([
                    header,
                    html.Div(f'No edges found for {label}.',
                             style={'color': '#999', 'fontStyle': 'italic',
                                    'fontSize': '13px', 'padding': '10px', 'textAlign': 'center'}),
                ]),
                label,
                empty_paging,
            )

        MAX_ROWS = 200
        rows = [
            {
                '#':           rank,
                'Source Node': int(item[1]),
                'Target Node': int(item[2]),
                'Confidence':  round(float(item[0]), 4),
            }
            for rank, item in enumerate(edge_list[:MAX_ROWS], start=1)
        ]
        columns = [
            {'name': '#',           'id': '#',           'type': 'numeric'},
            {'name': 'Source Node', 'id': 'Source Node', 'type': 'numeric'},
            {'name': 'Target Node', 'id': 'Target Node', 'type': 'numeric'},
            {'name': 'Confidence',  'id': 'Confidence',  'type': 'numeric'},
        ]
        footer_text = f'Showing top {min(count, MAX_ROWS)} of {count} edges  ·  sorted by confidence descending'
        page_children, page_data = _render_cm_detail_page(rows, columns, badge_color, footer_text, 0)
        return html.Div([header, page_children], style={'display': 'flex', 'flexDirection': 'column', 'height': '100%', 'minHeight': 0}), label, page_data

    except Exception as e:
        print(f"Error in update_edge_list_from_cm: {e}")
        import traceback
        traceback.print_exc()
        return no_update, no_update, no_update


@callback(
    Output(ids.EDGE_PREDICTIONS_DISPLAY, 'children', allow_duplicate=True),
    Output(ids.CM_DETAIL_PAGING_STORE, 'data', allow_duplicate=True),
    Input(ids.CM_PAGE_PREV, 'n_clicks'),
    Input(ids.CM_PAGE_NEXT, 'n_clicks'),
    State(ids.CM_DETAIL_PAGING_STORE, 'data'),
    prevent_initial_call=True
)
def update_cm_detail_page(prev_clicks, next_clicks, paging_data):
    if not paging_data or 'rows' not in paging_data:
        return no_update, no_update

    triggered_id = ctx.triggered_id
    if triggered_id not in {ids.CM_PAGE_PREV, ids.CM_PAGE_NEXT}:
        return no_update, no_update

    current_page = paging_data.get('page', 0)
    page_count = paging_data.get('page_count', 1)

    if triggered_id == ids.CM_PAGE_PREV:
        new_page = max(0, current_page - 1)
    else:
        new_page = min(page_count - 1, current_page + 1)

    if new_page == current_page:
        return no_update, no_update

    page_children, updated_paging = _render_cm_detail_page(
        paging_data['rows'],
        paging_data['columns'],
        paging_data['badge_color'],
        paging_data['footer_text'],
        new_page,
    )

    return page_children, updated_paging

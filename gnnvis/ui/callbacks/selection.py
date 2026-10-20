"""Selection callbacks: selected node, link and causal path."""

import dash.dependencies
from dash import ALL, Input, Output, State, callback, ctx, no_update


@callback(
    Output('selected-node-store', 'data'),
    Input('graph-view', 'tapNodeData'),
    Input('embeddings-view', 'clickData'),
    Input('full-graph-view', 'clickData'),
    prevent_initial_call=True,
)
def update_selected_node(tapNode, emb_click, full_click):
    trigger_id, node_id = ctx.triggered_id, None

    if trigger_id == 'graph-view' and tapNode:
        node_id = tapNode['id']
    elif trigger_id in ('embeddings-view', 'full-graph-view'):
        click = emb_click if trigger_id == 'embeddings-view' else full_click
        cd = ((click or {}).get('points') or [{}])[0].get('customdata')
        if isinstance(cd, (list, tuple)):
            cd = cd[0] if cd else None
        if cd is not None:
            node_id = cd

    if node_id is not None:
        print(f"[SELECTED_NODE] update_selected_node triggered by {trigger_id}, node_id={node_id}")
        # Store ids as strings everywhere (Cytoscape ids are strings)
        return {'id': str(node_id)}
    else:
        print(f"[SELECTED_NODE] update_selected_node triggered by {trigger_id}, no node selected")
        return no_update


@callback(

    Output('selected-node-store', 'data', allow_duplicate=True),

    Input('select-node-input-nc', 'value'),

    prevent_initial_call=True

)

def update_selected_node_from_input(input_value):

    if input_value is not None:
        print(f"[SELECTED_NODE] update_selected_node_from_input value={input_value}")
        return {'id': str(input_value)}

    return no_update


@callback(

    Output('selected-node-store', 'data', allow_duplicate=True),

    Input('select-node-input-nc-active', 'value'),

    prevent_initial_call=True

)

def update_selected_node_from_input_active(input_value):

    if input_value is not None:
        print(f"[SELECTED_NODE] update_selected_node_from_input_active value={input_value}")
        return {'id': str(input_value)}

    return no_update


@callback(
    Output('selected-node-store', 'data', allow_duplicate=True),
    Input('global-run-button', 'n_clicks'),
    prevent_initial_call=True
)
def clear_node_selection_on_run(run_clicks):
    """Clear node selection when RUN is clicked so Layer 3 stays hidden until a node is picked."""
    # Preserve the selected node on Run to avoid hiding CPA/Layer3 unexpectedly.
    # Previous behavior cleared selection unconditionally; this caused CPA views
    # to disappear when the user clicked Run. Keep selection intact.
    return no_update


@callback(

    Output('selected-link-store', 'data'),

    # Triggers

    Input('global-run-button', 'n_clicks'),

    Input({'type': 'link-prediction-item', 'index': ALL, 'source': ALL, 'target': ALL}, 'n_clicks'),

    # States for manual inputs (kept for signature compatibility but ignored for selection)

    State({'type': 'lp-input', 'index': ALL}, 'value'),

    State({'type': 'lp-input', 'index': ALL}, 'id'),

    State('analysis-mode-store', 'data'),

    State('layer3-tabs', 'value'),

    State('full-dataset-store', 'data'),

    prevent_initial_call=True

)

def update_selected_link_master(run_clicks, item_clicks, lp_input_values, lp_input_ids, analysis_mode, active_layer3_tab, dataset_meta):

    """

    STRICT INTERACTION FIX:

    - Updates selected-link-store ONLY when a link prediction item is clicked.

    - CLEARS selection (returns None) when RUN ANALYSIS is clicked.

    - IGNORES manual inputs for opening Layer 3.

    """

    if not ctx.triggered:

        raise dash.exceptions.PreventUpdate

    triggered_id = ctx.triggered_id

    # Priority 1: A specific link was clicked from the prediction list

    # This is the ONLY allowed trigger for opening Layer 3

    if isinstance(triggered_id, dict) and triggered_id.get('type') == 'link-prediction-item':

        if not any(item_clicks): raise dash.exceptions.PreventUpdate

        source = triggered_id.get('source')

        target = triggered_id.get('target')

        if source is not None and target is not None:

            print(f"Selected link from prediction list: ({source}, {target})")

            return {'source': int(source), 'target': int(target)}

    # Priority 2: "RUN ANALYSIS" clicked
    # Preserve current selection on Run to avoid race conditions where
    # the selection is cleared before downstream callbacks read it.
    elif triggered_id == 'global-run-button':
        # Link mode: a typed source + target pair becomes the selected link, which
        # opens its subgraph, explanation and CPA-IV (invalid pairs are reported
        # by run_link_prediction_unified via link-status-store).
        if analysis_mode == 'link_prediction' and lp_input_values and lp_input_ids:
            vals = {i.get('index'): v for v, i in zip(lp_input_values, lp_input_ids)}
            try:
                u, v = int(vals.get('source-active')), int(vals.get('target-active'))
            except (TypeError, ValueError):
                return no_update
            n = (dataset_meta or {}).get('num_nodes') or 0
            if u != v and 0 <= u < n and 0 <= v < n:
                print(f"RUN ANALYSIS with typed link ({u}, {v}) - selecting it.")
                return {'source': u, 'target': v}
        print("RUN ANALYSIS clicked - preserving selected link (no clear).")
        return no_update

    raise dash.exceptions.PreventUpdate


@callback(
    Output('selected-path-store', 'data'),
    Input({'type': 'cpa-path-item', 'index': ALL}, 'n_clicks'),
    Input({'type': 'cpa-path-item-side', 'index': ALL}, 'n_clicks'),
    Input('global-run-button', 'n_clicks'),
    prevent_initial_call=True
)
def update_selected_path(main_clicks, side_clicks, run_clicks):
    trigger = ctx.triggered_id
    
    # Reset on Run
    if trigger == 'global-run-button':
        return None
        
    # Handle clicks
    if isinstance(trigger, dict):
        if 'index' in trigger:
            print(f"Path selected: {trigger['index']}")
            return {'index': trigger['index']}
            
    return no_update


# Allow selecting a node by clicking a cell in the CM detail DataTable
@callback(
    Output('selected-node-store', 'data', allow_duplicate=True),
    Input('cm-detail-table', 'active_cell'),
    State('cm-detail-table', 'data'),
    prevent_initial_call=True,
)
def select_node_from_detail_table(active_cell, table_data):
    if not active_cell or not table_data:
        return no_update

    row_idx = active_cell.get('row')
    col_id = active_cell.get('column_id')
    try:
        row = table_data[row_idx]
    except Exception:
        return no_update

    # Prefer explicit node id columns
    for candidate in ('Node ID', 'Source Node', 'Target Node', 'Source', 'Target'):
        if candidate in row and row.get(candidate) is not None:
            val = str(row.get(candidate))
            print(f"[SELECTED_NODE] select_node_from_detail_table choosing {candidate}={val}")
            return {'id': val}

    # Fallback: if clicked column contains a numeric-looking value
    val = row.get(col_id)
    if val is not None:
        print(f"[SELECTED_NODE] select_node_from_detail_table fallback value={val}")
        return {'id': str(val)}

    return no_update


# If the details table row represents a link (has both source and target), select that link
@callback(
    Output('selected-link-store', 'data', allow_duplicate=True),
    Input('cm-detail-table', 'active_cell'),
    State('cm-detail-table', 'data'),
    prevent_initial_call=True,
)
def select_link_from_detail_table(active_cell, table_data):
    if not active_cell or not table_data:
        return no_update

    row_idx = active_cell.get('row')
    try:
        row = table_data[row_idx]
    except Exception:
        return no_update

    # Recognize common link column names
    src_keys = ['Source Node', 'Source', 'Source_ID']
    tgt_keys = ['Target Node', 'Target', 'Target_ID']
    src = None
    tgt = None
    for k in src_keys:
        if k in row and row.get(k) is not None:
            src = row.get(k)
            break
    for k in tgt_keys:
        if k in row and row.get(k) is not None:
            tgt = row.get(k)
            break

    if src is not None and tgt is not None:
        try:
            return {'source': int(src), 'target': int(tgt)}
        except Exception:
            return {'source': str(src), 'target': str(tgt)}

    return no_update

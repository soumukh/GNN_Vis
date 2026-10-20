"""Session-level callbacks: model / dataset loading, task switch, graph edits, status line, layer visibility, reset, RUN-button feedback."""

import dash.dependencies
from dash import clientside_callback, Input, Output, State, callback, ctx, dcc, html, no_update

from app_expl import run_inference

from gnnvis.runtime import get_cached_dataset_if_missing, get_cached_model, get_cached_package_if_missing, package_meta_from_selection, prepare_torch_data, serialize_model_use
from gnnvis.ui import ids


@callback(
    Output(ids.STATUS_BAR, 'children'),
    Input(ids.DATASET_DROPDOWN, 'value'),
    Input(ids.MODEL_DROPDOWN, 'value'),
    Input(ids.ANALYSIS_MODE_STORE, 'data'),
    Input(ids.CURRENT_PREDICTIONS_STORE, 'data'),
    Input(ids.SELECTED_NODE_STORE, 'data'),
    Input(ids.SELECTED_LINK_STORE, 'data'),
    Input(ids.EXPLANATION_STORE, 'data'),
    Input(ids.LINK_STATUS_STORE, 'data'),
)
def update_status_bar(dataset_name, model_type, analysis_mode, predictions_data, selected_node, selected_link, explanation, link_status):
    """Breadcrumb of the current context plus the next step to take."""
    task = 'Link prediction' if analysis_mode == 'link_prediction' else 'Node classification'
    chips = [html.Span(f"{dataset_name} · {model_type} · {task}", className='chip chip-context')]

    if not predictions_data:
        chips.append(html.Span("Loading model…", className='chip chip-busy'))
        return chips

    if analysis_mode == 'link_prediction':
        if isinstance(link_status, dict) and link_status.get('type') == 'error':
            chips.append(html.Span(link_status.get('text', 'Error'), className='chip chip-error'))
        if selected_link:
            chips.append(html.Span(f"Link {selected_link.get('source')} → {selected_link.get('target')}", className='chip'))
            step = "Inspect the causal paths and subgraph below."
        else:
            step = ("Enter source and target node IDs and press RUN ANALYSIS, or click a "
                    "confusion-matrix cell and then a link in its list.")
    else:
        node_id = (selected_node or {}).get('id')
        explained = (isinstance(explanation, dict) and explanation.get('explanation_type') == 'node_classification'
                     and not explanation.get('error')
                     and str(explanation.get('node_idx')) == str(node_id))
        if explanation and isinstance(explanation, dict) and explanation.get('error'):
            chips.append(html.Span(explanation['error'], className='chip chip-error'))
        if node_id is None:
            step = "Click a node in the embedding plot or full graph, or type an ID under Selection."
        else:
            chips.append(html.Span(f"Node {node_id}", className='chip'))
            step = ("Explanation ready: see the Explanation tab." if explained
                    else "Press RUN ANALYSIS to explain this node.")
    chips.append(html.Span(step, className='next-step'))
    return chips


@callback(
    Output(ids.ANALYSIS_MODE_STORE, 'data'),
    Output(ids.SELECTION_INTERFACE, 'children'),
    Output(ids.DYNAMIC_ANALYSIS_TABS, 'children'),
    Output(ids.GRAPH_TYPE_STORE, 'data'),
    Input(ids.ANALYSIS_MODE_SWITCH, 'value'),
)
def update_analysis_mode(mode):
    """Layer 1 only controls task mode. All explanation logic lives in Layer 3 tabs."""
    graph_type = 'full' if mode == 'node_classification' else 'sub'

    if mode == 'link_prediction':
        # Source + Target for link prediction (used by LP inference + Causal analysis)
        selection_interface = html.Div(style={'padding': '5px 0', 'width': '100%'}, children=[
            html.Div(style={'display': 'flex', 'gap': '10px', 'marginBottom': '5px'}, children=[
                html.Div(style={'flex': '1'}, children=[
                    dcc.Input(
                        id={'type': ids.LP_INPUT_TYPE, 'index': 'source-active'},
                        type='number', placeholder='Source', min=0, step=1,
                        style={'width': '100px'}
                    )
                ]),
                html.Div(style={'flex': '1'}, children=[
                    dcc.Input(
                        id={'type': ids.LP_INPUT_TYPE, 'index': 'target-active'},
                        type='number', placeholder='Target', min=0, step=1,
                        style={'width': '130px'}
                    )
                ])
            ])
        ])

    else:
        # Node classification — node ID input
        selection_interface = dcc.Input(
            id=ids.SELECT_NODE_INPUT_NC_ACTIVE,
            type='number', placeholder='Enter Node ID...',
            min=0, step=1, style={'width': '100%', 'marginTop': '5px'}
        )

    # Keep DOM components that callbacks depend on (hidden — they were inside the old tabs)
    kept_components = html.Div(style={'display': 'none'}, children=[
        dcc.Input(id=ids.EDGE_SOURCE_INPUT, type='number', value=None),
        dcc.Input(id=ids.EDGE_TARGET_INPUT, type='number', value=None),
        html.Button(id=ids.ADD_EDGE_BUTTON_NC, n_clicks=0),
        html.Button(id=ids.REMOVE_EDGE_BUTTON_NC, n_clicks=0),
        html.Div(id=ids.SUBGRAPH_EVAL_STATUS),
        dcc.Input(id={'type': ids.LP_INPUT_TYPE, 'index': 'pred-node'}, type='number', value=None),
    ])

    return mode, selection_interface, kept_components, graph_type


@callback(

    Output(ids.NODE_CLASSIFICATION_SECTION, 'style'),

    Input(ids.ANALYSIS_MODE_SWITCH, 'value')

)

def toggle_node_classification_section(mode):

    """Show/hide node classification section based on analysis mode."""

    if mode == 'link_prediction':

        # Hide the section in link prediction mode

        return {'display': 'none'}

    else:

        # Show the section in node classification mode

        return {'margin': 0, 'marginTop': '20px'}


@callback(

    Output(ids.PRECOMPUTED_PACKAGE_STORE, 'data'), Output(ids.FULL_DATASET_STORE, 'data'),

    Output(ids.EDITABLE_GRAPH_STORE, 'data'), Output(ids.CURRENT_PREDICTIONS_STORE, 'data'),

    Input(ids.DATASET_DROPDOWN, 'value'), Input(ids.MODEL_DROPDOWN, 'value'),

    prevent_initial_call=False

)
@serialize_model_use
def load_initial_data(dataset_name, model_type):

    try:

        print(f"\n{'='*60}")

        print(f"LOADING: {model_type} model for {dataset_name} dataset...")

        print(f"{ '='*60}")

        # Load package metadata into server-side cache only

        package_meta = package_meta_from_selection(model_type, dataset_name)
        package = get_cached_package_if_missing(package_meta)
        if package is None:
            error_msg = f"Failed to load {model_type} package for {dataset_name}"
            print(f"ERROR: {error_msg}")
            return None, None, None, None

        print(f"Package loaded into server cache. Keys: {list(package.keys())}")

        # Load dataset metadata into server-side cache only

        dataset_meta = {'name': dataset_name}
        dataset = get_cached_dataset_if_missing(dataset_meta)
        if dataset is None:
            error_msg = f"Failed to load dataset {dataset_name}"
            print(f"ERROR: {error_msg}")
            return None, None, None, None

        print(f"Dataset loaded into server cache: {dataset['num_nodes']} nodes, {dataset['num_features']} features, {dataset['num_classes']} classes")

        # Prepare lightweight metadata for browser-side stores

        true_labels = None
        if 'y' in dataset:
            try:
                true_labels = dataset['y'].cpu().tolist() if hasattr(dataset['y'], 'cpu') else list(dataset['y'])
            except Exception:
                try:
                    true_labels = list(dataset['y'])
                except Exception:
                    true_labels = None

        full_dataset_meta = {
            'name': dataset_name,
            'num_nodes': dataset['num_nodes'],
            'num_features': dataset['num_features'],
            'num_classes': dataset['num_classes'],
            'original_edge_index': dataset['original_edge_index'].cpu().tolist(),
            'y': true_labels,
        }
        editable_graph = {'edge_index': dataset['original_edge_index'].cpu().tolist()}

        # Get predictions by live inference on the full graph, so the initial view
        # matches what later re-predictions (after graph edits) compute. The saved
        # package predictions can come from the training graph (GCN cached=True)
        # and carry no confidences.

        initial_predictions = None
        model = get_cached_model(package_meta, dataset['num_features'], dataset['num_classes'])
        if model is not None:
            pyg_data = prepare_torch_data(dataset_meta, None, include_labels=False)
            if pyg_data is not None:
                initial_predictions = run_inference(model, pyg_data)
        if initial_predictions is None:
            print("WARNING: Live inference failed; falling back to saved package predictions")
            initial_predictions = {'preds': package.get('predictions', []),
                                   'confidences': package.get('confidences', [])}
        predictions = initial_predictions['preds']
        if not predictions:
            print("WARNING: No predictions available")

        print(f"SUCCESS: {model_type} model loaded successfully for {dataset_name}")

        print(f"  - Model args: {package.get('model_init_args', {})}")

        print(f"  - Predictions: {len(predictions)} nodes")

        print(f"  - Embeddings: {{'embeddings_2d' in package}}")

        print(f"{ '='*60}\n")

        print(f"✅ {model_type} model loaded successfully for {dataset_name}")

        return package_meta, full_dataset_meta, editable_graph, initial_predictions

    except Exception as e:

        error_msg = f"Failed to load {model_type} model: {str(e)}"

        print(f"\nERROR in load_initial_data: {error_msg}")

        import traceback

        traceback.print_exc()

        print(f"{ '='*60}\n")

        return None, None, None, None


@callback(

    Output(ids.EDITABLE_GRAPH_STORE, 'data', allow_duplicate=True),

    Output(ids.EDGE_STATUS_STORE, 'data'),

    Input(ids.ADD_EDGE_BUTTON_NC, 'n_clicks'), Input(ids.REMOVE_EDGE_BUTTON_NC, 'n_clicks'),

    Input(ids.ADD_EDGE_BUTTON_LP, 'n_clicks'), Input(ids.REMOVE_EDGE_BUTTON_LP, 'n_clicks'),

    State(ids.EDGE_SOURCE_INPUT, 'value'), State(ids.EDGE_TARGET_INPUT, 'value'),

    State(ids.EDITABLE_GRAPH_STORE, 'data'), State(ids.FULL_DATASET_STORE, 'data'),

    prevent_initial_call=True

)

def update_edges(add_clicks, remove_clicks, add_clicks_lp, remove_clicks_lp, source, target, graph_data, dataset):

    if not all([source is not None, target is not None, graph_data, dataset]): raise dash.exceptions.PreventUpdate

    if not (0 <= source < dataset['num_nodes'] and 0 <= target < dataset['num_nodes']):

        return no_update, {'text': 'Node ID out of range.', 'type': 'error'}

    # Copy so the State object is never mutated in place
    edge_index = [list(graph_data['edge_index'][0]), list(graph_data['edge_index'][1])]

    triggered_id = ctx.triggered_id

    status = no_update

    if triggered_id in [ids.ADD_EDGE_BUTTON_NC, ids.ADD_EDGE_BUTTON_LP]:

        if (source, target) not in set(zip(edge_index[0], edge_index[1])):

             edge_index[0].extend([source, target])

             edge_index[1].extend([target, source])

             status = {'text': f'Added edge {source}-{target}. Re-running model...', 'type': 'info'}

        else:

             return no_update, {'text': f'Edge {source}-{target} already exists.', 'type': 'info'}

    elif triggered_id in [ids.REMOVE_EDGE_BUTTON_NC, ids.REMOVE_EDGE_BUTTON_LP]:

        new_edge_index = [[], []]

        removed = False

        for u, v in zip(edge_index[0], edge_index[1]):

            if not ((u == source and v == target) or (u == target and v == source)):

                new_edge_index[0].append(u)

                new_edge_index[1].append(v)

            else:

                removed = True

        if not removed:

            return no_update, {'text': f'Edge {source}-{target} does not exist.', 'type': 'info'}

        edge_index = new_edge_index

        status = {'text': f'Removed edge {source}-{target}. Re-running model...', 'type': 'info'}

    return {'edge_index': edge_index}, status


@callback(

    Output(ids.CURRENT_PREDICTIONS_STORE, 'data', allow_duplicate=True),

    Input(ids.EDITABLE_GRAPH_STORE, 'data'),

    State(ids.FULL_DATASET_STORE, 'data'),

    State(ids.PRECOMPUTED_PACKAGE_STORE, 'data'),

    State(ids.MODEL_DROPDOWN, 'value'),

    prevent_initial_call=True

)
@serialize_model_use
def update_predictions_on_edit(graph_data, dataset_meta, package_meta, model_type):

    try:

        if not all([graph_data, dataset_meta, package_meta]):
            raise dash.exceptions.PreventUpdate

        dataset = get_cached_dataset_if_missing(dataset_meta)
        package = get_cached_package_if_missing(package_meta)
        if not dataset or not package:
            raise dash.exceptions.PreventUpdate

        model = get_cached_model(package_meta, dataset_meta['num_features'], dataset_meta['num_classes'])
        if model is None:
            raise dash.exceptions.PreventUpdate

        pyg_data = prepare_torch_data(dataset_meta, graph_data, include_labels=False)
        res = run_inference(model, pyg_data)
        # res is a dict {'preds': [...], 'confidences': [...]}
        return res

    except Exception as e:

        print(f"ERROR in update_predictions_on_edit: {e}")
        raise dash.exceptions.PreventUpdate


@callback(

    Output(ids.URL, 'href'),

    Input(ids.RESET_BUTTON, 'n_clicks'),

    prevent_initial_call=True

)

def reset_application(n_clicks):

    """Reload the application to reset state."""

    if n_clicks:

        return "/" 

    return no_update


@callback(
    Output(ids.LAYER_2, 'style'),
    Output(ids.LAYER_3, 'style'),
    Input(ids.GLOBAL_RUN_BUTTON, 'n_clicks'),
    Input(ids.SELECTED_LINK_STORE, 'data'),
    Input(ids.SELECTED_NODE_STORE, 'data'),
    Input(ids.ANALYSIS_MODE_STORE, 'data'),
    Input(ids.RESET_BUTTON, 'n_clicks'),
    Input(ids.CURRENT_PREDICTIONS_STORE, 'data'),
    State(ids.LAYER3_TABS, 'value'),
    prevent_initial_call=False
)

def manage_layer_visibility(run_clicks, selected_link, selected_node, analysis_mode, reset_clicks, predictions_data, active_layer3_tab):
    """
    3-LAYER SEQUENTIAL FLOW:
      Step 1 — RUN clicked  : Layer 2 visible, Layer 3 hidden.
      Step 2 — Node or link selected in Layer 2 : Layer 3 becomes visible.

    Layer 3 is intentionally withheld until the user makes an explicit selection
    so the UI guides them through the workflow naturally.
    """
    l2_hidden  = {'display': 'none', 'borderTop': '2px solid #dee2e6', 'paddingTop': '15px', 'marginTop': '5px'}
    l3_hidden  = {'display': 'none', 'borderTop': '2px solid #dee2e6', 'paddingTop': '15px', 'marginTop': '10px'}
    l2_visible = {'display': 'block', 'borderTop': '2px solid #dee2e6', 'paddingTop': '15px', 'marginTop': '10px', 'height': '49.5vh', 'maxHeight': '49.5vh', 'overflow': 'hidden', 'boxSizing': 'border-box'}
    l3_visible = {'display': 'block', 'borderTop': '2px solid #dee2e6', 'paddingTop': '15px', 'marginTop': '10px', 'minHeight': '48vh', 'overflow': 'hidden'}

    # Show the overview as soon as a model is loaded; RUN is only needed to explain.
    if not predictions_data and not run_clicks:
        return l2_hidden, l3_hidden

    return l2_visible, l3_visible


clientside_callback(

    """

    function(n_clicks) {

        if (n_clicks > 0) {

            return ["Processing...", true];

        }

        return [window.dash_clientside.no_update, window.dash_clientside.no_update];

    }

    """,

    Output(ids.GLOBAL_RUN_BUTTON, 'children'),

    Output(ids.GLOBAL_RUN_BUTTON, 'disabled'),

    Input(ids.GLOBAL_RUN_BUTTON, 'n_clicks')

)

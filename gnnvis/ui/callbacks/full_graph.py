"""Full Graph View callback."""

from dash import Input, Output, State, callback

from gnnvis.ui.figures.full_graph import build_full_graph_figure
from gnnvis.ui import ids


@callback(
    Output(ids.FULL_GRAPH_VIEW, 'figure'),
    Input(ids.LAYER3_TABS, 'value'),
    Input(ids.FULL_GRAPH_COLOR_MODE, 'value'),
    Input(ids.FULL_GRAPH_FILTER_METRIC, 'value'),
    Input(ids.FULL_GRAPH_FILTER_RANGE, 'value'),
    Input(ids.SELECTED_NODE_STORE, 'data'),
    Input(ids.CURRENT_PREDICTIONS_STORE, 'data'),
    State(ids.FULL_DATASET_STORE, 'data'),
    State(ids.EDITABLE_GRAPH_STORE, 'data'),
    State(ids.STABILITY_RESULTS_STORE, 'data'),
)
def update_full_graph_view(active_tab, color_mode, filter_metric, filter_range, selected_node, predictions_data, dataset, graph_data, stability_results):
    return build_full_graph_figure(active_tab, color_mode, filter_metric, filter_range, selected_node, predictions_data, dataset, graph_data, stability_results)

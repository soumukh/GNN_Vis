"""Full Graph View callback."""

from dash import Input, Output, State, callback

from gnnvis.ui.figures.full_graph import build_full_graph_figure


@callback(
    Output('full-graph-view', 'figure'),
    Input('layer3-tabs', 'value'),
    Input('full-graph-color-mode', 'value'),
    Input('full-graph-filter-metric', 'value'),
    Input('full-graph-filter-range', 'value'),
    Input('selected-node-store', 'data'),
    Input('current-predictions-store', 'data'),
    State('full-dataset-store', 'data'),
    State('editable-graph-store', 'data'),
    State('stability-results-store', 'data'),
)
def update_full_graph_view(active_tab, color_mode, filter_metric, filter_range, selected_node, predictions_data, dataset, graph_data, stability_results):
    return build_full_graph_figure(active_tab, color_mode, filter_metric, filter_range, selected_node, predictions_data, dataset, graph_data, stability_results)

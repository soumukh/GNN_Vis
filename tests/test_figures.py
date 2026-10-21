"""Figure builders on small synthetic inputs: the properties the UI relies on."""
import numpy as np
import pytest
import torch
from dash import no_update

from gnnvis.ui.components import (cell_details_placeholder, confusion_matrix_style, explanation_cards,
                                  metric_tiles, stability_unsupported_message, subgraph_color_key)
from gnnvis.ui.figures.confusion import link_confusion_figure, node_confusion_figure
from gnnvis.ui.figures.embeddings import get_2d_embeddings, plot_embeddings
from gnnvis.ui.figures.full_graph import build_full_graph_figure
from gnnvis.ui.figures.importance import get_one_hop_neighbors, plot_feature_importance, plot_neighbor_importance
from gnnvis.ui.figures.stability import build_stability_plots
from gnnvis.ui.figures.subgraph import create_subgraph_cytoscape
from gnnvis.ui.theme import (CORRECT_COLOR, INCORRECT_COLOR, class_color_map, class_label, class_name,
                             empty_figure)


def texts(component):
    """All strings inside a Dash component tree."""
    out = []

    def walk(c):
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, (list, tuple)):
            for x in c:
                walk(x)
        elif hasattr(c, 'children'):
            walk(c.children)
    walk(component)
    return ' '.join(out)


# ── theme / components ────────────────────────────────────────────────────────

def test_class_names_and_fallbacks():
    assert class_name('Cora', 3) == 'Neural Networks'
    assert class_label('Cora', 3) == '3: Neural Networks'
    assert class_name('CiteSeer', 0) == 'AI'
    assert class_name('Unknown', 2) == 'Class 2'
    assert class_name('Cora', 99) == 'Class 99'
    assert len(set(class_color_map(7).values())) == 7


def test_empty_figure_has_centred_hint_and_hidden_axes():
    fig = empty_figure('Pick a node')
    assert fig.layout.annotations[0].text == 'Pick a node'
    assert fig.layout.xaxis.visible is False and fig.layout.yaxis.visible is False


def test_components():
    tiles = metric_tiles([('Test acc', 0.79712, 'tip'), ('Count', 5, 't')])
    assert texts(tiles).split() == ['0.797', 'Test', 'acc', '5', 'Count']
    assert 'its edges' in texts(cell_details_placeholder('link_prediction'))
    assert 'the nodes in it' in texts(cell_details_placeholder('node_classification'))
    assert confusion_matrix_style('link_prediction')['minHeight'] < confusion_matrix_style('node_classification')['minHeight']
    assert 'GAT' in stability_unsupported_message('GAT') and 'GCN' in stability_unsupported_message('GAT')
    assert texts(subgraph_color_key('correctness', 'Cora', 7)).split() == ['Correct', 'Incorrect']
    cards = explanation_cards('CPA', None).children
    assert len(cards) == 1 and cards[0].style['width'] == 'calc((100% - 50px) / 2)'
    assert len(explanation_cards('CPA', 'ATT').children) == 2


# ── confusion matrices ────────────────────────────────────────────────────────

def test_node_confusion_matrix_is_numeric_and_labelled():
    fig = node_confusion_figure([0, 0, 1, 2, 2], [0, 1, 1, 2, 0], 3, 'Cora')
    heat = fig.data[0]
    assert np.array(heat.z).tolist() == [[1, 1, 0], [0, 1, 0], [1, 0, 1]]
    assert list(heat.x) == [0, 1, 2]              # numeric: clickData stays parseable
    assert list(fig.layout.xaxis.ticktext) == ['0', '1', '2']
    assert 'Actual: 0: Theory' in heat.hovertext[0][1] and 'Predicted: 1: Reinforcement Learning' in heat.hovertext[0][1]


def test_link_confusion_matrix_cells_and_hover():
    fig = link_confusion_figure([1, 1, 0, 0], [1, 0, 1, 0],
                                [(0.9, 1, 2)], [(0.7, 5, 6)], [(0.2, 3, 4)], [(0.1, 7, 8)])
    heat = fig.data[0]
    assert np.array(heat.z).tolist() == [[1, 1], [1, 1]]
    assert 'True Positives' in heat.customdata[1][1] and '1 → 2' in heat.customdata[1][1]
    assert 'False Positives' in heat.customdata[0][1]


# ── embeddings ────────────────────────────────────────────────────────────────

@pytest.fixture
def emb_inputs(toy_graph, toy_predictions):
    preds, conf = toy_predictions
    rng = np.random.default_rng(0)
    return rng.normal(size=(8, 2)), np.array(preds), toy_graph.y.numpy(), conf


@pytest.mark.parametrize('mode,expected', [
    ('actual', {'0: Theory', '1: Reinforcement Learning', '2: Genetic Algorithms', 'Misclassified', 'Selected'}),
    ('predicted', {'0: Theory', '1: Reinforcement Learning', '2: Genetic Algorithms', 'Misclassified', 'Selected'}),
    ('correctness', {'Correct', 'Incorrect', 'Selected'}),
])
def test_embedding_traces_per_colour_mode(emb_inputs, mode, expected):
    emb, preds, y, conf = emb_inputs
    fig = plot_embeddings(emb, preds, y, '3', class_color_map(3), 3, dataset_name='Cora', color_mode=mode, confidences=conf)
    assert {t.name for t in fig.data} == expected
    ring = [t for t in fig.data if t.name == 'Misclassified']
    if ring:
        assert [c[0] for c in ring[0].customdata] == ['2', '7']   # clickable: carries node ids


def test_embedding_confidence_mode_and_fallback(emb_inputs):
    emb, preds, y, conf = emb_inputs
    fig = plot_embeddings(emb, preds, y, None, class_color_map(3), 3, dataset_name='Cora', color_mode='confidence', confidences=conf)
    assert fig.layout.coloraxis.cmin == 0 and fig.layout.coloraxis.cmax == 1
    # no confidences available -> falls back to the class colouring
    fig = plot_embeddings(emb, preds, y, None, class_color_map(3), 3, dataset_name='Cora', color_mode='confidence')
    assert 'true class' in fig.layout.title.text


def test_embedding_projection_choice_and_fallback(emb_inputs):
    emb, preds, y, conf = emb_inputs
    package = {'embeddings_2d': {'umap': emb.tolist(), 'tsne': (emb * 2).tolist(), 'pca': None}}
    arr, used = get_2d_embeddings(package, 'tsne', return_method=True)
    assert used == 'tsne' and np.allclose(arr, emb * 2)
    # missing projection -> falls back to another precomputed one
    assert get_2d_embeddings(package, 'pca', return_method=True)[1] == 'umap'
    assert np.allclose(get_2d_embeddings(package), emb)          # default unchanged: UMAP
    fig = plot_embeddings(emb, preds, y, None, class_color_map(3), 3, dataset_name='Cora', projection='tsne')
    assert fig.layout.title.text.startswith('Node embeddings, t-SNE')


# ── importance ────────────────────────────────────────────────────────────────

def test_neighbour_table_sorted_and_named(toy_graph, toy_predictions):
    preds, _ = toy_predictions
    nbrs = get_one_hop_neighbors(toy_graph, 2)
    assert nbrs == [0, 1, 3]
    fig = plot_neighbor_importance({0: 0.2, 3: 0.9}, 2, preds=preds, y_labels=toy_graph.y.tolist(),
                                   one_hop_neighbors=nbrs, dataset_name='Cora')
    cells = fig.data[0].cells.values
    assert list(cells[0]) == ['3', '0', '1']              # by importance, missing -> 0
    assert list(cells[1])[0] == 'Reinforcement Learning'  # node 3's true class name


def test_feature_importance_top_k_and_labels():
    fig = plot_feature_importance({i: float(i) for i in range(30)}, 5, top_k=5, title='T', value_label='Raw value')
    assert list(fig.data[0].x) == ['29', '28', '27', '26', '25']
    assert fig.layout.yaxis.title.text == 'Raw value' and fig.layout.title.text == 'T'


# ── subgraph ──────────────────────────────────────────────────────────────────

def test_subgraph_colours_weights_and_centre(toy_graph, toy_predictions):
    preds, _ = toy_predictions
    mask = torch.zeros(toy_graph.edge_index.size(1))
    mask[0] = 0.8   # edge 0-1
    explanation = {'edge_mask': mask.tolist()}
    elements, nodes = create_subgraph_cytoscape(toy_graph, 2, np.array(preds), class_color_map(3), explanation, None,
                                                color_mode='correctness', true_labels=toy_graph.y.tolist())
    assert nodes == {0, 1, 2, 3, 4}                       # 2-hop neighbourhood of node 2
    by_id = {e['data']['id']: e for e in elements if 'id' in e['data']}
    assert by_id['2']['data']['color'] == INCORRECT_COLOR and by_id['0']['data']['color'] == CORRECT_COLOR
    assert 'center-node' in by_id['2']['classes']
    weighted = [e for e in elements if e['data'].get('weight')]
    assert len(weighted) == 1 and weighted[0]['data']['weight'] == 0.8 and 'explained-edge' in weighted[0]['classes']


def test_subgraph_without_selection_is_empty(toy_graph, toy_predictions):
    preds, _ = toy_predictions
    assert create_subgraph_cytoscape(toy_graph, -1, np.array(preds), class_color_map(3), None, None) == ([], set())


# ── full graph ────────────────────────────────────────────────────────────────

@pytest.fixture
def full_graph_inputs(toy_graph, toy_predictions):
    preds, conf = toy_predictions
    dataset = {'name': 'Cora', 'num_nodes': 8, 'num_classes': 3, 'y': toy_graph.y.tolist()}
    graph = {'edge_index': toy_graph.edge_index.tolist()}
    return dataset, graph, {'preds': preds, 'confidences': conf}


def test_full_graph_only_built_when_visible(full_graph_inputs):
    dataset, graph, preds = full_graph_inputs
    assert build_full_graph_figure('tab-gnn', 'predicted', 'confidence', [0, 1], None, preds, dataset, graph, None) is no_update


def test_full_graph_selection_and_filter(full_graph_inputs):
    dataset, graph, preds = full_graph_inputs
    fig = build_full_graph_figure('tab-full-graph', 'correctness', 'confidence', [0, 1], {'id': '3'}, preds, dataset, graph, None)
    names = [t.name for t in fig.data[1:]]
    assert names == ['Correct', 'Incorrect', 'Selected (node 3)']
    shown = sorted(n for t in fig.data[1:] for n in t.customdata)
    assert shown == list(range(8))
    # confidence filter keeps only nodes with confidence in [0.8, 1]
    fig = build_full_graph_figure('tab-full-graph', 'predicted', 'confidence', [0.8, 1], None, preds, dataset, graph, None)
    assert sorted(n for t in fig.data[1:] for n in t.customdata) == [0, 1, 5, 6]


# ── stability ─────────────────────────────────────────────────────────────────

def _stability_results(n=40):
    rng = np.random.default_rng(1)
    return [{'node_idx': i, 'confidence': float(rng.uniform(0.3, 1)), 'stability': float(rng.uniform()),
             'degree': int(rng.integers(1, 10)), 'lipschitz': float(rng.uniform(0, 0.01)),
             'fidelity': float(rng.uniform(-0.1, 0.5)), 'correct': bool(i % 4)} for i in range(n)]


@pytest.mark.parametrize('plot_type', ['scatter', 'hexbin', 'kde'])
@pytest.mark.parametrize('metric', ['confidence', 'degree', 'lipschitz', 'fidelity'])
def test_stability_plots_build_for_every_view(plot_type, metric):
    fig, stats = build_stability_plots(_stability_results(), plot_type, metric)
    assert len(fig.data) > 0
    assert '40' in texts(stats)   # nodes analysed


def test_stability_plots_empty_results():
    fig, stats = build_stability_plots([], 'scatter', 'confidence')
    assert len(fig.data) == 0

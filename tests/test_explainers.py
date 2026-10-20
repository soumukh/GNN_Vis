import pytest
import torch

from gnnvis.explainers import ATTRIBUTION, CAUSAL_PATHS, available_explainers, get_explainer, register_explainer
from gnnvis.explainers.gnnexplainer import neighbour_importance
from gnnvis.models import build_model
from gnnvis.tasks import LINK, LINK_PREDICTION, NODE, NODE_CLASSIFICATION, TASKS, get_task


@pytest.fixture
def gcn(toy_graph):
    torch.manual_seed(0)
    return build_model('GCN', 8, 3, hidden_channels=16, enable_link_prediction=True).eval()


def test_tasks():
    assert set(TASKS) == {NODE_CLASSIFICATION, LINK_PREDICTION}
    assert get_task(LINK_PREDICTION) is LINK and LINK.is_link and not NODE.is_link
    assert get_task(None) is NODE and get_task('unknown') is NODE


def test_every_task_has_both_explainer_kinds():
    for task in TASKS.values():
        for kind in (ATTRIBUTION, CAUSAL_PATHS):
            assert task.explainer(kind).task == task.key
    assert {s.name for s in available_explainers(NODE_CLASSIFICATION)} == {'GNNExplainer', 'CPA-IV'}


def test_registry_rejects_duplicates_and_unknown_kinds():
    with pytest.raises(ValueError):
        register_explainer('GNNExplainer', NODE_CLASSIFICATION, ATTRIBUTION)(lambda *a: None)
    with pytest.raises(ValueError):
        register_explainer('X', NODE_CLASSIFICATION, 'heatmap')
    with pytest.raises(KeyError):
        get_explainer(NODE_CLASSIFICATION, ATTRIBUTION, 'CPA-IV')   # registered, but as causal_paths


def test_neighbour_importance_takes_max_over_incident_edges():
    edge_index = torch.tensor([[0, 1, 0, 2, 3], [1, 0, 2, 0, 1]])
    mask = torch.tensor([0.2, -0.7, 0.1, 0.3, 0.9])
    assert neighbour_importance(mask, edge_index, 0) == pytest.approx({1: 0.7, 2: 0.3})


def test_node_explanation_payload(toy_graph, gcn):
    torch.manual_seed(0)
    out = NODE.explainer(ATTRIBUTION)(gcn, toy_graph, 2, predicted_class=1, epochs=5)
    assert out['explanation_type'] == NODE_CLASSIFICATION and out['node_idx'] == 2
    assert len(out['edge_mask']) == toy_graph.edge_index.size(1)
    assert set(out['node_importance']) == {0, 1, 3}
    assert set(out['feature_importance']) == set(range(8))


def test_link_explanation_and_paths(toy_graph, gcn):
    out = LINK.explainer(ATTRIBUTION)(gcn, toy_graph, (0, 4))
    assert out['explanation_type'] == LINK_PREDICTION and 0 <= out['link_score'] <= 1
    assert {0, 4} <= set(out['subgraph_nodes']) and 0 not in out['node_importance']
    paths = NODE.explainer(CAUSAL_PATHS)(gcn, toy_graph, 3)
    assert paths and all(p['path'][-1] == 3 for p in paths)

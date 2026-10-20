import os

import pytest
import torch

from gnnvis.models import (available_models, build_model, load_model, load_package,
                           model_capabilities, package_path)


def test_registry_lists_models_with_capabilities():
    assert {'GCN', 'GAT'} <= set(available_models())
    assert model_capabilities('GAT')['attention'] is True
    assert model_capabilities('GCN')['stability'] is True
    assert model_capabilities('GAT')['stability'] is False
    assert model_capabilities('nonexistent') == {'attention': False, 'stability': False}


def test_unknown_model_type_raises():
    with pytest.raises(ValueError, match='Unknown model type'):
        build_model('NOPE', 4, 2)


@pytest.mark.parametrize('model_type,args,emb_dim', [
    ('GCN', dict(hidden_channels=16, enable_link_prediction=True), 16),
    ('GAT', dict(hidden_channels=4, heads=2, enable_link_prediction=True), 8),
])
def test_forward_and_link_scores(toy_graph, model_type, args, emb_dim):
    model = build_model(model_type, 8, 3, **args).eval()
    log_probs, emb = model(toy_graph.x, toy_graph.edge_index, return_embeddings=True)
    assert log_probs.shape == (8, 3) and emb.shape == (8, emb_dim)
    assert torch.allclose(log_probs.exp().sum(1), torch.ones(8), atol=1e-5)
    # predict_links is always 1-D, also for a single pair
    assert model.predict_links(emb, torch.tensor([[0], [5]])).shape == (1,)
    assert model.predict_links(emb, torch.tensor([[0, 1, 2], [5, 6, 7]])).shape == (3,)


def test_seeded_build_is_reproducible():
    torch.manual_seed(7); a = build_model('GCN', 8, 3, hidden_channels=16, enable_link_prediction=True)
    torch.manual_seed(7); b = build_model('GCN', 8, 3, hidden_channels=16, enable_link_prediction=True)
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())


def test_round_trip_through_package_is_strict(toy_graph):
    model = build_model('GCN', 8, 3, hidden_channels=16, enable_link_prediction=True).eval()
    package = {'model_state_dict': model.state_dict(), 'model_init_args': model.init_args}
    loaded = load_model(package, 'GCN', 8, 3)
    assert torch.equal(loaded(toy_graph.x, toy_graph.edge_index), model(toy_graph.x, toy_graph.edge_index))
    # weights that do not fit the architecture fail loudly
    bad = dict(package, model_init_args=dict(model.init_args, enable_link_prediction=False))
    with pytest.raises(RuntimeError):
        load_model(bad, 'GCN', 8, 3)


@pytest.mark.skipif(not os.path.exists(package_path('GCN', 'Cora')), reason='trained models not present')
@pytest.mark.parametrize('model_type', ['GCN', 'GAT'])
def test_saved_packages_load_and_match_saved_predictions(model_type):
    from torch_geometric.datasets import Planetoid
    ds = Planetoid(root=os.path.join(os.path.dirname(package_path('GCN', 'Cora')), '..', 'data'), name='Cora')
    data = ds[0]
    package = load_package(model_type, 'Cora')
    model = load_model(package, model_type, ds.num_node_features, ds.num_classes)
    with torch.no_grad():
        preds = model(data.x, data.edge_index).argmax(1)
    assert preds.tolist() == list(package['predictions'])

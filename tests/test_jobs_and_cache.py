import time

import pytest
import torch

from gnnvis import stability_cache as sc
from gnnvis.jobs import CANCELLED, DONE, FAILED, JobManager
from gnnvis.models import build_model


def _wait(job, timeout=10):
    t0 = time.time()
    while job.state == 'running' and time.time() - t0 < timeout:
        time.sleep(0.01)
    return job.state


def test_job_reports_progress_and_result():
    jobs = JobManager()

    def work(n, progress):
        for i in range(n):
            progress(i, n)
        progress(n, n)
        return n * 2

    job = jobs.submit('demo', 'items', work, 5)
    assert _wait(job) == DONE and job.result == 10
    snap = job.snapshot()
    assert (snap['done'], snap['total'], snap['unit']) == (5, 5, 'items')
    assert jobs.take(job.id) is job and jobs.get(job.id) is None


def test_job_cancel_and_failure():
    jobs = JobManager()

    def slow(progress):
        for i in range(1000):
            progress(i, 1000)
            time.sleep(0.005)

    job = jobs.submit('slow', 'items', slow)
    time.sleep(0.05)
    jobs.cancel(job.id)
    assert _wait(job) == CANCELLED and job.done < 1000

    def boom(progress):
        raise RuntimeError('bad input')

    job = jobs.submit('boom', 'items', boom)
    assert _wait(job) == FAILED and 'bad input' in job.error


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, 'CACHE_DIR', str(tmp_path))
    return tmp_path


def _package(seed):
    torch.manual_seed(seed)
    model = build_model('GCN', 8, 3, hidden_channels=16, enable_link_prediction=True).eval()
    return model, {'model_state_dict': model.state_dict()}


def test_fingerprint_tracks_weights():
    _, a = _package(0)
    _, b = _package(0)
    _, c = _package(1)
    assert sc.model_fingerprint(a) == sc.model_fingerprint(b) != sc.model_fingerprint(c)


def test_versioned_cache_hit_and_stale(cache_dir):
    results = [{'node_idx': 0, 'confidence': 0.5, 'stability': 1.0}]
    sc.save(sc.NODE, 'Cora', 'GCN', 0.05, 'abc', results)
    assert sc.load(sc.NODE, 'Cora', 'GCN', 0.05, 'abc')[:2] == (results, sc.HIT)
    assert sc.load(sc.NODE, 'Cora', 'GCN', 0.05, 'retrained')[:2] == (None, sc.STALE)
    assert sc.load(sc.NODE, 'Cora', 'GCN', 0.1, 'abc')[1] == sc.MISSING


def test_legacy_cache_is_verified_then_stamped(cache_dir, toy_graph):
    import pickle
    model, package = _package(0)
    fp = sc.model_fingerprint(package)
    with torch.no_grad():
        probs = model(toy_graph.x, toy_graph.edge_index).exp()
    good = [{'node_idx': i, 'confidence': float(probs[i].max())} for i in range(8)]
    bad = [dict(r, confidence=r['confidence'] - 0.1) for r in good]

    def write_legacy(results):
        with open(sc.results_path(sc.NODE, 'Cora', 'GCN', 0.05), 'wb') as f:
            pickle.dump(results, f)

    verify = lambda r: sc.node_results_match(model, toy_graph, r)  # noqa: E731
    write_legacy(bad)
    assert sc.load(sc.NODE, 'Cora', 'GCN', 0.05, fp, verify=verify)[1] == sc.STALE
    write_legacy(good)
    assert sc.load(sc.NODE, 'Cora', 'GCN', 0.05, fp, verify=verify)[1] == sc.STAMPED
    assert sc.load(sc.NODE, 'Cora', 'GCN', 0.05, fp)[1] == sc.HIT     # metadata written


def test_link_results_match(toy_graph):
    model, _ = _package(0)
    with torch.no_grad():
        _, emb = model(toy_graph.x, toy_graph.edge_index, return_embeddings=True)
        conf = torch.sigmoid(model.predict_links(emb, torch.tensor([[0, 2], [1, 3]])))
    rows = [{'source': 0, 'target': 1, 'confidence': float(conf[0])},
            {'source': 2, 'target': 3, 'confidence': float(conf[1])}]
    assert sc.link_results_match(model, toy_graph, rows)
    assert not sc.link_results_match(model, toy_graph, [dict(rows[0], confidence=0.0)])

"""Stability-analysis callbacks: runs (cached / background job), progress, plots, drill-down."""

import math
import numpy as np
import torch
import traceback
from dash import clientside_callback, Input, Output, State, callback, ctx, html, no_update

from app_stability import run_link_stability_analysis, run_stability_analysis

from gnnvis import stability_cache
from gnnvis.jobs import CANCELLED, DONE, JOBS, RUNNING
from gnnvis.models import model_capabilities
from gnnvis.runtime import LINK_STABILITY_LIVE_SAMPLE, get_cached_package_or_load, prepare_torch_data, private_model
from gnnvis.ui.components import stability_unsupported_message
from gnnvis.ui.figures.stability import build_drilldown_scatter, build_region_highlight, build_stability_plots
from gnnvis.ui import ids
from gnnvis.tasks import get_task

_HIDDEN = {'display': 'none'}
_SHOWN = {'display': 'flex'}


def _sanitize(results):
    """NaN / Inf -> None so the results serialise to JSON."""
    def safe(v):
        return None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v
    return [{k: safe(v) for k, v in r.items()} for r in results]


def _avg_stability(results):
    return sum(r.get('stability', 0) or 0 for r in results) / max(1, len(results))


def _node_summary(results):
    n_correct = sum(1 for r in results if r.get('correct'))
    return f"avg S = {_avg_stability(results):.3f} · correct={n_correct} / wrong={len(results) - n_correct}"


def _clock(seconds):
    seconds = int(round(seconds or 0))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _progress_view(snap):
    done, total = snap['done'], snap['total']
    pct = int(100 * done / total) if total else 0
    text = f"{snap['label']}: {done} / {total or '?'} {snap['unit']} · {_clock(snap['elapsed'])} elapsed"
    if snap['cancel_requested']:
        text += " · cancelling…"
    elif snap['eta'] is not None:
        text += f" · ~{_clock(snap['eta'])} left"
    # data-label: assets/activity.js shows it in the activity badge on every tab
    return html.Div(className='job-progress', **{'data-label': f"{snap['label']} {pct}%"}, children=[
        html.Div(className='job-progress-track', children=html.Div(className='job-progress-fill',
                                                                   style={'width': f'{pct}%'})),
        html.Span(text, className='job-progress-text'),
    ])


def _cancel_page_job(job_data):
    if isinstance(job_data, dict) and job_data.get('id'):
        JOBS.cancel(job_data['id'])


@callback(
    Output(ids.STABILITY_RESULTS_STORE, 'data', allow_duplicate=True),
    Output(ids.STABILITY_STATUS, 'children', allow_duplicate=True),
    Output(ids.STABILITY_JOB_STORE, 'data', allow_duplicate=True),
    Output(ids.STABILITY_JOB_POLL, 'disabled', allow_duplicate=True),
    Output(ids.STABILITY_JOB_PANEL, 'style', allow_duplicate=True),
    Output(ids.RUN_STABILITY_BUTTON, 'disabled', allow_duplicate=True),
    Input(ids.DATASET_DROPDOWN, 'value'),
    Input(ids.MODEL_DROPDOWN, 'value'),
    State(ids.STABILITY_JOB_STORE, 'data'),
    prevent_initial_call=True,
)
def reset_stability_on_model_change(_dataset, model_type, job_data):
    """Stability results belong to one dataset/model: clear them on a switch so
    another model's plots are never shown under the new selection (and stop a
    run that is still going for the old one)."""
    _cancel_page_job(job_data)
    if not model_capabilities(model_type)['stability']:
        return [], stability_unsupported_message(model_type), None, True, _HIDDEN, False
    return [], " ", None, True, _HIDDEN, False


# Immediate feedback for stability runs: the server reply replaces this text.
clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) { return window.dash_clientside.no_update; }
        return "⏳ Starting…";
    }
    """,
    Output(ids.STABILITY_STATUS, 'children', allow_duplicate=True),
    Input(ids.RUN_STABILITY_BUTTON, 'n_clicks'),
    prevent_initial_call=True,
)


def _node_job(model, pyg_data, predictions_data, sigma, sample_size, num_nodes, save_as, stale, progress):
    results = run_stability_analysis(model, pyg_data, predictions_data, sigma, sample_size, progress=progress)
    if not results:
        return [], "⚠️ No results — check console logs."
    results = _sanitize(results)
    if save_as:
        stability_cache.save(stability_cache.NODE, *save_as, results)
    is_full = sample_size == 0 or len(results) == num_nodes
    scope = f"FULL GRAPH ({len(results)} nodes)" if is_full else f"{len(results)} nodes (sampled)"
    note = " · precomputed file was out of date and has been replaced" if (stale and save_as) else ""
    return results, f"✅ {scope} · σ={sigma} · {_node_summary(results)}{note}"


def _link_job(model, pyg_data, sigma, sample, n_total, save_as, stale, progress):
    results = run_link_stability_analysis(model, pyg_data, sigma, sample, progress=progress)
    if not results:
        return [], "⚠️ No results — check console logs."
    results = _sanitize(results)
    is_full = sample >= n_total
    # persist only complete runs, so the cache always means "full graph"
    if save_as and is_full:
        stability_cache.save(stability_cache.LINK, *save_as, results)
    avg = _avg_stability(results)
    if is_full:
        return results, f"✅ FULL GRAPH ({len(results)} edges) · σ={sigma} · avg S = {avg:.3f}"
    why = "the precomputed file is out of date (model retrained)" if stale else "no precomputed file"
    return results, (f"🔹 SAMPLE of {len(results)} / {n_total} edges · σ={sigma} · avg S = {avg:.3f} · "
                     f"{why}; run precompute_link_stability.py for the full graph")


@callback(
    Output(ids.STABILITY_RESULTS_STORE, 'data'),
    Output(ids.STABILITY_STATUS, 'children'),
    Output(ids.STABILITY_JOB_STORE, 'data'),
    Output(ids.STABILITY_JOB_POLL, 'disabled'),
    Output(ids.STABILITY_JOB_PANEL, 'style'),
    Output(ids.RUN_STABILITY_BUTTON, 'disabled'),
    Input(ids.RUN_STABILITY_BUTTON, 'n_clicks'),
    State(ids.STABILITY_SIGMA_SELECTOR, 'value'),
    State(ids.STABILITY_SAMPLE_SIZE, 'value'),
    State(ids.PRECOMPUTED_PACKAGE_STORE, 'data'),
    State(ids.FULL_DATASET_STORE, 'data'),
    State(ids.EDITABLE_GRAPH_STORE, 'data'),
    State(ids.CURRENT_PREDICTIONS_STORE, 'data'),
    State(ids.MODEL_DROPDOWN, 'value'),
    State(ids.ANALYSIS_MODE_STORE, 'data'),
    State(ids.STABILITY_JOB_STORE, 'data'),
    prevent_initial_call=True
)
def run_stability_callback(n_clicks, sigma, sample_size,
                            package, dataset, graph_data, predictions_data, model_type,
                            analysis_mode, job_data):
    """
    Global stability analysis — GNNExplainer Top-k Jaccard (paper-defined).
    Runs on every node (or a random sample when sample_size > 0).
    Independent of any selected node.

    A valid precomputed file answers at once. Otherwise the analysis starts as
    a background job (gnnvis/jobs.py) and poll_stability_job shows its
    progress until it finishes.
    """
    print(f"\n[STABILITY] run_stability_callback fired: n_clicks={n_clicks}")
    idle = (True, _HIDDEN, False)

    # Stability analysis is supported for GCN only; clear any earlier (GCN)
    # results so they are not shown under a GAT label.
    if n_clicks and not model_capabilities(model_type)['stability']:
        return [], stability_unsupported_message(model_type), None, *idle

    if not n_clicks or not package or not dataset or not predictions_data:
        print("[STABILITY] Guard failed — missing data")
        return no_update, "Missing data — load a model first.", no_update, *idle

    running = JOBS.get(job_data['id']) if isinstance(job_data, dict) and job_data.get('id') else None
    if running is not None and running.state == RUNNING:
        return no_update, no_update, no_update, False, _SHOWN, True

    try:
        sigma       = float(sigma)     if sigma       is not None else 0.05
        sample_size = int(sample_size) if sample_size is not None else 0  # 0 = ALL nodes
        model_type  = model_type or 'gcn'
        task        = get_task(analysis_mode)
        kind        = stability_cache.LINK if task.is_link else stability_cache.NODE
        print(f"[STABILITY] {kind}: sigma={sigma}, sample_size={sample_size}, model={model_type}")

        dataset_name = dataset.get('name', 'unknown')
        # Only use / write the cache for full-graph runs on the *unedited*
        # graph: the cache has no notion of user edge edits.
        graph_edited = bool(graph_data) and graph_data.get('edge_index') != dataset.get('original_edge_index')
        use_cache = (sample_size == 0) and not graph_edited
        if graph_edited:
            print("[STABILITY] Graph has been edited — bypassing stability cache")

        model = private_model(package, dataset['num_features'], dataset.get('num_classes', 2))
        if model is None:
            return [], "❌ Could not load model.", None, *idle
        pyg_data = prepare_torch_data(dataset, graph_data, include_labels=not task.is_link)
        if pyg_data is None:
            return [], "❌ Could not construct graph data.", None, *idle

        fingerprint = stability_cache.model_fingerprint(get_cached_package_or_load(package))
        save_as = (dataset_name, model_type, sigma, fingerprint) if use_cache else None
        stale = False
        if use_cache:
            verify = stability_cache.link_results_match if task.is_link else stability_cache.node_results_match
            cached, state, _path = stability_cache.load(kind, dataset_name, model_type, sigma, fingerprint,
                                                        verify=lambda r: verify(model, pyg_data, r))
            stale = state == stability_cache.STALE
            if cached is not None:
                cached = _sanitize(cached)
                unit = 'edges' if task.is_link else 'nodes'
                summary = (f"avg S = {_avg_stability(cached):.3f}" if task.is_link else _node_summary(cached))
                return (cached, f"⚡ FULL GRAPH ({len(cached)} {unit}) [from cache] · σ={sigma} · {summary}",
                        None, *idle)

        if task.is_link:
            ei = pyg_data.edge_index.cpu()
            n_total = int(torch.unique(torch.minimum(ei[0], ei[1]) * pyg_data.num_nodes
                                       + torch.maximum(ei[0], ei[1])).numel())
            sample = sample_size or LINK_STABILITY_LIVE_SAMPLE
            job = JOBS.submit(f"Link stability, σ={sigma}", 'edges', _link_job,
                              model, pyg_data, sigma, sample, n_total, save_as, stale)
        else:
            job = JOBS.submit(f"Node stability, σ={sigma}", 'nodes', _node_job,
                              model, pyg_data, predictions_data, sigma, sample_size, dataset['num_nodes'],
                              save_as, stale)
        why = " (precomputed file is out of date: model retrained)" if stale else ""
        return (no_update, f"⏳ Running{why}…", {'id': job.id}, False, _SHOWN, True)

    except Exception as e:
        traceback.print_exc()
        return [], f"❌ Error: {str(e)}", None, *idle


@callback(
    Output(ids.STABILITY_RESULTS_STORE, 'data', allow_duplicate=True),
    Output(ids.STABILITY_STATUS, 'children', allow_duplicate=True),
    Output(ids.STABILITY_JOB_STORE, 'data', allow_duplicate=True),
    Output(ids.STABILITY_JOB_POLL, 'disabled', allow_duplicate=True),
    Output(ids.STABILITY_JOB_PANEL, 'style', allow_duplicate=True),
    Output(ids.RUN_STABILITY_BUTTON, 'disabled', allow_duplicate=True),
    Output(ids.STABILITY_PROGRESS, 'children'),
    Input(ids.STABILITY_JOB_POLL, 'n_intervals'),
    State(ids.STABILITY_JOB_STORE, 'data'),
    prevent_initial_call=True,
)
def poll_stability_job(_n, job_data):
    """Progress of the page's stability job; its results once it finishes."""
    job_id = job_data.get('id') if isinstance(job_data, dict) else None
    job = JOBS.get(job_id) if job_id else None
    if job is None:
        return no_update, no_update, None, True, _HIDDEN, False, None
    snap = job.snapshot()
    if job.state == RUNNING:
        return no_update, no_update, no_update, False, _SHOWN, True, _progress_view(snap)

    JOBS.take(job_id)
    done = (None, True, _HIDDEN, False, None)
    took = _clock(snap['elapsed'])
    if job.state == DONE:
        results, status = job.result
        return results, f"{status} · {took}", *done
    if job.state == CANCELLED:
        return no_update, f"⏹ Cancelled after {snap['done']} / {snap['total']} {snap['unit']} ({took}).", *done
    return [], f"❌ Error: {snap['error']}", *done


@callback(
    Output(ids.STABILITY_STATUS, 'children', allow_duplicate=True),
    Input(ids.STABILITY_CANCEL_BUTTON, 'n_clicks'),
    State(ids.STABILITY_JOB_STORE, 'data'),
    prevent_initial_call=True,
)
def cancel_stability_job(n_clicks, job_data):
    if not n_clicks:
        return no_update
    _cancel_page_job(job_data)
    return "⏹ Cancelling…"


@callback(
    Output(ids.MAIN_STABILITY_PLOT, 'figure'),
    Output(ids.STABILITY_STATS_PANEL, 'children'),
    Input(ids.STABILITY_RESULTS_STORE, 'data'),
    Input(ids.STABILITY_PLOT_TYPE, 'value'),
    Input(ids.STABILITY_METRIC_SELECTOR, 'value'),
)

def render_stability_plots(results, plot_type, metric_type):
    return build_stability_plots(results, plot_type, metric_type)


_PANEL_HIDDEN = {'flex': '1', 'minWidth': '0', 'flexDirection': 'column', 'gap': '12px', 'display': 'none'}


_PANEL_SHOWN  = {'flex': '1', 'minWidth': '0', 'flexDirection': 'column', 'gap': '12px', 'display': 'flex'}


@callback(
    Output(ids.DRILLDOWN_CLICK_STORE,   'data'),
    Output(ids.DRILLDOWN_HINT,          'children'),
    Output(ids.DRILLDOWN_RIGHT_PANEL,   'style'),
    Input(ids.MAIN_STABILITY_PLOT,      'clickData'),
    Input(ids.DRILLDOWN_RESET_BTN,      'n_clicks'),
    State(ids.STABILITY_PLOT_TYPE,      'value'),
    State(ids.STABILITY_METRIC_SELECTOR,'value'),
    State(ids.STABILITY_RESULTS_STORE,  'data'),
    prevent_initial_call=True,
)
def update_drilldown_store(main_click, _reset, plot_type, metric_type, results):
    # Mode-aware hint text shown in the controls bar
    _hints = {
        'hexbin':  'Click a hexbin cell → see individual nodes in that region',
        'kde':     'Click a KDE contour region → see individual nodes in that region',
        'scatter': 'Click a scatter point → see its nearest neighbors',
    }
    plot_type = (plot_type or 'scatter').lower()
    hint = _hints.get(plot_type, 'Click a plot region to drill down into nodes')

    triggered = ctx.triggered_id
    if triggered == ids.DRILLDOWN_RESET_BTN:
        return None, hint, _PANEL_HIDDEN

    def _extract_xy(click_payload):
        """Return (x, y) floats from clickData, or (None, None) on failure.
        Handles list-format bin values from go.Histogram2d (takes midpoint)."""
        if not click_payload:
            return None, None
        pt = click_payload['points'][0]
        x_val = pt.get('x')
        y_val = pt.get('y')
        if x_val is None or y_val is None:
            return None, None
        def _to_float(v):
            if isinstance(v, (list, tuple)) and len(v) == 2:
                return float((v[0] + v[1]) / 2)
            return float(v)
        try:
            return _to_float(x_val), _to_float(y_val)
        except (TypeError, ValueError):
            return None, None

    def _snap_to_bin_center(x_val, y_val, x_col):
        """For hexbin mode: snap raw click coords to the nearest bin center."""
        if plot_type != 'hexbin' or not results:
            return x_val, y_val
        try:
            import pandas as _pd
            _df = _pd.DataFrame(results).dropna(subset=[x_col, 'stability'])
            _df[x_col]       = _df[x_col].astype(float)
            _df['stability'] = _df['stability'].astype(float)
            if _df.empty:
                return x_val, y_val
            NBINS = 28
            x_min, x_max = _df[x_col].min(), _df[x_col].max()
            y_min, y_max = _df['stability'].min(), _df['stability'].max()
            bw = (x_max - x_min) / NBINS if x_max > x_min else 1.0
            bh = (y_max - y_min) / NBINS if y_max > y_min else 1.0
            bi = int((x_val - x_min) / bw)
            bj = int((y_val - y_min) / bh)
            bi = max(0, min(NBINS - 1, bi))
            bj = max(0, min(NBINS - 1, bj))
            return x_min + (bi + 0.5) * bw, y_min + (bj + 0.5) * bh
        except Exception:
            return x_val, y_val

    def _kde_density_at(x_val, y_val, x_col):
        """For KDE mode: fit gaussian_kde on all nodes and return density at (x, y)."""
        if plot_type != 'kde' or not results:
            return None
        try:
            import pandas as _pd
            from scipy.stats import gaussian_kde as _gkde
            _df = _pd.DataFrame(results).dropna(subset=[x_col, 'stability'])
            _df[x_col]       = _df[x_col].astype(float)
            _df['stability'] = _df['stability'].astype(float)
            if len(_df) < 5:
                return None
            _x = _df[x_col].values
            _y = _df['stability'].values
            if float(np.std(_x)) < 1e-9 or float(np.std(_y)) < 1e-9:
                return None
            kde_fn = _gkde(np.vstack([_x, _y]), bw_method='silverman')
            return float(kde_fn(np.array([[x_val], [y_val]]))[0])
        except Exception:
            return None

    if triggered == ids.MAIN_STABILITY_PLOT:
        if not main_click:
            return no_update, hint, no_update
        x_val, y_val = _extract_xy(main_click)
        if x_val is None:
            return no_update, hint, no_update

        source_col = metric_type if metric_type in ['confidence', 'degree', 'lipschitz'] else 'confidence'

        x_val, y_val = _snap_to_bin_center(x_val, y_val, source_col)
        clicked_density = _kde_density_at(x_val, y_val, source_col)
        return (
            {'x': x_val, 'y': y_val, 'source': source_col,
             'plot_type': plot_type, 'clicked_density': clicked_density},
            hint,
            _PANEL_SHOWN,
        )

    return no_update, hint, no_update


@callback(
    Output(ids.DRILLDOWN_SCATTER,      'figure'),
    Output(ids.DRILLDOWN_PANEL_HEADER, 'children'),
    Input(ids.DRILLDOWN_CLICK_STORE,    'data'),
    Input(ids.DRILLDOWN_EPSILON_SLIDER, 'value'),
    Input(ids.STABILITY_PLOT_TYPE,      'value'),
    State(ids.STABILITY_RESULTS_STORE,  'data'),
    prevent_initial_call=True,
)
def render_drilldown_scatter(click_data, epsilon, plot_type, results):
    return build_drilldown_scatter(click_data, epsilon, plot_type, results)


@callback(
    Output(ids.MAIN_STABILITY_PLOT,   'figure', allow_duplicate=True),
    Input(ids.DRILLDOWN_CLICK_STORE,     'data'),
    Input(ids.DRILLDOWN_EPSILON_SLIDER,  'value'),
    prevent_initial_call=True,
)
def highlight_region_on_main_plots(click_data, epsilon):
    return build_region_highlight(click_data, epsilon)

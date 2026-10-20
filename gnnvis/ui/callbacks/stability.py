"""Stability-analysis callbacks: runs (cached / live), plots, drill-down."""

import numpy as np
import torch
import traceback
from dash import clientside_callback, Input, Output, State, callback, ctx, no_update

from app_stability import get_link_stability_cache_key, get_stability_cache_key, load_link_stability_cache, load_stability_cache, run_link_stability_analysis, run_stability_analysis, save_link_stability_cache, save_stability_cache

from gnnvis.models import model_capabilities
from gnnvis.runtime import LINK_STABILITY_LIVE_SAMPLE, prepare_torch_data, private_model
from gnnvis.ui.components import stability_unsupported_message
from gnnvis.ui.figures.stability import build_drilldown_scatter, build_region_highlight, build_stability_plots
from gnnvis.ui import ids
from gnnvis.tasks import get_task


@callback(
    Output(ids.STABILITY_RESULTS_STORE, 'data', allow_duplicate=True),
    Output(ids.STABILITY_STATUS, 'children', allow_duplicate=True),
    Input(ids.DATASET_DROPDOWN, 'value'),
    Input(ids.MODEL_DROPDOWN, 'value'),
    prevent_initial_call=True,
)
def reset_stability_on_model_change(_dataset, model_type):
    """Stability results belong to one dataset/model: clear them on a switch so
    another model's plots are never shown under the new selection."""
    if not model_capabilities(model_type)['stability']:
        return [], stability_unsupported_message(model_type)
    return [], " "


# Immediate feedback for stability runs: the server reply replaces this text.
# Cached (sigma, model, dataset) combinations return at once; anything else
# runs GNNExplainer twice per node and can take minutes.
clientside_callback(
    """
    function(n_clicks) {
        if (!n_clicks) { return window.dash_clientside.no_update; }
        return "⏳ Running… instant if precomputed; otherwise GNNExplainer runs per node (or on a 100-edge sample for links) and this can take a few minutes.";
    }
    """,
    Output(ids.STABILITY_STATUS, 'children', allow_duplicate=True),
    Input(ids.RUN_STABILITY_BUTTON, 'n_clicks'),
    prevent_initial_call=True,
)


@callback(
    Output(ids.STABILITY_RESULTS_STORE, 'data'),
    Output(ids.STABILITY_STATUS, 'children'),
    Input(ids.RUN_STABILITY_BUTTON, 'n_clicks'),
    State(ids.STABILITY_SIGMA_SELECTOR, 'value'),
    State(ids.STABILITY_SAMPLE_SIZE, 'value'),
    State(ids.PRECOMPUTED_PACKAGE_STORE, 'data'),
    State(ids.FULL_DATASET_STORE, 'data'),
    State(ids.EDITABLE_GRAPH_STORE, 'data'),
    State(ids.CURRENT_PREDICTIONS_STORE, 'data'),
    State(ids.MODEL_DROPDOWN, 'value'),
    State(ids.ANALYSIS_MODE_STORE, 'data'),
    prevent_initial_call=True
)
def run_stability_callback(n_clicks, sigma, sample_size,
                            package, dataset, graph_data, predictions_data, model_type,
                            analysis_mode):
    """
    Global stability analysis — GNNExplainer Top-k Jaccard (paper-defined).
    Runs on every node (or a random sample when sample_size > 0).
    Independent of any selected node.
    """
    print(f"\n[STABILITY] run_stability_callback fired: n_clicks={n_clicks}")

    # Stability analysis is supported for GCN only; clear any earlier (GCN)
    # results so they are not shown under a GAT label.
    if n_clicks and not model_capabilities(model_type)['stability']:
        return [], stability_unsupported_message(model_type)

    if not n_clicks or not package or not dataset or not predictions_data:
        print("[STABILITY] Guard failed — missing data")
        return no_update, "Missing data — load a model first."

    try:
        sigma       = float(sigma)     if sigma       is not None else 0.05
        sample_size = int(sample_size) if sample_size is not None else 0  # 0 = ALL nodes
        print(f"[STABILITY] sigma={sigma}, sample_size={sample_size}, model={model_type}")

        # ── Cache check (only cache full-graph runs; sampled runs are not cached) ──
        dataset_name = dataset.get('name', 'unknown')
        # Only cache full-graph results on the *unedited* graph: the cache key has no
        # notion of user edge edits, so an edited graph must always be recomputed.
        _graph_edited = bool(graph_data) and graph_data.get('edge_index') != dataset.get('original_edge_index')
        _use_cache   = (sample_size == 0) and not _graph_edited
        if _graph_edited:
            print("[STABILITY] Graph has been edited — bypassing stability cache")

        # If analysis mode indicates link prediction, run link-stability branch
        if get_task(analysis_mode).is_link:
            _cache_key = get_link_stability_cache_key(dataset_name, model_type or 'gcn', sigma)
            if _use_cache:
                _cached = load_link_stability_cache(_cache_key)
                if _cached is not None:
                    print(f"[LINK STABILITY] Loading from cache (key={_cache_key})")
                    safe_results = _cached
                    import math
                    def _safe(v):
                        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                            return None
                        return v
                    safe_results = [{k: _safe(v) for k, v in r.items()} for r in safe_results]
                    n_edges = len(safe_results)
                    avg_stab = sum(r.get('stability', 0) or 0 for r in safe_results) / max(1, n_edges)
                    scope_label = f"FULL GRAPH ({n_edges} edges) [from cache]"
                    status = (f"⚡ {scope_label} · σ={sigma} · avg S = {avg_stab:.3f}")
                    return safe_results, status
            else:
                print(f"[LINK STABILITY] Sampled run — skipping cache lookup")

            model = private_model(package, dataset['num_features'], dataset.get('num_classes', 2))
            if model is None:
                return [], "❌ Could not load model."

            pyg_data = prepare_torch_data(dataset, graph_data, include_labels=False)
            if pyg_data is None:
                return [], "❌ Could not construct graph data."

            ei = pyg_data.edge_index.cpu()
            n_total = int(torch.unique(torch.minimum(ei[0], ei[1]) * pyg_data.num_nodes
                                       + torch.maximum(ei[0], ei[1])).numel())
            live_sample = sample_size or LINK_STABILITY_LIVE_SAMPLE
            print(f"[LINK STABILITY] Computing from scratch (key={_cache_key}), "
                  f"{min(live_sample, n_total)} of {n_total} edges")
            results = run_link_stability_analysis(model, pyg_data, sigma, live_sample)
            print(f"[LINK STABILITY] raw results: {len(results)} entries")

            if not results:
                return [], "⚠️ No results — check console logs."

            # sanitize
            import math
            def _safe(v):
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    return None
                return v
            safe_results = [{k: _safe(v) for k, v in r.items()} for r in results]

            n_edges = len(safe_results)
            is_full = live_sample >= n_total
            # persist only complete runs, so the cache always means "full graph"
            if _use_cache and is_full:
                save_link_stability_cache(_cache_key, safe_results)

            avg_stab = sum(r.get('stability', 0) or 0 for r in safe_results) / max(1, n_edges)
            if is_full:
                status = f"✅ FULL GRAPH ({n_edges} edges) · σ={sigma} · avg S = {avg_stab:.3f}"
            else:
                status = (f"🔹 SAMPLE of {n_edges} / {n_total} edges · σ={sigma} · avg S = {avg_stab:.3f} · "
                          f"no precomputed file; run precompute_link_stability.py for the full graph")
            return safe_results, status

        # Default: node stability branch
        _cache_key   = get_stability_cache_key(dataset_name, model_type or 'gcn', sigma)
        if _use_cache:
            _cached = load_stability_cache(_cache_key)
            if _cached is not None:
                print(f"[STABILITY] Loading from cache (key={_cache_key})")
                safe_results = _cached
                import math
                def _safe(v):
                    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                        return None
                    return v
                safe_results = [{k: _safe(v) for k, v in r.items()} for r in safe_results]
                n_correct = sum(1 for r in safe_results if r.get('correct'))
                n_wrong   = len(safe_results) - n_correct
                avg_stab  = sum(r.get('stability', 0) or 0 for r in safe_results) / len(safe_results)
                scope_label = f"FULL GRAPH ({len(safe_results)} nodes) [from cache]"
                status = (f"⚡ {scope_label} · σ={sigma} · "
                          f"avg S = {avg_stab:.3f} · correct={n_correct} / wrong={n_wrong}")
                return safe_results, status
        else:
            print(f"[STABILITY] Sampled run — skipping cache lookup")

        model = private_model(package, dataset['num_features'], dataset['num_classes'])
        if model is None:
            return [], "❌ Could not load model."

        pyg_data = prepare_torch_data(dataset, graph_data, include_labels=True)
        if pyg_data is None:
            return [], "❌ Could not construct graph data."
        preds_len = len(predictions_data.get('preds', []))
        print(f"[STABILITY] x={pyg_data.x.shape}, edge_index={pyg_data.edge_index.shape}, preds={preds_len}")
        print(f"[STABILITY] Computing from scratch (key={_cache_key})")

        results = run_stability_analysis(model, pyg_data, predictions_data, sigma, sample_size)
        print(f"[STABILITY] raw results: {len(results)} entries")

        if not results:
            return [], "⚠️ No results — check console logs."

        # ── Sanitize NaN/Inf → None so JSON serialization works ────────────
        import math
        def _safe(v):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v

        safe_results = [{k: _safe(v) for k, v in r.items()} for r in results]

        import json as _json
        try:
            _json.dumps(safe_results)
        except (TypeError, ValueError) as je:
            print(f"[STABILITY] JSON validation failed: {je}")
            return [], f"❌ Serialization error: {je}"

        # ── Persist to cache (full-graph runs only) ────────────────────────
        if _use_cache:
            save_stability_cache(_cache_key, safe_results)

        n_correct = sum(1 for r in safe_results if r.get('correct'))
        n_wrong   = len(safe_results) - n_correct
        avg_stab  = sum(r.get('stability', 0) or 0 for r in safe_results) / len(safe_results)
        print(f"[STABILITY] correct={n_correct}, wrong={n_wrong}, avg_stability={avg_stab:.3f}")
        print(f"[STABILITY] first result: {safe_results[0]}")

        is_full = (sample_size == 0 or len(safe_results) == dataset['num_nodes'])
        scope_label = f"FULL GRAPH ({len(safe_results)} nodes)" if is_full else f"{len(safe_results)} nodes (sampled)"
        status = (f"✅ {scope_label} · σ={sigma} · "
                  f"avg S = {avg_stab:.3f} · correct={n_correct} / wrong={n_wrong}")
        return safe_results, status

    except Exception as e:
        traceback.print_exc()
        return [], f"❌ Error: {str(e)}"


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

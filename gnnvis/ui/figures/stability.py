"""Stability-analysis plots: main scatter / hexbin / KDE, statistics panel, drill-down."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import html
from gnnvis.ui.theme import CORRECT_COLOR, INCORRECT_COLOR


def build_stability_plots(results, plot_type, metric_type):
    plot_type = (plot_type or 'scatter').strip().lower()
    metric_type = (metric_type or 'confidence').strip().lower()
    print(f"\n[STABILITY PLOT] plot_type={repr(plot_type)}  results_len={len(results) if results else 0}")

    empty_layout = dict(margin=dict(l=40, r=20, t=40, b=40), height=280,
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    if not results:
        print("[STABILITY PLOT] results is empty/None — returning empty titled figures")

        def _empty_fig(title):
            f = go.Figure(layout={**empty_layout, 'title': dict(text=title, font=dict(size=13))})
            f.add_annotation(
                xref='paper', yref='paper', x=0.5, y=0.5,
                text='No valid stability data found',
                showarrow=False, font=dict(size=13, color='#999'),
                xanchor='center', yanchor='middle'
            )
            return f

        title_map = {
            'confidence': 'Confidence vs Stability',
            'degree': 'Degree vs Stability',
            'lipschitz': 'Lipschitz vs Stability',
            'fidelity': 'Correct vs Wrong Stability'
        }
        return (
            _empty_fig(title_map.get(metric_type, 'Stability Plot')),
            html.Div("No data.", style={'color': '#999', 'textAlign': 'center', 'paddingTop': '20px'})
        )

    df = pd.DataFrame(results)
    print(f"[STABILITY PLOT] ── DataFrame loaded ──────────────────────────────")
    print(f"[STABILITY PLOT] columns : {list(df.columns)}")
    print(f"[STABILITY PLOT] shape   : {df.shape}")
    print(f"[STABILITY PLOT] nulls   : { {c: int(df[c].isna().sum()) for c in df.columns} }")

    # Normalize link-oriented results to node-oriented plotting schema
    if 'node_idx' not in df.columns and 'source' in df.columns and 'target' in df.columns:
        try:
            df['node_idx'] = df['source'].astype(str) + '-' + df['target'].astype(str)
            # map degree_sum -> degree for link results if present
            if 'degree' not in df.columns and 'degree_sum' in df.columns:
                df['degree'] = df['degree_sum']
            print("[STABILITY PLOT] Normalized link result columns: created 'node_idx' and mapped 'degree_sum'->'degree'")
        except Exception:
            print("[STABILITY PLOT] Could not normalize link result columns")
    for _req_col in ('confidence', 'stability', 'degree', 'lipschitz'):
        if _req_col in df.columns:
            _nv = int(df[_req_col].notna().sum())
            _dtype = str(df[_req_col].dtype)
            _sample = df[_req_col].dropna().head(3).tolist()
            print(f"[STABILITY PLOT]   {_req_col}: {_nv}/{len(df)} non-null  dtype={_dtype}  sample={_sample}")
        else:
            print(f"[STABILITY PLOT]   ⚠ MISSING column: '{_req_col}'")
    print(f"[STABILITY PLOT] ─────────────────────────────────────────────────")
    print(f"[STABILITY PLOT] correct distribution: {df['correct'].value_counts().to_dict()}")

    # Force correct column to Python bool (JSON round-trip can sometimes produce int 0/1)
    df['correct'] = df['correct'].apply(lambda v: bool(v))

    color_map = {True: CORRECT_COLOR, False: INCORRECT_COLOR}
    label_map = {True: 'Correct', False: 'Wrong'}
    df['color'] = df['correct'].map(color_map)
    df['label'] = df['correct'].map(label_map)

    import math as _math
    from scipy.stats import t as _t_dist, linregress as _linregress

    def _pearson_r(a, b):
        mask = ~(np.isnan(a.astype(float)) | np.isnan(b.astype(float)))
        a, b = a[mask].astype(float), b[mask].astype(float)
        if len(a) < 3:
            return float('nan'), float('nan')
        r = float(np.corrcoef(a, b)[0, 1])
        if abs(r) >= 1.0:
            return r, 0.0
        t_s = r * _math.sqrt(len(a) - 2) / _math.sqrt(1 - r * r)
        return r, float(2 * _t_dist.sf(abs(t_s), df=len(a) - 2))

    def _scatter_with_trend(x_col, y_col, x_title, y_title, title):
        # Column existence check
        for _c in (x_col, y_col):
            if _c not in df.columns:
                print(f"[STABILITY PLOT] ⚠ {title}: column '{_c}' not in df — skipping")
                f = go.Figure()
                f.update_layout(title=dict(text=title, font=dict(size=13)),
                                margin=dict(l=45, r=15, t=42, b=40), height=290,
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                f.add_annotation(xref='paper', yref='paper', x=0.5, y=0.5,
                                 text=f"Column '{_c}' not available in results",
                                 showarrow=False, font=dict(size=13, color='#c00'),
                                 xanchor='center', yanchor='middle')
                return f

        n_x_null = int(df[x_col].isna().sum())
        n_y_null = int(df[y_col].isna().sum())
        sub = df.dropna(subset=[x_col, y_col]).copy()
        sub[x_col] = sub[x_col].astype(float)
        sub[y_col] = sub[y_col].astype(float)
        print(f"[STABILITY PLOT] {title}: {len(sub)}/{len(df)} valid rows after dropna "
              f"({x_col} nulls={n_x_null}, {y_col} nulls={n_y_null})")

        fig = go.Figure()
        if sub.empty:
            if n_x_null == len(df):
                reason = f"All {len(df)} values in '{x_col}' are null"
            elif n_y_null == len(df):
                reason = f"All {len(df)} values in '{y_col}' are null"
            else:
                reason = f"No rows with valid ({x_col}, {y_col}) pair"
            print(f"[STABILITY PLOT] ⚠ {title} empty: {reason}")
            fig.update_layout(title=dict(text=title, font=dict(size=13)),
                              margin=dict(l=45, r=15, t=42, b=40), height=290,
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            fig.add_annotation(xref='paper', yref='paper', x=0.5, y=0.5,
                               text=reason, showarrow=False,
                               font=dict(size=12, color='#999'),
                               xanchor='center', yanchor='middle')
            return fig

        has_lip = 'lipschitz' in sub.columns
        for bool_val in [True, False]:
            grp = sub[sub['correct'] == bool_val]
            if grp.empty:
                continue
            lip_vals = (grp['lipschitz'].tolist() if has_lip
                        else [float('nan')] * len(grp))
            custom = list(zip(
                grp['node_idx'].tolist(),
                grp['degree'].tolist(),
                grp['confidence'].tolist(),
                lip_vals,
            ))
            hover = (
                f'<b>Node %{{customdata[0]}}</b><br>'
                f'{x_title}: %{{x:.4f}}<br>'
                f'Stability: %{{y:.4f}}<br>'
                f'Degree: %{{customdata[1]}}<br>'
                f'Confidence: %{{customdata[2]:.4f}}'
                + ('<br>Lipschitz: %{customdata[3]:.4f}' if has_lip else '')
                + '<extra></extra>'
            )
            fig.add_trace(go.Scatter(
                x=grp[x_col].tolist(), y=grp[y_col].tolist(),
                mode='markers',
                name=label_map[bool_val],
                marker=dict(color=color_map[bool_val], size=7, opacity=0.72,
                            line=dict(width=0.4, color='white')),
                hovertemplate=hover,
                customdata=custom,
            ))
            print(f"[STABILITY PLOT]   added {label_map[bool_val]} trace: {len(grp)} points")

        if len(sub) >= 3:
            try:
                slope, intercept, _, _, _ = _linregress(sub[x_col].values, sub[y_col].values)
                x_range = np.linspace(sub[x_col].min(), sub[x_col].max(), 100)
                fig.add_trace(go.Scatter(
                    x=x_range.tolist(), y=(slope * x_range + intercept).tolist(),
                    mode='lines', name='Trend',
                    line=dict(color='#333', width=1.5, dash='dash'),
                    hoverinfo='skip',
                ))
            except ValueError:
                pass

        r, p = _pearson_r(sub[x_col].values, sub[y_col].values)
        r_str = f'r = {r:.3f}' if not _math.isnan(r) else 'r = N/A'
        p_str = f'p = {p:.4f}' if not _math.isnan(p) else ''
        sig_str = ' *' if not _math.isnan(p) and p < 0.05 else ''
        fig.add_annotation(
            xref='paper', yref='paper', x=0.98, y=0.04,
            text=f'{r_str},  {p_str}{sig_str}',
            showarrow=False, font=dict(size=11, color='#333'),
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#ccc', borderwidth=1,
            align='right', xanchor='right',
        )
        fig.update_layout(
            title=dict(text=title, font=dict(size=13)),
            xaxis_title=x_title, yaxis_title=y_title,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='#ccc', borderwidth=1, font=dict(size=11)),
            margin=dict(l=45, r=15, t=42, b=40), height=290,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        )
        fig.update_xaxes(showgrid=True, gridcolor='#eee')
        fig.update_yaxes(showgrid=True, gridcolor='#eee')
        # Log scale only when all visible x values are strictly positive
        if x_col in ('degree', 'lipschitz') and sub[x_col].min() > 0:
            fig.update_xaxes(type='log')
            print(f"[STABILITY PLOT] {title}: log x-axis applied (min={sub[x_col].min():.4f})")
        elif x_col in ('degree', 'lipschitz'):
            print(f"[STABILITY PLOT] {title}: log x-axis SKIPPED (min={sub[x_col].min():.4f} ≤ 0)")
        return fig

    # ── Hexbin helper (Histogram2d density + invisible scatter overlay) ───────
    def _hexbin_plot(x_col, y_col, x_title, y_title, title):
        # Column existence check
        for _c in (x_col, y_col):
            if _c not in df.columns:
                print(f"[STABILITY PLOT] ⚠ {title}: column '{_c}' not in df — skipping")
                f = go.Figure()
                f.update_layout(title=dict(text=title, font=dict(size=13)),
                                margin=dict(l=45, r=15, t=42, b=40), height=290,
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                f.add_annotation(xref='paper', yref='paper', x=0.5, y=0.5,
                                 text=f"Column '{_c}' not available in results",
                                 showarrow=False, font=dict(size=13, color='#c00'),
                                 xanchor='center', yanchor='middle')
                return f

        n_x_null = int(df[x_col].isna().sum())
        n_y_null = int(df[y_col].isna().sum())
        sub = df.dropna(subset=[x_col, y_col]).copy()
        sub[x_col] = sub[x_col].astype(float)
        sub[y_col] = sub[y_col].astype(float)
        print(f"[STABILITY PLOT] {title}: {len(sub)}/{len(df)} valid rows after dropna "
              f"({x_col} nulls={n_x_null}, {y_col} nulls={n_y_null})")

        fig = go.Figure()
        if sub.empty:
            if n_x_null == len(df):
                reason = f"All {len(df)} values in '{x_col}' are null"
            elif n_y_null == len(df):
                reason = f"All {len(df)} values in '{y_col}' are null"
            else:
                reason = f"No rows with valid ({x_col}, {y_col}) pair"
            print(f"[STABILITY PLOT] ⚠ {title} empty: {reason}")
            fig.update_layout(title=dict(text=title, font=dict(size=13)),
                              margin=dict(l=45, r=15, t=42, b=40), height=290,
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            fig.add_annotation(xref='paper', yref='paper', x=0.5, y=0.5,
                               text=reason, showarrow=False,
                               font=dict(size=12, color='#999'),
                               xanchor='center', yanchor='middle')
            return fig

        # Density heatmap (rectangular binning, Viridis)
        fig.add_trace(go.Histogram2d(
            x=sub[x_col].tolist(), y=sub[y_col].tolist(),
            colorscale=[
                [0.0,  'rgba(68,1,84,0)'],    # transparent for empty bins
                [0.001,'#440154'],
                [0.25, '#31688e'],
                [0.5,  '#35b779'],
                [0.75, '#fde725'],
                [1.0,  '#fde725'],
            ],
            nbinsx=28, nbinsy=28,
            zsmooth='best',
            colorbar=dict(
                title=dict(text='Node count', font=dict(size=11)),
                thickness=12, len=0.85,
            ),
            hovertemplate=(
                f'{x_title}: %{{x:.3f}}<br>'
                f'Stability: %{{y:.3f}}<br>'
                'Count: %{z}<br>'
                '<i>Click to explore this region</i>'
                '<extra></extra>'
            ),
            name='Density',
        ))

        # Invisible scatter overlay — enables precise click coordinates
        has_lip = 'lipschitz' in sub.columns
        lip_vals = sub['lipschitz'].tolist() if has_lip else [float('nan')] * len(sub)
        custom = list(zip(
            sub['node_idx'].tolist(),
            sub['degree'].tolist(),
            sub['confidence'].tolist(),
            lip_vals,
        ))
        fig.add_trace(go.Scatter(
            x=sub[x_col].tolist(), y=sub[y_col].tolist(),
            mode='markers',
            marker=dict(size=6, color='rgba(0,0,0,0)', opacity=0),
            hovertemplate=(
                f'<b>Node %{{customdata[0]}}</b><br>'
                f'{x_title}: %{{x:.4f}}<br>Stability: %{{y:.4f}}<br>'
                f'Degree: %{{customdata[1]}}<br>Confidence: %{{customdata[2]:.4f}}'
                + ('<br>Lipschitz: %{customdata[3]:.4f}' if has_lip else '')
                + '<br><i>Click to drill down</i><extra></extra>'
            ),
            customdata=custom,
            showlegend=False,
            name='_overlay',
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=13)),
            xaxis_title=x_title, yaxis_title=y_title,
            margin=dict(l=45, r=15, t=42, b=40), height=290,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        if x_col in ('degree', 'lipschitz') and sub[x_col].min() > 0:
            fig.update_xaxes(type='log')
            print(f"[STABILITY PLOT] {title}: log x-axis applied (min={sub[x_col].min():.4f})")
        elif x_col in ('degree', 'lipschitz'):
            print(f"[STABILITY PLOT] {title}: log x-axis SKIPPED (min={sub[x_col].min():.4f} ≤ 0)")
        return fig

    # ── KDE helper (gaussian_kde contour + invisible scatter overlay) ─────────
    def _kde_plot(x_col, y_col, x_title, y_title, title):
        from scipy.stats import gaussian_kde as _kde

        # Column existence check
        for _c in (x_col, y_col):
            if _c not in df.columns:
                print(f"[STABILITY PLOT] ⚠ {title}: column '{_c}' not in df — skipping")
                f = go.Figure()
                f.update_layout(title=dict(text=title, font=dict(size=13)),
                                margin=dict(l=45, r=15, t=42, b=40), height=290,
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                f.add_annotation(xref='paper', yref='paper', x=0.5, y=0.5,
                                 text=f"Column '{_c}' not available in results",
                                 showarrow=False, font=dict(size=13, color='#c00'),
                                 xanchor='center', yanchor='middle')
                return f

        n_x_null = int(df[x_col].isna().sum())
        n_y_null = int(df[y_col].isna().sum())
        sub = df.dropna(subset=[x_col, y_col]).copy()
        sub[x_col] = sub[x_col].astype(float)
        sub[y_col] = sub[y_col].astype(float)
        print(f"[STABILITY PLOT] {title}: {len(sub)}/{len(df)} valid rows after dropna "
              f"({x_col} nulls={n_x_null}, {y_col} nulls={n_y_null})")

        fig = go.Figure()
        if sub.empty or len(sub) < 5:
            if n_x_null == len(df):
                reason = f"All {len(df)} values in '{x_col}' are null"
            elif n_y_null == len(df):
                reason = f"All {len(df)} values in '{y_col}' are null"
            elif len(sub) < 5:
                reason = f"Not enough data for KDE (need ≥ 5, got {len(sub)})"
            else:
                reason = f"No rows with valid ({x_col}, {y_col}) pair"
            print(f"[STABILITY PLOT] ⚠ {title} empty/insufficient: {reason}")
            fig.update_layout(title=dict(text=title, font=dict(size=13)),
                              margin=dict(l=45, r=15, t=42, b=40), height=290,
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            fig.add_annotation(xref='paper', yref='paper', x=0.5, y=0.5,
                               text=reason, showarrow=False,
                               font=dict(size=12, color='#999'),
                               xanchor='center', yanchor='middle')
            return fig

        x_vals = sub[x_col].values
        y_vals = sub[y_col].values
        try:
            if np.std(x_vals) < 1e-9 or np.std(y_vals) < 1e-9:
                raise ValueError("Zero variance in one axis — KDE singular")
            kde_fn = _kde(np.vstack([x_vals, y_vals]), bw_method='silverman')
            # Evaluate on a fine grid
            xi = np.linspace(x_vals.min(), x_vals.max(), 100)
            yi = np.linspace(y_vals.min(), y_vals.max(), 100)
            xi_g, yi_g = np.meshgrid(xi, yi)
            zi = kde_fn(np.vstack([xi_g.ravel(), yi_g.ravel()])).reshape(xi_g.shape)
            zi = zi / zi.max() if zi.max() > 0 else zi  # normalize to [0, 1]

            fig.add_trace(go.Contour(
                x=xi.tolist(), y=yi.tolist(), z=zi.tolist(),
                colorscale='Viridis',
                ncontours=22,
                contours=dict(
                    coloring='fill',
                    showlabels=False,
                ),
                colorbar=dict(
                    title=dict(text='Density', font=dict(size=11)),
                    thickness=12, len=0.85,
                ),
                line=dict(width=0.5, color='rgba(255,255,255,0.4)'),
                hovertemplate=(
                    f'{x_title}: %{{x:.3f}}<br>'
                    'Stability: %{y:.3f}<br>'
                    'Density: %{z:.5f}<br>'
                    '<i>Click to explore this region</i>'
                    '<extra></extra>'
                ),
                name='KDE',
            ))
        except Exception as kde_err:
            print(f"[STABILITY PLOT] KDE failed for {title}: {kde_err}")
            fig.add_annotation(xref='paper', yref='paper', x=0.5, y=0.5,
                               text=f'KDE failed: {kde_err}',
                               showarrow=False, font=dict(size=11, color='#999'),
                               xanchor='center', yanchor='middle')

        # Invisible scatter overlay — gives exact node coordinates on click
        # Opacity ~0 so invisible, but renders in DOM so clickData fires reliably
        has_lip = 'lipschitz' in sub.columns
        lip_vals = sub['lipschitz'].tolist() if has_lip else [float('nan')] * len(sub)
        custom = list(zip(
            sub['node_idx'].tolist(),
            sub['degree'].tolist(),
            sub['confidence'].tolist(),
            lip_vals,
        ))
        fig.add_trace(go.Scatter(
            x=sub[x_col].tolist(), y=sub[y_col].tolist(),
            mode='markers',
            marker=dict(size=7, color='rgba(0,0,0,0)', opacity=0),
            hovertemplate=(
                f'<b>Node %{{customdata[0]}}</b><br>'
                f'{x_title}: %{{x:.4f}}<br>Stability: %{{y:.4f}}<br>'
                f'Degree: %{{customdata[1]}}<br>Confidence: %{{customdata[2]:.4f}}'
                + ('<br>Lipschitz: %{customdata[3]:.4f}' if has_lip else '')
                + '<br><i>Click to drill down</i><extra></extra>'
            ),
            customdata=custom,
            showlegend=False,
            name='_overlay',
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=13)),
            xaxis_title=x_title, yaxis_title=y_title,
            margin=dict(l=45, r=15, t=42, b=40), height=290,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        if x_col in ('degree', 'lipschitz') and sub[x_col].min() > 0:
            fig.update_xaxes(type='log')
            print(f"[STABILITY PLOT] {title}: log x-axis applied (min={sub[x_col].min():.4f})")
        elif x_col in ('degree', 'lipschitz'):
            print(f"[STABILITY PLOT] {title}: log x-axis SKIPPED (min={sub[x_col].min():.4f} ≤ 0)")
        return fig

    # Paper-defined method: GNNExplainer Top-k Jaccard
    # Y-axis is always "Top-k Jaccard"; Plot 3 is always Lipschitz vs Stability
    stab_label = 'Stability'

    # ── Explicit per-plot dispatch (no closure ambiguity) ─────────────────────
    print(f"[STABILITY PLOT] DISPATCH BRANCH → {plot_type.upper()}")

    def _safe_plot(fn, *args):
        """Call fn(*args); on any exception return an error-annotation figure."""
        try:
            return fn(*args)
        except Exception as _pe:
            print(f"[STABILITY PLOT] ⚠ plot error in {fn.__name__}: {_pe}")
            _ef = go.Figure()
            _ef.update_layout(margin=dict(l=45, r=15, t=42, b=40), height=290,
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            _ef.add_annotation(xref='paper', yref='paper', x=0.5, y=0.5,
                               text=f'Plot error: {_pe}',
                               showarrow=False, font=dict(size=11, color='#c00'),
                               xanchor='center', yanchor='middle')
            return _ef

    if plot_type == 'hexbin':
        print("[STABILITY PLOT]   → building HEXBIN plots")
        conf_fig = _safe_plot(_hexbin_plot, 'confidence', 'stability', 'Confidence',  stab_label, 'Confidence vs Stability')
        deg_fig  = _safe_plot(_hexbin_plot, 'degree',     'stability', 'Node Degree', stab_label, 'Degree vs Stability')
    elif plot_type == 'kde':
        print("[STABILITY PLOT]   → building KDE plots")
        conf_fig = _safe_plot(_kde_plot, 'confidence', 'stability', 'Confidence',  stab_label, 'Confidence vs Stability')
        deg_fig  = _safe_plot(_kde_plot, 'degree',     'stability', 'Node Degree', stab_label, 'Degree vs Stability')
    else:
        print("[STABILITY PLOT]   → building SCATTER plots")
        conf_fig = _safe_plot(_scatter_with_trend, 'confidence', 'stability', 'Confidence',  stab_label, 'Confidence vs Stability')
        deg_fig  = _safe_plot(_scatter_with_trend, 'degree',     'stability', 'Node Degree', stab_label, 'Degree vs Stability')

    # Plot 3: Lipschitz sensitivity vs Stability
    # Require ≥5 non-null lipschitz values — otherwise show an informative placeholder
    _lip_n_valid = int(df['lipschitz'].notna().sum()) if 'lipschitz' in df.columns else 0
    lip_col = 'lipschitz' if _lip_n_valid >= 5 else None
    print(f"[STABILITY PLOT] lipschitz: column_present={'lipschitz' in df.columns}, "
          f"n_valid={_lip_n_valid}, using={lip_col is not None}")
    if lip_col:
        if plot_type == 'hexbin':
            lip_fig = _safe_plot(_hexbin_plot, 'lipschitz', 'stability', 'Lipschitz  L', stab_label, 'Lipschitz vs Stability')
        elif plot_type == 'kde':
            lip_fig = _safe_plot(_kde_plot, 'lipschitz', 'stability', 'Lipschitz  L', stab_label, 'Lipschitz vs Stability')
        else:
            lip_fig = _safe_plot(_scatter_with_trend, 'lipschitz', 'stability', 'Lipschitz  L', stab_label, 'Lipschitz vs Stability')
    else:
        _lip_msg = ('No Lipschitz data available'
                    if 'lipschitz' not in df.columns
                    else f'Lipschitz unavailable ({_lip_n_valid} non-null values — need ≥ 5)')
        print(f"[STABILITY PLOT] ⚠ Lipschitz plot skipped: {_lip_msg}")
        f = go.Figure(layout={**empty_layout, 'title': dict(text='Lipschitz vs Stability', font=dict(size=13))})
        f.add_annotation(xref='paper', yref='paper', x=0.5, y=0.5,
                         text=_lip_msg,
                         showarrow=False, font=dict(size=13, color='#999'), xanchor='center', yanchor='middle')
        lip_fig = f

    # Plot 4: Correct vs Wrong Stability — varies by plot_type
    def _jitter_correct_stability():
        sub = df.dropna(subset=['stability']).copy()
        title = 'Correct vs Wrong Stability'

        f = go.Figure()
        if sub.empty:
            f.update_layout(**{**empty_layout, 'title': dict(text=title, font=dict(size=13))})
            f.add_annotation(xref='paper', yref='paper', x=0.5, y=0.5,
                             text='No valid stability data found',
                             showarrow=False, font=dict(size=13, color='#999'),
                             xanchor='center', yanchor='middle')
            return f

        rng = np.random.default_rng(42)
        for bool_val in [True, False]:
            grp = sub[sub['correct'] == bool_val]
            if grp.empty:
                continue
            x_jitter = rng.uniform(-0.15, 0.15, size=len(grp)) + (1 if bool_val else 0)
            f.add_trace(go.Scatter(
                x=x_jitter.tolist(), y=grp['stability'].tolist(),
                mode='markers',
                name=label_map[bool_val],
                marker=dict(color=color_map[bool_val], size=7, opacity=0.7),
                hovertemplate='Node %{customdata}<br>Stability: %{y:.3f}<extra></extra>',
                customdata=grp['node_idx'].tolist()
            ))

        f.update_layout(
            title=dict(text=title, font=dict(size=13)),
            xaxis=dict(tickvals=[0, 1], ticktext=['Wrong', 'Correct'], title='Prediction', range=[-0.5, 1.5]),
            yaxis_title=stab_label,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='#ccc', borderwidth=1, font=dict(size=11)),
            margin=dict(l=45, r=15, t=42, b=40), height=290,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        f.update_xaxes(showgrid=True, gridcolor='#eee')
        f.update_yaxes(showgrid=True, gridcolor='#eee')
        return f

    # Plot 4 hexbin variant — overlaid stability histograms per class
    def _hexbin_correct_stability():
        sub = df.dropna(subset=['stability']).copy()
        title = 'Correct vs Wrong Stability'
        f = go.Figure()
        if sub.empty:
            f.update_layout(title=dict(text=title, font=dict(size=13)),
                            margin=dict(l=45, r=15, t=42, b=40), height=290,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            f.add_annotation(xref='paper', yref='paper', x=0.5, y=0.5,
                             text='No valid stability data found',
                             showarrow=False, font=dict(size=13, color='#999'),
                             xanchor='center', yanchor='middle')
            return f

        for bool_val in [True, False]:
            grp = sub[sub['correct'] == bool_val]
            if grp.empty:
                continue
            f.add_trace(go.Histogram(
                x=grp['stability'].tolist(),
                name=label_map[bool_val],
                marker_color=color_map[bool_val],
                opacity=0.65,
                nbinsx=30,
            ))
        f.update_layout(
            barmode='overlay',
            title=dict(text=title, font=dict(size=13)),
            xaxis_title=stab_label, yaxis_title='Count',
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='#ccc', borderwidth=1, font=dict(size=11)),
            margin=dict(l=45, r=15, t=42, b=40), height=290,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        f.update_xaxes(showgrid=True, gridcolor='#eee')
        f.update_yaxes(showgrid=True, gridcolor='#eee')
        return f

    # Plot 4 KDE variant — violin plots per class
    def _kde_correct_stability():
        sub = df.dropna(subset=['stability']).copy()
        title = 'Correct vs Wrong Stability'
        f = go.Figure()
        if sub.empty:
            f.update_layout(title=dict(text=title, font=dict(size=13)),
                            margin=dict(l=45, r=15, t=42, b=40), height=290,
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            f.add_annotation(xref='paper', yref='paper', x=0.5, y=0.5,
                             text='No valid stability data found',
                             showarrow=False, font=dict(size=13, color='#999'),
                             xanchor='center', yanchor='middle')
            return f

        for bool_val in [True, False]:
            grp = sub[sub['correct'] == bool_val]
            if grp.empty:
                continue
            f.add_trace(go.Violin(
                y=grp['stability'].tolist(),
                name=label_map[bool_val],
                box=dict(visible=True),
                meanline=dict(visible=True),
                fillcolor=color_map[bool_val],
                line=dict(color=color_map[bool_val]),
                opacity=0.7,
            ))
        f.update_layout(
            title=dict(text=title, font=dict(size=13)),
            yaxis_title=stab_label,
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)',
                        bordercolor='#ccc', borderwidth=1, font=dict(size=11)),
            margin=dict(l=45, r=15, t=42, b=40), height=290,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        f.update_xaxes(showgrid=True, gridcolor='#eee')
        f.update_yaxes(showgrid=True, gridcolor='#eee')
        return f

    if plot_type == 'hexbin':
        print("[STABILITY PLOT]   → plot4 branch: HEXBIN histogram")
        cor_fig = _safe_plot(_hexbin_correct_stability)
    elif plot_type == 'kde':
        print("[STABILITY PLOT]   → plot4 branch: KDE violin")
        cor_fig = _safe_plot(_kde_correct_stability)
    else:
        print("[STABILITY PLOT]   → plot4 branch: SCATTER jitter")
        cor_fig = _safe_plot(_jitter_correct_stability)

    # --- Stats ---
    correct_df = df[df['correct']]
    wrong_df   = df[~df['correct']]

    def _mean(series):
        v = series.dropna()
        return float(v.mean()) if len(v) else float('nan')

    avg_stab_c = _mean(correct_df['stability'])
    avg_stab_w = _mean(wrong_df['stability'])
    stab_gap   = avg_stab_c - avg_stab_w if not (pd.isna(avg_stab_c) or pd.isna(avg_stab_w)) else float('nan')

    r_conf, p_conf = _pearson_r(df['confidence'].values, df['stability'].values)
    r_deg,  p_deg  = _pearson_r(df['degree'].values.astype(float), df['stability'].values)

    from scipy.stats import ttest_ind
    # Stability t-test
    t_pval = float('nan')
    if len(correct_df) >= 2 and len(wrong_df) >= 2:
        _, t_pval = ttest_ind(correct_df['stability'].values, wrong_df['stability'].values, equal_var=False)

    # Lipschitz stats
    avg_lip_c, avg_lip_w, lip_pval = float('nan'), float('nan'), float('nan')
    if lip_col:
        avg_lip_c  = _mean(correct_df['lipschitz'])
        avg_lip_w  = _mean(wrong_df['lipschitz'])
        lip_c_vals = correct_df['lipschitz'].dropna().values
        lip_w_vals = wrong_df['lipschitz'].dropna().values
        if len(lip_c_vals) >= 2 and len(lip_w_vals) >= 2:
            _, lip_pval = ttest_ind(lip_c_vals, lip_w_vals, equal_var=False)

    # Fidelity stats
    avg_fid_c, avg_fid_w, fid_pval = float('nan'), float('nan'), float('nan')
    if 'fidelity' in df.columns:
        avg_fid_c  = _mean(correct_df['fidelity'])
        avg_fid_w  = _mean(wrong_df['fidelity'])
        fid_c_vals = correct_df['fidelity'].dropna().values
        fid_w_vals = wrong_df['fidelity'].dropna().values
        if len(fid_c_vals) >= 2 and len(fid_w_vals) >= 2:
            _, fid_pval = ttest_ind(fid_c_vals, fid_w_vals, equal_var=False)

    def _fmt(v, decimals=3):
        if pd.isna(v):
            return 'N/A'
        return f'{v:.{decimals}f}'

    def _row(label, value, color='#333'):
        return html.Div(
            style={'display': 'flex', 'justifyContent': 'space-between',
                   'marginBottom': '5px', 'paddingBottom': '4px',
                   'borderBottom': '1px solid #f0f0f0'},
            children=[
                html.Span(label, style={'color': '#666', 'fontSize': '11px'}),
                html.Span(value, style={'fontWeight': 'bold', 'color': color, 'fontSize': '12px'})
            ])

    def _sig(p): return ' *' if not pd.isna(p) and p < 0.05 else ''

    stats_panel = html.Div([
        html.Div("Global Stability Statistics",
                 style={'fontWeight': 'bold', 'fontSize': '13px', 'color': '#0c6e8a',
                        'marginBottom': '8px', 'borderBottom': '2px solid #17a2b8',
                        'paddingBottom': '4px'}),

        # Top-k Jaccard stability
        html.Div("Stability  S(v) = Top-k Jaccard",
                 style={'fontSize': '11px', 'fontWeight': 'bold', 'color': '#555',
                        'marginBottom': '4px', 'marginTop': '4px'}),
        _row('Nodes analysed',     str(len(df))),
        _row('Avg S (correct)',    _fmt(avg_stab_c), CORRECT_COLOR),
        _row('Avg S (wrong)',      _fmt(avg_stab_w), INCORRECT_COLOR),
        _row('Gap  S(C) − S(W)',   _fmt(stab_gap),
             '#007bff' if not pd.isna(stab_gap) and stab_gap > 0 else '#dc3545'),
        _row('t-test p' + _sig(t_pval), _fmt(t_pval, 4)),

        html.Hr(style={'margin': '6px 0', 'borderColor': '#eee'}),

        # Correlations with stability
        html.Div("Correlations with S(v)",
                 style={'fontSize': '11px', 'fontWeight': 'bold', 'color': '#555',
                        'marginBottom': '4px'}),
        _row('r (confidence – S)' + _sig(p_conf), _fmt(r_conf)),
        _row('p',                                   _fmt(p_conf, 4)),
        _row('r (degree – S)'     + _sig(p_deg),  _fmt(r_deg)),
        _row('p',                                   _fmt(p_deg, 4)),

        html.Hr(style={'margin': '6px 0', 'borderColor': '#eee'}),

        # Lipschitz sensitivity
        html.Div("Lipschitz Sensitivity  L(v)",
                 style={'fontSize': '11px', 'fontWeight': 'bold', 'color': '#555',
                        'marginBottom': '4px'}),
        _row('Avg L (correct)', _fmt(avg_lip_c), CORRECT_COLOR),
        _row('Avg L (wrong)',   _fmt(avg_lip_w), INCORRECT_COLOR),
        _row('t-test p' + _sig(lip_pval), _fmt(lip_pval, 4)),

        html.Hr(style={'margin': '6px 0', 'borderColor': '#eee'}),

        # Fidelity
        html.Div("Explanation Fidelity",
                 style={'fontSize': '11px', 'fontWeight': 'bold', 'color': '#555',
                        'marginBottom': '4px'}),
        _row('Avg fidelity (correct)', _fmt(avg_fid_c), CORRECT_COLOR),
        _row('Avg fidelity (wrong)',   _fmt(avg_fid_w), INCORRECT_COLOR),
        _row('t-test p' + _sig(fid_pval), _fmt(fid_pval, 4)),

        html.Div("* p < 0.05",
                 style={'fontSize': '10px', 'color': '#999',
                        'marginTop': '6px', 'fontStyle': 'italic'})
    ])

    if metric_type == 'confidence':
        main_fig = conf_fig
    elif metric_type == 'degree':
        main_fig = deg_fig
    elif metric_type == 'lipschitz':
        main_fig = lip_fig
    else:
        main_fig = cor_fig

    # Maintain consistent height for the main plot
    main_fig.update_layout(height=360)
    
    return main_fig, stats_panel


def build_drilldown_scatter(click_data, epsilon, plot_type, results):
    plot_type = (plot_type or 'scatter').lower()

    # ── per-mode default ε (scatter = exact point, density = wider bin) ──────
    _default_eps = {'hexbin': 0.08, 'kde': 0.08, 'scatter': 0.05}
    eps = float(epsilon) if epsilon is not None else _default_eps.get(plot_type, 0.08)

    def _empty(text, color='#aaa'):
        fig = go.Figure(layout=dict(
            height=400, margin=dict(l=45, r=15, t=40, b=40),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        ))
        fig.add_annotation(xref='paper', yref='paper', x=0.5, y=0.5,
                           text=text, showarrow=False,
                           font=dict(size=13, color=color),
                           xanchor='center', yanchor='middle')
        return fig

    # ── Mode-aware placeholder messages ──────────────────────────────────────
    if not click_data:
        _no_click_msg = {
            'hexbin':  'Click a hexbin cell above to zoom into its nodes',
            'kde':     'Click a KDE contour region above to zoom into its nodes',
            'scatter': 'Click a scatter point above to see its nearest neighbors',
        }.get(plot_type, 'Click on a plot region above to see node-level details')
        return _empty(_no_click_msg), "Click on a region to explore node-level details"

    if not results:
        return _empty('Run Stability Analysis first', '#999'), "No stability data — run analysis first"

    # ── Rebuild DataFrame from cached results (no recomputation) ─────────────
    df = pd.DataFrame(results)

    # Normalize link-oriented results to node-oriented schema if necessary
    if 'node_idx' not in df.columns and 'source' in df.columns and 'target' in df.columns:
        try:
            df['node_idx'] = df['source'].astype(str) + '-' + df['target'].astype(str)
            if 'degree' not in df.columns and 'degree_sum' in df.columns:
                df['degree'] = df['degree_sum']
            # ensure 'correct' exists
            if 'correct' not in df.columns:
                df['correct'] = False
            # ensure confidence exists
            if 'confidence' not in df.columns and 'confidence' in df.columns:
                df['confidence'] = df['confidence']
        except Exception:
            pass

    # After normalization, ensure required plotting columns exist; fill with defaults if missing
    required = {'confidence', 'stability', 'degree', 'correct', 'node_idx'}
    for col in required:
        if col not in df.columns:
            if col == 'correct':
                df[col] = False
            elif col == 'node_idx':
                df[col] = df.index.astype(str)
            else:
                df[col] = np.nan

    source  = click_data.get('source', 'confidence')
    x_col   = 'confidence' if source == 'confidence' else 'degree'
    x_click = click_data['x']
    y_click = click_data['y']
    x_label = 'Confidence' if x_col == 'confidence' else 'Node Degree'

    # ── Vectorised NumPy filter — mode-aware ─────────────────────────────────
    df = df.dropna(subset=[x_col, 'stability']).copy()
    df[x_col]       = df[x_col].astype(float)
    df['stability'] = df['stability'].astype(float)
    x_arr = df[x_col].values
    y_arr = df['stability'].values

    if plot_type == 'kde':
        # Density-contour filter: fit KDE on all nodes, keep those whose density
        # is within ±30 % of the density at the clicked point.
        clicked_density = click_data.get('clicked_density')
        kde_mask_ok = False
        if clicked_density is not None and clicked_density > 0:
            try:
                from scipy.stats import gaussian_kde as _gkde
                if float(np.std(x_arr)) > 1e-9 and float(np.std(y_arr)) > 1e-9:
                    kde_fn    = _gkde(np.vstack([x_arr, y_arr]), bw_method='silverman')
                    node_dens = kde_fn(np.vstack([x_arr, y_arr]))
                    lo, hi    = clicked_density * 0.7, clicked_density * 1.3
                    mask      = (node_dens >= lo) & (node_dens <= hi)
                    kde_mask_ok = True
                    kde_band  = (clicked_density * 0.7, clicked_density * 1.3)
            except Exception as _ke:
                print(f"[DRILLDOWN] KDE density filter failed: {_ke}")
        if not kde_mask_ok:
            # Fallback: rectangular box when KDE fitting fails
            mask = ((x_arr >= x_click - eps) & (x_arr <= x_click + eps) &
                    (y_arr >= y_click - eps) & (y_arr <= y_click + eps))
            kde_band = None

    elif plot_type == 'hexbin':
        # Circular distance from snapped bin center
        dist = np.sqrt((x_arr - x_click) ** 2 + (y_arr - y_click) ** 2)
        mask = dist <= eps

    else:
        # Scatter: rectangular epsilon box
        mask = ((x_arr >= x_click - eps) & (x_arr <= x_click + eps) &
                (y_arr >= y_click - eps) & (y_arr <= y_click + eps))

    subset   = df[mask].copy()
    n_total  = len(df)
    n_subset = len(subset)

    if plot_type == 'kde':
        if kde_mask_ok and kde_band:
            header = (
                f"KDE Region  ·  density ∈ [{kde_band[0]:.5f}, {kde_band[1]:.5f}]"
                f"  ·  {n_subset} / {n_total} nodes"
            )
        else:
            header = (
                f"KDE Region (fallback ε-box)  ·  {n_subset} / {n_total} nodes"
            )
    elif plot_type == 'hexbin':
        header = (
            f"Zoomed Region  ·  Bin center ({x_click:.3f}, {y_click:.3f})"
            f"  ·  radius ε = {eps:.2f}"
            f"  ·  {n_subset} / {n_total} nodes"
        )
    else:
        header = (
            f"Zoomed Region  ·  {x_label} ∈ [{x_click - eps:.3f}, {x_click + eps:.3f}]"
            f"  ·  Stability ∈ [{y_click - eps:.3f}, {y_click + eps:.3f}]"
            f"  ·  {n_subset} / {n_total} nodes"
        )

    if subset.empty:
        msg = ('No nodes in this density band  —  try clicking a denser region'
               if plot_type == 'kde' else
               f'No points in selected region (ε = {eps:.2f})  —  try a larger ε')
        return _empty(msg, '#999'), header

    # ── Build drill-down scatter figure ──────────────────────────────────────
    color_map = {True: CORRECT_COLOR, False: INCORRECT_COLOR}
    label_map = {True: 'Correct',  False: 'Wrong'}
    has_lip   = 'lipschitz' in subset.columns

    fig = go.Figure()
    for correct_val in [True, False]:
        grp = subset[subset['correct'] == correct_val]
        if grp.empty:
            continue

        hover_parts = [
            '<b>Node %{customdata[0]}</b>',
            f'{x_label}: %{{x:.4f}}',
            'Stability: %{y:.4f}',
            'Degree: %{customdata[1]}',
            'Confidence: %{customdata[2]:.4f}',
        ]
        if has_lip:
            hover_parts.append('Lipschitz: %{customdata[3]:.4f}')
        hover_parts.append('<extra></extra>')

        lip_vals = (grp['lipschitz'].tolist() if has_lip
                    else [float('nan')] * len(grp))
        custom = list(zip(
            grp['node_idx'].tolist(),
            grp['degree'].tolist(),
            grp['confidence'].tolist(),
            lip_vals,
        ))

        fig.add_trace(go.Scatter(
            x=grp[x_col].tolist(),
            y=grp['stability'].tolist(),
            mode='markers',
            name=label_map[correct_val],
            marker=dict(color=color_map[correct_val], size=7,
                        opacity=0.75, line=dict(width=0.6, color='#fff')),
            hovertemplate='<br>'.join(hover_parts),
            customdata=custom,
        ))

    # Selection indicator — mode-aware
    if plot_type == 'kde':
        # KDE: crosshair only — no box (density band has no simple geometric boundary)
        fig.add_shape(type='line', xref='x', yref='paper',
                      x0=x_click, x1=x_click, y0=0, y1=1,
                      line=dict(color='#8e44ad', width=1.5, dash='dot'))
        fig.add_shape(type='line', xref='paper', yref='y',
                      x0=0, x1=1, y0=y_click, y1=y_click,
                      line=dict(color='#8e44ad', width=1.5, dash='dot'))
    elif plot_type == 'hexbin':
        # Hexbin: circle matching the circular filter radius
        fig.add_shape(type='circle', xref='x', yref='y',
                      x0=x_click - eps, x1=x_click + eps,
                      y0=y_click - eps, y1=y_click + eps,
                      line=dict(color='#1a73e8', width=1.5, dash='dash'),
                      fillcolor='rgba(26,115,232,0.04)')
        fig.add_shape(type='line', xref='x', yref='paper',
                      x0=x_click, x1=x_click, y0=0, y1=1,
                      line=dict(color='rgba(0,0,0,0.15)', width=1, dash='dot'))
        fig.add_shape(type='line', xref='paper', yref='y',
                      x0=0, x1=1, y0=y_click, y1=y_click,
                      line=dict(color='rgba(0,0,0,0.15)', width=1, dash='dot'))
    else:
        # Scatter: rectangular window + crosshair
        fig.add_shape(type='rect', xref='x', yref='y',
                      x0=x_click - eps, x1=x_click + eps,
                      y0=y_click - eps, y1=y_click + eps,
                      line=dict(color='#1a73e8', width=1.5, dash='dash'),
                      fillcolor='rgba(26,115,232,0.04)')
        fig.add_shape(type='line', xref='x', yref='paper',
                      x0=x_click, x1=x_click, y0=0, y1=1,
                      line=dict(color='rgba(0,0,0,0.15)', width=1, dash='dot'))
        fig.add_shape(type='line', xref='paper', yref='y',
                      x0=0, x1=1, y0=y_click, y1=y_click,
                      line=dict(color='rgba(0,0,0,0.15)', width=1, dash='dot'))

    # In-figure "Showing N nodes" annotation (top-left corner)
    n_correct = int((subset['correct'] == True).sum())
    n_wrong   = int((subset['correct'] == False).sum())
    if plot_type == 'kde':
        cd = click_data.get('clicked_density')
        density_str = f'  ·  density ≈ {cd:.5f}' if cd else ''
        annot_text = (f'KDE band{density_str}<br>'
                      f'Showing <b>{n_subset}</b> nodes  ({n_correct} correct, {n_wrong} wrong)')
    else:
        annot_text = f'Showing <b>{n_subset}</b> nodes  ({n_correct} correct, {n_wrong} wrong)'
    fig.add_annotation(
        xref='paper', yref='paper', x=0.01, y=0.99,
        text=annot_text,
        showarrow=False, xanchor='left', yanchor='top',
        font=dict(size=11, color='#444'),
        bgcolor='rgba(255,255,255,0.80)',
        bordercolor='#ccc', borderwidth=1, borderpad=4,
    )

    if plot_type == 'kde':
        title_text = f'KDE Density Band — {n_subset} nodes'
    elif plot_type == 'hexbin':
        title_text = f'Zoomed Region — {n_subset} nodes  (dist ≤ {eps:.2f})'
    else:
        title_text = f'Zoomed Region — {n_subset} nodes  (ε ≤ {eps:.2f})'

    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=13, color='#333'),
            x=0, xanchor='left',
        ),
        height=400,
        margin=dict(l=45, r=15, t=45, b=40),
        xaxis_title=x_label,
        yaxis_title='Stability (Top-k Jaccard)',
        legend=dict(x=0.88, y=0.99, bgcolor='rgba(255,255,255,0.85)',
                    bordercolor='#ccc', borderwidth=1, font=dict(size=11)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='closest',
    )
    fig.update_xaxes(showgrid=True, gridcolor='#eee', zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='#eee', zeroline=False)
    return fig, header


def build_region_highlight(click_data, epsilon):
    from dash import Patch

    eps = float(epsilon) if epsilon is not None else 0.08

    # Build a Patch that wipes shapes and annotations (reset case)
    def _clear():
        p = Patch()
        p['layout']['shapes'] = []
        p['layout']['annotations'] = []
        return p

    if not click_data:
        return _clear()

    x_click  = click_data['x']
    y_click  = click_data['y']
    source   = click_data.get('source', 'confidence')
    click_pt = click_data.get('plot_type', 'scatter')

    # Build shapes + annotation depending on mode
    if click_pt == 'kde':
        # KDE: crosshair at clicked point — no shape (density band has no box boundary)
        shapes = [
            dict(type='line', xref='x', yref='paper',
                 x0=x_click, x1=x_click, y0=0, y1=1,
                 line=dict(color='#8e44ad', width=1.5, dash='dot')),
            dict(type='line', xref='paper', yref='y',
                 x0=0, x1=1, y0=y_click, y1=y_click,
                 line=dict(color='#8e44ad', width=1.5, dash='dot')),
        ]
        label = dict(
            xref='x', yref='y',
            x=x_click, y=y_click,
            text='Clicked density',
            showarrow=True, arrowhead=2, arrowsize=0.8,
            arrowcolor='#8e44ad', ax=30, ay=-25,
            font=dict(size=10, color='#8e44ad'),
            bgcolor='rgba(255,255,255,0.80)',
            borderpad=2,
        )
    elif click_pt == 'hexbin':
        # Hexbin: circle matching circular distance filter
        shapes = [
            dict(type='circle', xref='x', yref='y',
                 x0=x_click - eps, x1=x_click + eps,
                 y0=y_click - eps, y1=y_click + eps,
                 line=dict(color='#ff6b35', width=2, dash='dash'),
                 fillcolor='rgba(255,107,53,0.08)'),
        ]
        label = dict(
            xref='x', yref='y',
            x=x_click, y=y_click + eps,
            text='Selected Region',
            showarrow=False,
            font=dict(size=10, color='#ff6b35'),
            bgcolor='rgba(255,255,255,0.75)',
            borderpad=2, yanchor='bottom', xanchor='center',
        )
    else:
        # Scatter: rectangle
        shapes = [
            dict(type='rect', xref='x', yref='y',
                 x0=x_click - eps, x1=x_click + eps,
                 y0=y_click - eps, y1=y_click + eps,
                 line=dict(color='#ff6b35', width=2, dash='dash'),
                 fillcolor='rgba(255,107,53,0.08)'),
        ]
        label = dict(
            xref='x', yref='y',
            x=x_click, y=y_click + eps,
            text='Selected Region',
            showarrow=False,
            font=dict(size=10, color='#ff6b35'),
            bgcolor='rgba(255,255,255,0.75)',
            borderpad=2, yanchor='bottom', xanchor='center',
        )

    p_main = Patch()
    p_main['layout']['shapes'] = shapes
    p_main['layout']['annotations'] = [label]
    return p_main

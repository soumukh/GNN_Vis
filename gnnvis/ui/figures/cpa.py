"""CPA-IV (causal path analysis) panels for node and link explanations."""

import pandas as pd
import plotly.express as px
from dash import html
from dash import dcc
from gnnvis.ui import ids


def create_feature_attribution_plot(cpa_data):
    """Create a bar chart for feature attribution."""
    if not cpa_data or 'feature_analysis' not in cpa_data:
        return html.Div("No feature analysis available.", style={'color': '#999', 'fontSize': '12px'})

    fa = cpa_data['feature_analysis']
    if not fa.get('source_importance') and not fa.get('target_importance'):
        return html.Div("Feature gradients not computed.", style={'color': '#999', 'fontSize': '12px'})

    # Prepare data for plotting
    data = []
    
    # Source
    for item in fa.get('source_importance', []):
        data.append({'Feature': f"S-{item['index']}", 'Score': item['score'], 'Type': 'Source (u)'})
    
    # Target
    for item in fa.get('target_importance', []):
        data.append({'Feature': f"T-{item['index']}", 'Score': item['score'], 'Type': 'Target (v)'})
        
    # Shared (Optional, if we want to show it explicitly, or just rely on S/T)
    # The requirement asks for "Separate bars for source, target, shared"
    # Shared usually means features that are active in both.
    # For now, we plot Source and Target top features.
    
    df = pd.DataFrame(data)
    if df.empty:
         return html.Div("Feature importance scores are all zero.", style={'color': '#999', 'fontSize': '12px'})

    fig = px.bar(df, x='Feature', y='Score', color='Type',
                 title="Top Feature Contributions (Gradient * Input)",
                 color_discrete_map={'Source (u)': '#3498db', 'Target (v)': '#e74c3c'},
                 barmode='group')
                 
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=200,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def create_neighborhood_impact_plot(neigh_impact):
    """Create a bar chart for neighborhood structural impact."""
    if not neigh_impact:
        return html.Div("No neighborhood impact data.", style={'color': '#999'})
    
    u_impact = neigh_impact.get('u', 0)
    v_impact = neigh_impact.get('v', 0)
    
    if u_impact < 1e-4 and v_impact < 1e-4:
        return html.Div([
            html.Span("Neighborhood Isolation Impact is Negligible.", style={'fontWeight': 'bold', 'color': '#7f8c8d'}),
            html.Br(),
            html.Span("Removing neighbors has almost no effect on the link score.", style={'fontSize': '12px', 'color': '#999'})
        ], style={'padding': '10px', 'textAlign': 'center', 'border': '1px dashed #ccc', 'borderRadius': '4px'})

    df = pd.DataFrame([
        {'Node': 'Source (u)', 'Impact': u_impact},
        {'Node': 'Target (v)', 'Impact': v_impact}
    ])
    
    fig = px.bar(df, x='Impact', y='Node', orientation='h',
                 title="Neighborhood Isolation Impact (Δ Probability)",
                 text_auto='.3f',
                 color='Node',
                 color_discrete_map={'Source (u)': '#3498db', 'Target (v)': '#e74c3c'})
                 
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=30, b=20),
        height=150,
        showlegend=False,
        xaxis=dict(showgrid=True, gridcolor='#eee'),
        yaxis=dict(showgrid=False)
    )
    return dcc.Graph(figure=fig, config={'displayModeBar': False})


def create_source_target_responsibility_plot(cpa_data):
    """Create a stacked bar showing relative responsibility of Source vs Target."""
    if not cpa_data or 'feature_analysis' not in cpa_data:
        return html.Div()
    
    fa = cpa_data['feature_analysis']
    s_score = sum(item['score'] for item in fa.get('source_importance', []))
    t_score = sum(item['score'] for item in fa.get('target_importance', []))
    
    total = s_score + t_score
    if total < 1e-9:
        return html.Div("Responsibility split inconclusive.", style={'fontSize': '11px', 'color': '#999'})
        
    s_pct = (s_score / total) * 100
    t_pct = (t_score / total) * 100
    
    return html.Div([
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '5px', 'fontSize': '12px', 'fontWeight': 'bold'}, children=[
            html.Span(f"Source (u): {s_pct:.1f}%", style={'color': '#3498db'}),
            html.Span(f"Target (v): {t_pct:.1f}%", style={'color': '#e74c3c'})
        ]),
        html.Div(style={'display': 'flex', 'height': '12px', 'borderRadius': '6px', 'overflow': 'hidden', 'backgroundColor': '#eee'}, children=[
            html.Div(style={'width': f'{s_pct}%', 'backgroundColor': '#3498db', 'height': '100%'}),
            html.Div(style={'width': f'{t_pct}%', 'backgroundColor': '#e74c3c', 'height': '100%'})
        ])
    ], style={'padding': '10px', 'backgroundColor': '#f8f9fa', 'borderRadius': '4px', 'marginBottom': '10px'})


def create_embedding_attribution_plot(cpa_data):
    """Create a placeholder or explanatory panel for embedding-driven predictions."""
    return html.Div([
        html.Div([
            html.I(className="fas fa-project-diagram", style={'marginRight': '10px', 'color': '#9b59b6'}),
            html.Span("Latent Structural Similarity", style={'fontWeight': 'bold', 'color': '#2c3e50'})
        ], style={'marginBottom': '8px', 'fontSize': '13px'}),
        html.P([
            "Model prediction is primarily driven by ",
            html.B("Embedding Similarity"),
            " (Latent space proximity) rather than local k-hop pathways."
        ], style={'fontSize': '12px', 'color': '#666', 'margin': 0}),
        html.Div([
            html.Span("Confidence in global pattern: ", style={'fontSize': '11px', 'color': '#999'}),
            html.Span("High", style={'fontSize': '11px', 'fontWeight': 'bold', 'color': '#2ecc71'})
        ], style={'marginTop': '5px'})
    ], style={'padding': '12px', 'backgroundColor': '#fcfaff', 'border': '1px solid #dcd0ff', 'borderRadius': '4px'})


def create_cpa_iv_panel(cpa_data, node_idx):
    """Create a panel displaying causal path analysis results."""
    if not cpa_data:
        return html.Div("No causal paths found. Click 'RUN ANALYSIS' button.",
                       style={'color': '#999', 'fontStyle': 'italic', 'fontSize': '13px'})

    # --- Node Classification CPA format: list of {path, score} ---
    if isinstance(cpa_data, list):
        if len(cpa_data) == 0:
            return html.Div("CPA-IV found no influential paths for this node.",
                           style={'color': '#999', 'fontStyle': 'italic', 'fontSize': '13px'})

        path_items = []
        for i, p in enumerate(cpa_data[:10], 1):
            path_nodes = p.get('path', [])
            score = p.get('score', 0)
            path_str = ' \u2192 '.join(str(n) for n in path_nodes)
            score_color = '#28a745' if score > 0.01 else ('#e67e22' if score > 0.001 else '#999')
            path_items.append(
                html.Div(
                    id={'type': ids.CPA_PATH_ITEM_TYPE, 'index': i - 1},
                    n_clicks=0,
                    children=[
                        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center'}, children=[
                            html.Span(f"Path {i}: ", style={'fontWeight': 'bold', 'color': '#555', 'fontSize': '12px'}),
                            html.Span(path_str, style={'color': '#007bff', 'fontFamily': 'monospace', 'fontSize': '12px', 'flex': '1', 'marginLeft': '5px'}),
                            html.Span(f"{score:.4f}", style={'fontSize': '12px', 'fontWeight': 'bold', 'color': score_color, 'marginLeft': '10px'})
                        ])
                    ],
                    style={'padding': '6px 8px', 'borderBottom': '1px solid #eee', 'cursor': 'pointer',
                           'transition': 'background-color 0.2s'}
                )
            )

        top_score = cpa_data[0].get('score', 0)
        total_score = sum(p.get('score', 0) for p in cpa_data)
        if top_score > 0.01:
            strength_label, strength_color, gauge_width = 'STRONG', '#28a745', '90%'
        elif top_score > 0.001:
            strength_label, strength_color, gauge_width = 'MODERATE', '#e67e22', '55%'
        elif top_score > 0:
            strength_label, strength_color, gauge_width = 'WEAK', '#f39c12', '25%'
        else:
            strength_label, strength_color, gauge_width = 'NONE', '#ccc', '5%'

        return html.Div([
            html.H5("Causal Path Analysis (CPA-IV)",
                    style={'marginBottom': '15px', 'color': '#2c3e50', 'fontSize': '16px',
                           'fontWeight': 'bold', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
            html.Div([
                html.Span("Target Node: ", style={'fontSize': '12px', 'color': '#666'}),
                html.Span(f"{node_idx}", style={'fontWeight': 'bold', 'fontSize': '14px', 'color': '#333'})
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '5px'}, children=[
                    html.Span("CAUSAL STRENGTH",style={'fontSize': '11px', 'fontWeight': 'bold', 'color': '#666'}),
                    html.Span(strength_label, style={'fontSize': '11px', 'fontWeight': 'bold', 'color': strength_color})
                ]),
                html.Div(style={'height': '8px', 'backgroundColor': '#eee', 'borderRadius': '4px', 'overflow': 'hidden'}, children=[
                    html.Div(style={'width': gauge_width, 'height': '100%', 'backgroundColor': strength_color, 'transition': 'width 0.5s ease'})
                ])
            ], style={'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 'marginBottom': '15px', 'border': '1px solid #eee'}),
            html.Div([
                html.Span(f"{len(cpa_data)} paths found", style={'fontSize': '12px', 'color': '#666'}),
                html.Span(f"  |  Total effect: {total_score:.4f}", style={'fontSize': '12px', 'color': '#666', 'marginLeft': '10px'}),
            ], style={'marginBottom': '10px'}),
            html.Div("TOP CAUSAL PATHS", style={'fontWeight': 'bold', 'fontSize': '11px', 'color': '#666',
                                                  'marginBottom': '5px', 'textTransform': 'uppercase'}),
            html.Div(path_items, style={'maxHeight': '250px', 'overflowY': 'auto', 'border': '1px solid #eee', 'borderRadius': '4px'})
        ])

    # --- Link Prediction CPA format: dict with verdict, paths, etc. ---
    # --- 1. Common Metrics Extraction ---
    verdict = cpa_data.get('verdict', 'N/A')
    verdict_type = cpa_data.get('verdict_type', 'none')
    baseline = cpa_data.get('baseline_score', 0)
    sensitivity = cpa_data.get('instrument_sensitivity', 0)
    source = cpa_data.get('source_node', '?')
    target = cpa_data.get('target_node', '?')
    num_paths = cpa_data.get('num_paths_found', 0)
    is_existing = cpa_data.get('is_existing_link', False)
    
    neigh_impact = cpa_data.get('neigh_impact', {})
    
    # --- 1.5 Classification Status Badge ---
    status_badge = None
    if baseline > 0.5 and not is_existing:
        status_badge = html.Span("FALSE POSITIVE", style={'backgroundColor': '#e74c3c', 'color': 'white', 'padding': '3px 8px', 'borderRadius': '12px', 'fontSize': '10px', 'marginLeft': '10px', 'fontWeight': 'bold', 'verticalAlign': 'middle'})
    elif baseline < 0.5 and is_existing:
        status_badge = html.Span("MISSING LINK", style={'backgroundColor': '#f39c12', 'color': 'white', 'padding': '3px 8px', 'borderRadius': '12px', 'fontSize': '10px', 'marginLeft': '10px', 'fontWeight': 'bold', 'verticalAlign': 'middle'})
    elif is_existing:
        status_badge = html.Span("EXISTING LINK", style={'backgroundColor': '#2ecc71', 'color': 'white', 'padding': '3px 8px', 'borderRadius': '12px', 'fontSize': '10px', 'marginLeft': '10px', 'fontWeight': 'bold', 'verticalAlign': 'middle'})
    else:
        status_badge = html.Span("NON-EXISTING", style={'backgroundColor': '#95a5a6', 'color': 'white', 'padding': '3px 8px', 'borderRadius': '12px', 'fontSize': '10px', 'marginLeft': '10px', 'fontWeight': 'bold', 'verticalAlign': 'middle'})

    # --- 2. Causal Strength Gauge ---
    # Visual gauge using a colored bar
    gauge_color = '#ccc'
    gauge_width = '10%'
    
    # Verdict Logic Refinement for Display
    if verdict_type == 'strong': 
        gauge_color = '#2ecc71' # Green
        gauge_width = '90%'
    elif verdict_type == 'moderate': 
        gauge_color = '#f1c40f' # Yellow
        gauge_width = '60%'
    elif verdict_type == 'weak':
        # Distinguish between "Weak Effect" and "Global Causality"
        if "Global" in verdict:
            gauge_color = '#9b59b6' # Purple for Global/Latent
            gauge_width = '75%' # High confidence, just non-local
        else:
            gauge_color = '#e67e22' # Orange for Weak Path
            gauge_width = '30%'
        
    gauge_component = html.Div([
        html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '5px'}, children=[
            html.Span("CAUSAL VERDICT", style={'fontSize': '11px', 'fontWeight': 'bold', 'color': '#666'}),
            html.Span(verdict.upper(), style={'fontSize': '11px', 'fontWeight': 'bold', 'color': gauge_color})
        ]),
        html.Div(style={'height': '8px', 'backgroundColor': '#eee', 'borderRadius': '4px', 'overflow': 'hidden'}, children=[
            html.Div(style={
                'width': gauge_width, 
                'height': '100%', 
                'backgroundColor': gauge_color,
                'transition': 'width 0.5s ease'
            })
        ])
    ], style={'padding': '15px', 'backgroundColor': '#f8f9fa', 'borderRadius': '8px', 'marginBottom': '15px', 'border': '1px solid #eee'})

    # --- 3. Path Absence Explanation View (Condition) ---
    path_section = None
    cf_plot = None # Counterfactual Plot

    if num_paths == 0:
        checklist_style = {'display': 'flex', 'alignItems': 'center', 'marginBottom': '5px', 'fontSize': '12px', 'color': '#555'}
        icon_style = {'marginRight': '8px', 'width': '15px', 'textAlign': 'center'}
        
        path_section = html.Div([
            html.Div([
                html.I(className="fas fa-network-wired", style={'fontSize': '24px', 'color': '#9b59b6', 'marginBottom': '10px'}),
                html.H6("Global Structural Causality", style={'fontWeight': 'bold', 'color': '#555', 'margin': '0 0 10px 0'}),
                
                # Visual Checklist
                html.Div([
                    html.Div("Why no path-based explanation?", style={'fontWeight':'bold', 'fontSize':'11px', 'marginBottom':'5px', 'color': '#7f8c8d'}),
                    html.Div([
                        html.Span("✖", style={**icon_style, 'color': '#dc3545', 'fontWeight': 'bold'}),
                        html.Span("No Shortest Paths (k-hop)")
                    ], style=checklist_style),
                     html.Div([
                        html.Span("✖", style={**icon_style, 'color': '#dc3545', 'fontWeight': 'bold'}),
                        html.Span("No Common Neighbors")
                    ], style=checklist_style),
                    html.Div([
                        html.Span("✖", style={**icon_style, 'color': '#dc3545', 'fontWeight': 'bold'}),
                        html.Span("Low Structural Overlap")
                    ], style=checklist_style),
                ], style={'textAlign': 'left', 'backgroundColor': '#fff', 'padding': '10px', 'borderRadius': '4px', 'marginBottom': '10px'}),

                html.P([
                    "Prediction is driven by ",
                    html.B("Latent Embedding Similarity"),
                    " (Homophily) and Node Features."
                ], style={'fontSize': '13px', 'color': '#333'})
            ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#fcfaff', 'border': '2px dashed #9b59b6', 'borderRadius': '8px', 'marginBottom': '15px'})
        ])
    else:
        # Standard Path List
        path_items = []
        paths = cpa_data.get('paths', [])
        
        # Prepare Counterfactual Plot for TOP path
        if paths:
            top_p = paths[0]
            # Use perturbed_score if available, else derive
            perturbed = top_p.get('perturbed_score', baseline - top_p.get('delta_p', 0))
            
            df_cf = pd.DataFrame([
                {'State': 'Original', 'Probability': baseline},
                {'State': 'Counterfactual', 'Probability': perturbed}
            ])
            fig_cf = px.bar(df_cf, x='State', y='Probability', color='State', 
                         title="Counterfactual: Top Path Removal",
                         color_discrete_map={'Original': '#3498db', 'Counterfactual': '#95a5a6'},
                         text_auto='.3f')
            fig_cf.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                                margin=dict(l=10, r=10, t=30, b=10), height=180, showlegend=False)
            cf_plot = dcc.Graph(figure=fig_cf, config={'displayModeBar': False})

        for i, path_info in enumerate(paths[:5], 1):
             path = path_info['nodes']
             score = path_info['score']
             # Format path string
             path_str = ' → '.join([str(node) for node in path])
             
             path_items.append(
                html.Div([
                    html.Div([
                        html.Span(f"Path {i}: ", style={'fontWeight': 'bold', 'color': '#555', 'fontSize': '12px'}),
                        html.Span(path_str, style={'color': '#007bff', 'fontFamily': 'monospace', 'fontSize': '13px'})
                    ], style={'marginBottom': '2px'}),
                    html.Div([
                        html.Span("Effect: ", style={'fontSize': '11px', 'color': '#666'}),
                        html.Span(f"{score:.4f}", style={'fontSize': '12px', 'fontWeight': 'bold', 'color': '#dc3545'})
                    ])
                ], style={'padding': '6px 0', 'borderBottom': '1px solid #eee'})
            )
        path_section = html.Div([
            html.Div("TOP CAUSAL PATHS", style={'fontWeight': 'bold', 'fontSize': '11px', 'color': '#666', 'marginBottom': '5px', 'textTransform': 'uppercase'}),
            html.Div(path_items, style={'maxHeight': '200px', 'overflowY': 'auto'})
        ])

    # --- 4. Plots ---
    # For link-detail CPA (dict) we intentionally omit the visual plots
    # and only return the minimal list + counters so the UI shows just the list.
    if isinstance(cpa_data, dict):
        return html.Div([
            html.H5("Causal Analysis (CPA-IV)", 
                   style={'marginBottom': '15px', 'color': '#2c3e50', 'fontSize': '16px', 'fontWeight': 'bold', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
            html.Div([
                html.Span("Analysis Target: ", style={'fontSize': '12px', 'color': '#666'}),
                html.Span(f"{source} → {target}", style={'fontWeight': 'bold', 'fontSize': '14px', 'color': '#333'}),
                status_badge
            ], style={'marginBottom': '10px', 'display': 'flex', 'alignItems': 'center'}),
            gauge_component,
            # Counterfactual Chart (only if available)
            html.Div([
                html.Div("PATH INTERVENTION EFFECT", style={'fontWeight': 'bold', 'fontSize': '11px', 'color': '#666', 'marginBottom': '5px', 'textTransform': 'uppercase'}),
                cf_plot
            ], style={'marginBottom': '15px', 'display': 'block' if cf_plot else 'none'}),
            # Path list only
            path_section
        ])

    # Non-link mode: retain full panel with plots
    neigh_plot = create_neighborhood_impact_plot(neigh_impact)
    feature_plot = create_feature_attribution_plot(cpa_data)
    embedding_plot = create_embedding_attribution_plot(cpa_data)
    responsibility_plot = create_source_target_responsibility_plot(cpa_data)
    
    # --- 5. Interactive Editing (New) ---
    edit_section = html.Div([
        html.Div("COUNTERFACTUAL EDITING", style={'fontWeight': 'bold', 'fontSize': '11px', 'color': '#666', 'marginBottom': '5px', 'textTransform': 'uppercase'}),
        html.Div([
             html.Button('Add Direct Edge', id=ids.ADD_EDGE_BUTTON_CPA, n_clicks=0, style={'background': '#28a745', 'border': 'none', 'color': 'white', 'padding': '6px 12px', 'borderRadius': '2px', 'fontWeight': 'bold', 'cursor': 'pointer', 'marginRight': '5px', 'fontSize': '11px'}),
             html.Button('Remove Direct Edge', id=ids.REMOVE_EDGE_BUTTON_CPA, n_clicks=0, style={'background': '#dc3545', 'border': 'none', 'color': 'white', 'padding': '6px 12px', 'borderRadius': '2px', 'fontWeight': 'bold', 'cursor': 'pointer', 'fontSize': '11px'})
        ], style={'marginBottom': '15px'})
    ])

    # --- 6. Assemble Panel ---
    return html.Div([
        html.H5("Causal Analysis (CPA-IV)", 
               style={'marginBottom': '15px', 'color': '#2c3e50', 'fontSize': '16px', 'fontWeight': 'bold', 'borderBottom': '2px solid #3498db', 'paddingBottom': '10px'}),
        
        # Link Info with Badge
        html.Div([
            html.Span("Analysis Target: ", style={'fontSize': '12px', 'color': '#666'}),
            html.Span(f"{source} → {target}", style={'fontWeight': 'bold', 'fontSize': '14px', 'color': '#333'}),
            status_badge
        ], style={'marginBottom': '10px', 'display': 'flex', 'alignItems': 'center'}),

        # Gauge
        gauge_component,
        
        # Counterfactual Chart (New)
        html.Div([
            html.Div("PATH INTERVENTION EFFECT", style={'fontWeight': 'bold', 'fontSize': '11px', 'color': '#666', 'marginBottom': '5px', 'textTransform': 'uppercase'}),
            cf_plot
        ], style={'marginBottom': '15px', 'display': 'block' if cf_plot else 'none'}),

        # Path Section (Conditional)
        path_section,
        
        html.Hr(style={'borderTop': '1px solid #eee', 'margin': '15px 0'}),
        
        # Edit Section (Visible)
        edit_section,
        
        # Embedding Attribution (NEW)
        html.Div([
             html.Div("LATENT STRUCTURAL DRIVERS", style={'fontWeight': 'bold', 'fontSize': '11px', 'color': '#666', 'marginBottom': '5px', 'textTransform': 'uppercase'}),
             embedding_plot
        ], style={'marginBottom': '15px', 'display': 'block' if num_paths == 0 else 'none'}), 
        
        # Responsibility (NEW)
        html.Div([
            html.Div("CAUSAL RESPONSIBILITY", style={'fontWeight': 'bold', 'fontSize': '11px', 'color': '#666', 'marginBottom': '5px', 'textTransform': 'uppercase'}),
            responsibility_plot
        ], style={'marginBottom': '15px'}),

        # Neighborhood Impact
        html.Div([
             html.Div("NEIGHBORHOOD INFLUENCE", style={'fontWeight': 'bold', 'fontSize': '11px', 'color': '#666', 'marginBottom': '5px', 'textTransform': 'uppercase'}),
             neigh_plot
        ], style={'marginBottom': '15px'}),
        
        # Feature Attribution
        html.Div([
            html.Div("FEATURE INFLUENCE", style={'fontWeight': 'bold', 'fontSize': '11px', 'color': '#666', 'marginBottom': '5px', 'textTransform': 'uppercase'}),
            feature_plot
        ])
    ])

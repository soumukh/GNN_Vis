"""Confusion matrices (node and link views) and their paged cell-detail tables."""

import math
from dash import html
from dash import dash_table
import plotly.graph_objects as go

from gnnvis.ui.theme import class_label
from gnnvis.ui import ids


def _make_cm_detail_table(rows, columns, badge_color):
    return dash_table.DataTable(
        id=ids.CM_DETAIL_TABLE,
        data=rows,
        columns=columns,
        style_table={
            'overflowX': 'auto',
            'overflowY': 'auto',
            'maxHeight': '100%',
            'border': '1px solid #eee',
        },
        style_header={
            'backgroundColor': badge_color, 'color': 'white',
            'fontWeight': 'bold', 'fontSize': '12px',
            'padding': '6px 10px', 'border': 'none',
        },
        style_cell={
            'fontSize': '12px', 'padding': '5px 10px',
            'textAlign': 'center', 'border': '1px solid #f0f0f0',
            'fontFamily': 'monospace',
            'whiteSpace': 'normal',
        },
        style_data_conditional=[
            {'if': {'row_index': 'odd'}, 'backgroundColor': '#fafafa'},
        ],
        sort_action='native',
        filter_action='none',
        page_action='none',
    )


def _render_cm_detail_page(rows, columns, badge_color, footer_text, page):
    PAGE_SIZE = 10
    page_count = max(1, math.ceil(len(rows) / PAGE_SIZE))
    page = min(max(page, 0), page_count - 1)
    start = page * PAGE_SIZE
    paged_rows = rows[start:start + PAGE_SIZE]
    table = _make_cm_detail_table(paged_rows, columns, badge_color)
    # Reserve space at bottom for pager/footer so table scrolling doesn't hide controls
    nav = html.Div([
        html.Button('Previous', id=ids.CM_PAGE_PREV, n_clicks=0, disabled=(page == 0),
                    className='cm-page-button',
                    style={'padding': '7px 14px', 'fontSize': '12px', 'borderRadius': '5px', 'border': '1px solid #999', 'backgroundColor': '#f8f9fa', 'color': '#333'}),
        html.Span(f'Page {page + 1} of {page_count}', style={'fontSize': '12px', 'color': '#555'}),
        html.Button('Next', id=ids.CM_PAGE_NEXT, n_clicks=0, disabled=(page >= page_count - 1),
                    className='cm-page-button',
                    style={'padding': '7px 14px', 'fontSize': '12px', 'borderRadius': '5px', 'border': '1px solid #999', 'backgroundColor': '#f8f9fa', 'color': '#333'}),
    ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'space-between', 'gap': '10px', 'marginBottom': '8px', 'flex': 'none'})

    # Table container — show only the current page rows (no internal scrolling)
    table_container = html.Div(table, style={'flex': '1 1 auto', 'minHeight': 0, 'overflowY': 'visible'})

    return html.Div([
        nav,
        table_container,
        html.Div(footer_text, style={'fontSize': '10px', 'color': '#aaa', 'textAlign': 'right', 'marginTop': '4px', 'flex': 'none'})
    ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '10px', 'flex': '1 1 auto', 'minHeight': 0, 'overflow': 'visible'}), {
        'rows': rows,
        'columns': columns,
        'badge_color': badge_color,
        'footer_text': footer_text,
        'page': page,
        'page_count': page_count,
    }


def node_confusion_figure(true_labels, preds, num_classes, dataset_name):
    """Node-classification confusion matrix. Axis values are numeric class
    indices (clickData stays parseable); class names appear in the hover."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, preds, labels=list(range(num_classes)))
    # Numeric x/y keep clickData parseable by update_edge_list_from_cm;
    # class names are shown as tick labels only.
    names = [class_label(dataset_name, i) for i in range(num_classes)]
    hover = [[f"Actual: {names[a]}<br>Predicted: {names[p]}<br>{cm[a][p]} nodes<br><i>click to list them</i>"
              for p in range(num_classes)] for a in range(num_classes)]
    cm_fig = go.Figure(data=go.Heatmap(
        z=cm, x=list(range(num_classes)), y=list(range(num_classes)),
        colorscale='Blues', text=cm, texttemplate="%{text}", hoverongaps=False,
        hovertext=hover, hovertemplate="%{hovertext}<extra></extra>", showscale=False,
    ))
    # Index-only ticks keep the matrix compact; hover shows the class names
    idx_ticks = [str(i) for i in range(num_classes)]
    cm_fig.update_layout(
        title=dict(text="Confusion matrix (click a cell)", x=0.5, font=dict(size=13)),
        xaxis=dict(title=dict(text="Predicted class", standoff=4), tickmode='array',
                   tickvals=list(range(num_classes)), ticktext=idx_ticks, side='bottom',
                   constrain='domain'),
        yaxis=dict(title=dict(text="Actual class", standoff=4), tickmode='array',
                   tickvals=list(range(num_classes)), ticktext=idx_ticks, autorange='reversed',
                   scaleanchor='x', scaleratio=1, constrain='domain'),
        autosize=True,
        margin=dict(l=44, r=8, t=30, b=40),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        clickmode='event+select',
        dragmode=False
    )
    return cm_fig


def link_confusion_figure(true_labels, predicted_labels, tp_list, fp_list, fn_list, tn_list):
    """2x2 link-prediction confusion matrix; hover lists the top links per cell.
    Each list holds (score, u, v), already sorted for display."""
    from sklearn.metrics import confusion_matrix
    def format_hover(link_list, label, k=5):

        count = len(link_list)

        header = f"<b>{label}</b><br>Count: {count}<br><br>Top Contributing Links:<br>"

        rows = []

        for s, u, v in link_list[:k]:

            rows.append(f"{u} → {v} ({s:.4f})")

        return header + "<br>".join(rows)

    # Confusion Matrix Data

    cm = confusion_matrix(true_labels, predicted_labels)

    # The matrix structure from sklearn is:

    # [[TN, FP],

    #  [FN, TP]]

    # But we check the shape to be sure. If binary, it's 2x2.

    if cm.shape != (2, 2):

        # Fallback if somehow not 2x2 (shouldn't happen with our data)

        z_data = cm

        hover_text = [["", ""], ["", ""]]

    else:

        z_data = cm

        # Map lists to cells corresponding to sklearn output

        # [0,0]=TN, [0,1]=FP

        # [1,0]=FN, [1,1]=TP

        hover_text = [

            [format_hover(tn_list, "True Negatives (TN)"), format_hover(fp_list, "False Positives (FP)")],

            [format_hover(fn_list, "False Negatives (FN)"), format_hover(tp_list, "True Positives (TP)")]

        ]

    # Create confusion matrix plot

    fig = go.Figure(data=go.Heatmap(

        z=z_data,

        x=['Predicted: No Edge', 'Predicted: Edge'],

        y=['Actual: No Edge', 'Actual: Edge'],

        colorscale='Blues',

        text=z_data,

        texttemplate="%{text}",
        textfont={"size": 12},
        customdata=hover_text,
        hovertemplate="%{customdata}<extra></extra>"
    ))

    fig.update_layout(
        title="Link prediction: held-out test edges",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        font=dict(size=11),
        margin=dict(l=20, r=20, t=28, b=20),
        height=220,
        clickmode='event+select',
        dragmode=False,
        autosize=True,
    )
    return fig

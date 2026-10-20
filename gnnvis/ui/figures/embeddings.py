"""Node-embedding scatter plot (2-D projection coloured by class / correctness / confidence)."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from gnnvis.ui.theme import CORRECT_COLOR, INCORRECT_COLOR, class_label, empty_figure


def get_2d_embeddings(package):
    """Return a 2D embedding projection from the loaded package.

    Priority:
      1. precomputed UMAP
      2. precomputed PCA/TSNE
      3. fallback PCA from raw high-dimensional embeddings
    """
    if not package:
        return None

    embed2d = package.get('embeddings_2d', {})
    for method in ('umap', 'pca', 'tsne'):
        if method in embed2d and embed2d[method] is not None:
            try:
                arr = np.array(embed2d[method])
                if arr.ndim == 2 and arr.shape[1] == 2:
                    return arr
            except Exception:
                pass

    raw_embeddings = package.get('embeddings')
    if raw_embeddings is None:
        return None

    try:
        raw_arr = np.array(raw_embeddings)
        if raw_arr.ndim == 2 and raw_arr.shape[1] >= 2:
            pca = PCA(n_components=2)
            return pca.fit_transform(raw_arr)
    except Exception as e:
        print(f"[EMBEDDINGS] Failed to compute fallback PCA projection: {e}")

    return None


EMBEDDING_COLOR_TITLES = {
    'actual': 'true class', 'predicted': 'predicted class',
    'correctness': 'correct / incorrect', 'confidence': 'model confidence',
}


def plot_embeddings(embeddings_2d, predictions, true_labels, selected_node_id, class_to_color_map, num_classes,
                    target_node_id=None, dataset_name=None, color_mode='actual', confidences=None):

    if embeddings_2d is None: return empty_figure("Embeddings not available")

    df = pd.DataFrame(embeddings_2d, columns=['Dim 1', 'Dim 2'])

    df['Node ID'] = [str(i) for i in range(len(df))]

    names = [class_label(dataset_name, c) for c in range(num_classes)]
    preds_arr, true_arr = np.asarray(predictions), np.asarray(true_labels)

    df['Predicted Class'] = [class_label(dataset_name, int(p)) for p in preds_arr]
    df['True Label'] = [class_label(dataset_name, int(t)) for t in true_arr]
    df['Prediction'] = np.where(preds_arr == true_arr, 'Correct', 'Incorrect')
    has_conf = confidences is not None and len(confidences) == len(df)
    if has_conf:
        df['Confidence'] = np.round(np.asarray(confidences, dtype=float), 3)
    elif color_mode == 'confidence':
        color_mode = 'actual'   # no confidences available: fall back

    hover = ['Node ID', 'True Label', 'Predicted Class'] + (['Confidence'] if has_conf else [])
    class_colors = {names[k]: v for k, v in class_to_color_map.items() if k < len(names)}
    common = dict(x='Dim 1', y='Dim 2', hover_data=hover, custom_data=['Node ID'], template='plotly_white')

    # Order fixed so a class keeps its colour whatever the mode
    if color_mode == 'predicted':
        fig = px.scatter(df, color='Predicted Class', color_discrete_map=class_colors,
                         category_orders={'Predicted Class': names}, **common)
    elif color_mode == 'correctness':
        fig = px.scatter(df, color='Prediction',
                         color_discrete_map={'Correct': CORRECT_COLOR, 'Incorrect': INCORRECT_COLOR},
                         category_orders={'Prediction': ['Correct', 'Incorrect']}, **common)
    elif color_mode == 'confidence':
        # Low confidence = dark: uncertain regions stand out against confident clusters
        fig = px.scatter(df, color='Confidence', color_continuous_scale='Viridis', range_color=[0, 1], **common)
        fig.update_layout(coloraxis_colorbar=dict(title='Conf.', thickness=10, len=0.55, x=1.01, y=0.25, yanchor='middle'))
    else:
        fig = px.scatter(df, color='True Label', color_discrete_map=class_colors,
                         category_orders={'True Label': names}, **common)
    fig.update_traces(marker=dict(symbol='circle', size=6, opacity=0.8, line=dict(width=0)))

    # Ring misclassified nodes (redundant in correctness mode, where colour already says it)
    wrong = np.where(preds_arr != true_arr)[0]
    if len(wrong) and color_mode != 'correctness':
        fig.add_trace(go.Scatter(
            x=df['Dim 1'].values[wrong], y=df['Dim 2'].values[wrong], mode='markers',
            marker=dict(symbol='circle-open', size=9, color=INCORRECT_COLOR, line=dict(width=1.2)),
            name='Misclassified', hoverinfo='skip',
            customdata=[[str(i)] for i in wrong],
        ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    # Highlight Source Node
    if selected_node_id is not None:
        try:
            # Ensure ID is int for indexing if dataframe index matches node ID
            idx = int(selected_node_id)
            if 0 <= idx < len(df):
                source_row = df.iloc[idx]
                fig.add_trace(go.Scatter(
                    x=[source_row['Dim 1']], 
                    y=[source_row['Dim 2']],
                    mode='markers',
                    marker=dict(symbol='star', size=20, color='#f1c40f', line=dict(width=2, color='black')),
                    name='Selected',
                    hoverinfo='text',
                    text=f"Selected: Node {selected_node_id}",
                    customdata=[[str(idx)]],
                ))
        except: pass

    # Highlight Target Node
    if target_node_id is not None:
        try:
            idx = int(target_node_id)
            if 0 <= idx < len(df):
                target_row = df.iloc[idx]
                fig.add_trace(go.Scatter(
                    x=[target_row['Dim 1']], 
                    y=[target_row['Dim 2']],
                    mode='markers',
                    marker=dict(symbol='diamond', size=18, color='#e74c3c', line=dict(width=2, color='black')),
                    name='Target (v)',
                    hoverinfo='text',
                    text=f"Target: Node {target_node_id}",
                    customdata=[[str(idx)]],
                ))
                
                # Draw Line
                if selected_node_id is not None:
                     s_idx = int(selected_node_id)
                     s_row = df.iloc[s_idx]
                     fig.add_trace(go.Scatter(
                         x=[s_row['Dim 1'], target_row['Dim 1']],
                         y=[s_row['Dim 2'], target_row['Dim 2']],
                         mode='lines',
                         line=dict(color='#333', width=2, dash='dot'),
                         showlegend=False,
                         hoverinfo='skip'
                     ))
        except: pass

    fig.update_layout(

        title={

            'text': f"Node embeddings (colour = {EMBEDDING_COLOR_TITLES.get(color_mode, 'true class')})",

            'font': {'size': 13},

            'y':0.98,

            'x':0.5,

            'xanchor': 'center',

            'yanchor': 'top'

        },

        legend_title_text=None,

        xaxis=dict(title=None, showticklabels=False, zeroline=False),

        yaxis=dict(title=None, showticklabels=False, zeroline=False),

        clickmode='event+select', 

        paper_bgcolor='rgba(0,0,0,0)', 

        plot_bgcolor='rgba(0,0,0,0)',

        # Compact vertical key on the right: a horizontal key below the plot
        # wraps into several rows and overlaps the points on short panels
        legend=dict(

            orientation="v",

            font=dict(size=10),

            tracegroupgap=0,

            yanchor="top",

            y=1,

            xanchor="left",

            x=1.01,

            bgcolor='rgba(255,255,255,0)',

        ),

        margin=dict(t=30, b=6, l=6, r=6)

    )

    return fig

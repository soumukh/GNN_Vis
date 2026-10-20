"""Shared visual encoding: class names, class colours, correct/incorrect colours, empty figures."""

import plotly.express as px
import plotly.graph_objects as go


# --- Shared visual encoding ---
# Class names in Planetoid label-index order. Verified against per-index node
# counts (Cora: 351/217/418/818/426/298/180; CiteSeer index 0 = AI's 249 nodes
# plus the 15 isolated nodes PyG assigns label 0).
CLASS_NAMES = {
    'Cora': ['Theory', 'Reinforcement Learning', 'Genetic Algorithms', 'Neural Networks',
             'Probabilistic Methods', 'Case Based', 'Rule Learning'],
    'CiteSeer': ['AI', 'ML', 'IR', 'DB', 'Agents', 'HCI'],
}


CLASS_PALETTE = px.colors.qualitative.Plotly


# Colour-blind-safe pair for correct / incorrect (instead of green / red)
CORRECT_COLOR = '#2166ac'


INCORRECT_COLOR = '#e66101'


def class_name(dataset_name, idx):
    """Human-readable class name, falling back to 'Class k'."""
    try:
        idx = int(idx)
    except (TypeError, ValueError):
        return str(idx)
    names = CLASS_NAMES.get(dataset_name)
    if names and 0 <= idx < len(names):
        return names[idx]
    return f'Class {idx}'


def class_label(dataset_name, idx):
    """Class index plus name, e.g. '3: Neural Networks'."""
    return f"{idx}: {class_name(dataset_name, idx)}"


def class_color_map(num_classes):
    """The single class -> colour mapping used by every view."""
    return {i: CLASS_PALETTE[i % len(CLASS_PALETTE)] for i in range(num_classes)}


def empty_figure(message='', height=None):
    """Blank figure with a centred hint instead of empty axes."""
    layout = dict(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    if message:
        layout['annotations'] = [dict(
            text=message, xref='paper', yref='paper', x=0.5, y=0.5,
            showarrow=False, font=dict(size=13, color='#8a94a6'),
        )]
    if height:
        layout['height'] = height
    return go.Figure(layout=layout)

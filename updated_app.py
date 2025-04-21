
import dash
from dash import dcc, html, Input, Output, State, callback, ctx, no_update, ALL
import dash_cytoscape as cyto
import plotly.express as px
import plotly.graph_objects as go # For histogram and empty figures

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx, remove_self_loops, add_self_loops

from node2vec import Node2Vec
import networkx as nx
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import pickle
import os
import copy # For deep copying model state
import warnings

warnings.filterwarnings("ignore", category=UserWarning) # Suppress some warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

# --- Global Variables / Setup ---
MODEL_DIR = 'models' # Directory where trained models are saved
DATA_DIR = 'data'    # Directory where datasets are stored
DEFAULT_DATASET = 'Cora'
DEFAULT_MODEL = 'GCN'
AVAILABLE_DATASETS = ['Cora', 'CiteSeer', 'PubMed', 'Jazz']
AVAILABLE_MODELS = ['GCN', 'GAT']
NODE_DIM_REDUCTION = 'TSNE' # 'PCA' or 'TSNE'

# --- GPU Setup ---
# Check for CUDA availability and set the device
if torch.cuda.is_available():
    device = torch.device('cuda:0') # Use the first available GPU
    print(f"CUDA is available. Using device: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device('cpu')
    print("CUDA not available. Using CPU.")

# Ensure model and data directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# --- Model Definitions (Must match the ones in the training script) ---
class GCNNet(torch.nn.Module):
    """Basic GCN Network Definition."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.init_args = {'hidden_channels': hidden_channels} # Store args

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Note: Dropout is applied differently during eval vs train.
        # Ensure model.eval() is called for inference.
        # x = F.dropout(x, p=0.5, training=self.training)
        embeddings = x # Get embeddings from the hidden layer
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1), embeddings

class GATNet(torch.nn.Module):
    """Basic GAT Network Definition."""
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        # Use dropout=0.6 consistent with the original GAT paper
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        num_heads_out = 1
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=num_heads_out,
                             concat=False, dropout=0.6)
        self.init_args = {'hidden_channels': hidden_channels, 'heads': heads} # Store args

    def forward(self, data, return_attention_weights=False):
        x, edge_index = data.x, data.edge_index
        # Dropout layers are automatically handled by model.train() / model.eval()
        x, att1 = self.conv1(x, edge_index, return_attention_weights=return_attention_weights)
        x = F.elu(x)
        embeddings = x # Use embeddings after first layer + activation
        x, att2 = self.conv2(x, edge_index, return_attention_weights=return_attention_weights)

        attention_weights = (att1, att2) if return_attention_weights else None

        if return_attention_weights:
            return F.log_softmax(x, dim=1), embeddings, attention_weights
        else:
            return F.log_softmax(x, dim=1), embeddings

# --- Helper Functions ---
def load_dataset(name):
    """Loads a dataset, handling both Planetoid and custom Jazz dataset."""
    try:
        script_dir = os.path.dirname(__file__) if '__file__' in locals() else '.'
        # Construct path relative to the script directory
        path = os.path.join(script_dir, DATA_DIR, name)
        print(f"Attempting to load dataset '{name}' from path: {path}")

        if name != 'Jazz':
            # For Planetoid datasets, PyG handles downloading if not present in 'root'
            dataset = Planetoid(root=path, name=name)
            print(f"Dataset '{name}' loaded successfully. Path: {dataset.raw_dir}")
            return dataset
        else:
            return load_jazz_dataset(path) # Jazz needs specific handling
    except Exception as e:
        print(f"Error loading dataset {name}: {e}")
        print(f"Please ensure the dataset is available in the '{DATA_DIR}/{name}' directory or can be downloaded by PyG.")
        return None

def load_jazz_dataset(path):
    """Loads the Jazz dataset with Node2Vec embeddings."""
    try:
        edge_file = os.path.join(path, 'jazz.cites')
        feature_file = os.path.join(path, 'jazz.content')

        if not os.path.exists(edge_file) or not os.path.exists(feature_file):
            print(f"Jazz dataset files not found in {path}. Required: 'jazz.cites', 'jazz.content'")
            return None

        # Load edges
        edges = np.loadtxt(edge_file, dtype=int, delimiter='\t')

        # Load content (node_id, features..., label)
        # Using pandas for more robust loading, especially if delimiters are tricky
        content_df = pd.read_csv(feature_file, sep='\t', header=None)
        node_ids = content_df.iloc[:, 0].astype(int).values
        features = content_df.iloc[:, 1:-1].values # All columns between first and last
        labels = content_df.iloc[:, -1].astype(int).values

        # Remap node IDs to consecutive integers starting from 0
        node_map = {old_id: new_id for new_id, old_id in enumerate(sorted(np.unique(node_ids)))}
        num_nodes_original = len(node_map)

        # Apply mapping to edges
        mapped_edges = []
        valid_nodes = set(node_map.keys())
        for u, v in edges:
            if u in valid_nodes and v in valid_nodes:
                mapped_edges.append([node_map[u], node_map[v]])
            else:
                print(f"Warning: Skipping edge ({u}, {v}) due to unknown node ID(s).")
        edge_index = torch.tensor(mapped_edges, dtype=torch.long).t().contiguous()

        # Align features and labels with the new mapping
        # Create arrays indexed by the *new* IDs
        num_features = features.shape[1]
        aligned_features = np.zeros((num_nodes_original, num_features))
        aligned_labels = np.zeros(num_nodes_original, dtype=int)
        for i, old_id in enumerate(content_df.iloc[:, 0].astype(int).values):
             if old_id in node_map: # Ensure the node from content exists in the map
                 new_id = node_map[old_id]
                 aligned_features[new_id] = content_df.iloc[i, 1:-1].values
                 aligned_labels[new_id] = content_df.iloc[i, -1].astype(int)

        # Generate Node2Vec embeddings (based on the *original* graph structure before remapping nodes)
        G = nx.Graph() # Undirected graph for Node2Vec
        G.add_nodes_from(node_map.keys()) # Add original node IDs
        G.add_edges_from(edges) # Add original edges

        print("Generating Node2Vec embeddings for Jazz dataset...")
        node2vec = Node2Vec(G, dimensions=16, walk_length=30, num_walks=200, workers=4, quiet=True)
        model_n2v = node2vec.fit(window=10, min_count=1, batch_words=4)
        print("Node2Vec fitting complete.")

        # Create embedding matrix aligned with the *new* node IDs
        embeddings = np.zeros((num_nodes_original, 16))
        for old_id, new_id in node_map.items():
            try:
                embeddings[new_id] = model_n2v.wv[str(old_id)]
            except KeyError:
                print(f"Warning: Node ID {old_id} not found in Node2Vec model vocabulary. Using zero vector.")
                # Keep the zero vector initialized above

        # Concatenate base features with Node2Vec embeddings
        x = torch.tensor(np.hstack((aligned_features, embeddings)), dtype=torch.float)
        y = torch.tensor(aligned_labels, dtype=torch.long)

        # Create train/val/test masks (simple random split for Jazz)
        num_nodes = x.size(0)
        perm = torch.randperm(num_nodes)
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[perm[:train_size]] = True
        val_mask[perm[train_size:train_size + val_size]] = True
        test_mask[perm[train_size + val_size:]] = True

        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
        data.num_nodes = num_nodes # Explicitly set num_nodes

        # Create a dataset-like object for consistency
        dataset = type('JazzDataset', (), {
            'data': data,
            'num_node_features': x.size(1),
            'num_classes': len(torch.unique(y)),
            '__getitem__': lambda self, idx: self.data if idx == 0 else None,
            '__len__': lambda self: 1,
            'name': 'Jazz' # Add name attribute
        })()

        print("Dataset 'Jazz' loaded and processed successfully.")
        return dataset

    except Exception as e:
        print(f"Error loading or processing Jazz dataset: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback
        return None


def load_model(model_type, dataset_name, device):
    """Loads a pre-trained model from a .pkl file onto the specified device."""
    model_filename = f"{model_type}_{dataset_name}.pkl"
    script_dir = os.path.dirname(__file__) if '__file__' in locals() else '.'
    model_path = os.path.join(script_dir, MODEL_DIR, model_filename)

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print(f"Please run the training script (e.g., train_models.py) first to generate '{model_filename}'.")
        return None

    try:
        # Load dataset info first to get dimensions
        dataset = load_dataset(dataset_name)
        if dataset is None: return None
        in_channels = dataset.num_node_features
        out_channels = dataset.num_classes

        # Load model state dict and args, mapping storage to the target device
        with open(model_path, 'rb') as f:
            # map_location handles moving the loaded tensors to the correct device
            model_state_dict, model_args = pickle.load(f) # Removed map_location here, apply later

        # Recreate model instance
        if model_type == 'GCN':
            hidden_channels = model_args.get('hidden_channels', 16)
            model = GCNNet(in_channels, hidden_channels, out_channels)
        elif model_type == 'GAT':
            hidden_channels = model_args.get('hidden_channels', 8)
            heads = model_args.get('heads', 8)
            model = GATNet(in_channels, hidden_channels, out_channels, heads=heads)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Load state dict and move model to device *before* returning
        model.load_state_dict(model_state_dict)
        model.to(device) # Move the entire model to the target device
        model.eval() # Set model to evaluation mode

        print(f"Loaded model {model_filename} to {device} with args: {model_args}")
        return model
    except FileNotFoundError:
         print(f"Error: Model file not found at {model_path}. Ensure it exists.")
         return None
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

@torch.no_grad() # Disable gradient calculations for inference
def run_inference(model, data, model_type, device):
    """Runs GNN inference on the specified device."""
    if model is None or data is None:
        return None, None

    model.eval() # Ensure model is in evaluation mode
    model.to(device) # Ensure model is on the correct device
    data = data.to(device) # Move data to the correct device

    try:
        if model_type == 'GAT':
            log_probs, embeddings, *attention = model(data, return_attention_weights=False)
        else: # GCN
            log_probs, embeddings = model(data)

        predictions = log_probs.argmax(dim=1)

        # Move results back to CPU for use with Dash/Plotly etc.
        return predictions.cpu(), embeddings.cpu()

    except Exception as e:
        print(f"Error during model inference on device {device}: {e}")
        import traceback
        traceback.print_exc()
        # This could happen if graph structure is incompatible after edits or device issues
        return None, None

@torch.no_grad() # Disable gradients for attention calculation too
def get_attention_weights(model, data, node_idx, device):
    """Extracts attention weights for a specific node (GAT only) on the specified device."""
    if not isinstance(model, GATNet) or data is None or node_idx is None:
        return None

    model.eval() # Ensure model is in eval mode
    model.to(device) # Ensure model is on the correct device
    data = data.to(device) # Move data to the correct device

    num_nodes = data.num_nodes
    if not (0 <= node_idx < num_nodes):
        print(f"Invalid node index {node_idx} for attention calculation.")
        return None

    try:
        # Run forward pass requesting attention weights
        _, _, attention_output = model(data, return_attention_weights=True)

        if attention_output is None:
             print("Model did not return attention weights.")
             return None

        # Visualize attention from the *last* GAT layer (att2)
        edge_index_att, alpha_att = attention_output[-1] # Get edge_index and scores from last GAT layer

        # Find edges pointing TO the selected node (target node = node_idx)
        mask = edge_index_att[1] == node_idx
        if not torch.any(mask):
            # print(f"Node {node_idx} has no incoming edges in the attention mechanism.")
            return None # No attention scores to show if no incoming edges

        # Get the source nodes of these edges and their corresponding attention scores
        neighbor_indices = edge_index_att[0][mask]
        attention_scores = alpha_att[mask].mean(dim=1).squeeze() # Avg over heads

        # Move results back to CPU
        return neighbor_indices.cpu().numpy(), attention_scores.cpu().numpy()

    except Exception as e:
        print(f"Error getting attention weights: {e}")
        import traceback
        traceback.print_exc()
        return None



def data_to_cytoscape(data, predictions=None, class_map=None, selected_node_id=None, selected_edge_data=None, neighbor_ids=None): # <-- ADDED neighbor_ids argument
    """Converts PyG Data object to Cytoscape elements list using classes for highlighting selected node and its neighbors."""
    if data is None or data.num_nodes == 0:
        return []
    if neighbor_ids is None: # Initialize neighbor_ids if not provided
        neighbor_ids = set()
    else:
        neighbor_ids = set(neighbor_ids) # Ensure it's a set for quick lookup

    print(f"Debug data_to_cytoscape: Received selected_node_id='{selected_node_id}' (Type: {type(selected_node_id)}), neighbor_ids count: {len(neighbor_ids)}") # DEBUG

    nodes = []
    default_color = '#808080'
    color_palette = px.colors.qualitative.Plotly

    preds_list = None
    if predictions is not None:
        # ... (prediction processing logic) ...
        if isinstance(predictions, torch.Tensor):
            preds_list = predictions.cpu().numpy()
        elif isinstance(predictions, (list, np.ndarray)):
            preds_list = np.array(predictions)
        if len(preds_list) != data.num_nodes:
            print(f"Warning: Predictions length mismatch. Ignoring predictions for coloring.")
            preds_list = None


    for i in range(data.num_nodes):
        node_id_str = str(i)
        node_data = {'id': node_id_str, 'label': f'Node {i}'}
        node_classes = [] # Start with empty classes list

        # --- Determine Base Color (Optional) ---
        node_base_color = default_color
        if preds_list is not None and i < len(preds_list):
             pred_class = int(preds_list[i])
             node_data['class_pred'] = pred_class
             node_base_color = color_palette[pred_class % len(color_palette)]
             if class_map and pred_class in class_map:
                 node_data['label'] += f'\n({class_map[pred_class]})'

        # Add true label info
        if hasattr(data, 'y') and data.y is not None and i < len(data.y):
             true_class = int(data.y[i].item())
             node_data['true_class'] = true_class
             if class_map and true_class in class_map:
                 node_data['true_label'] = class_map[true_class]
             elif true_class != -1:
                 node_data['true_label'] = f'Class {true_class}'
             else:
                 node_data['true_label'] = 'N/A'

        # --- Add classes based on state ---
        is_selected = False
        if selected_node_id is not None:
            # Explicit type check and comparison
            if not isinstance(selected_node_id, str):
                 print(f"Warning: selected_node_id in data_to_cytoscape is not a string: {selected_node_id}")
            if node_id_str == selected_node_id:
                 is_selected = True
                 print(f"Debug data_to_cytoscape: MATCH! Node '{node_id_str}' == Selected '{selected_node_id}'. Adding 'selected' class.") # DEBUG
                 node_classes.append('selected')

        is_neighbor = i in neighbor_ids
        if is_neighbor and not is_selected: # Don't apply neighbor style if it's the selected node
            node_classes.append('neighbor')

        # Assign classes string
        node_data['classes'] = ' '.join(node_classes)

        # Minimal inline style (if needed, e.g., dynamic base color)
        node_style = {'background-color': node_base_color}

        nodes.append({'data': node_data, 'style': node_style})


    # --- Edge Processing ---
    # (Similar logic: add a 'selected-edge' class instead of inline style overrides)
    edges = []
    default_edge_color = '#cccccc'
    selected_edge_line_color = '#FF0000' # Red selected edge line
    selected_edge_width = 3

    if data.edge_index is not None and data.edge_index.shape[1] > 0:
        edge_index_display = data.edge_index.cpu().numpy()
        sel_source = selected_edge_data.get('source') if selected_edge_data else None
        sel_target = selected_edge_data.get('target') if selected_edge_data else None

        for i in range(edge_index_display.shape[1]):
            source = str(edge_index_display[0, i])
            target = str(edge_index_display[1, i])

            if int(source) >= data.num_nodes or int(target) >= data.num_nodes:
                continue

            edge_id = f"{source}_{target}_{i}"
            edge_data = {'id': edge_id, 'source': source, 'target': target}
            edge_classes = [] # Start with empty edge classes

            # Check if this edge is selected
            is_selected_edge = False
            if sel_source is not None and sel_target is not None:
                 if (sel_source == source and sel_target == target) or \
                    (sel_source == target and sel_target == source):
                      is_selected_edge = True

            if is_selected_edge:
                 edge_classes.append('selected-edge') # Add selected class

            # Assign classes string
            edge_data['classes'] = ' '.join(edge_classes)

            # Append edge with data (no inline style needed if handled by stylesheet)
            edges.append({'data': edge_data}) # Remove 'style': edge_style


    return nodes + edges

# ... (Rest of helper functions)
# Ensure the update_visualizations callback NO LONGER outputs or modifies graph-view.zoom
# Use the version from the step where zoom was removed.

# ... (The rest of your app code: layout, other callbacks, run command) ...

def plot_embeddings(embeddings, predictions, true_labels=None, class_map=None, dim_reduction='TSNE', selected_node_id=None):
    """Generates Plotly scatter plot for node embeddings with hover info and selection highlight."""
    if embeddings is None:
        return go.Figure(layout={'title': "Embeddings (Not Available)", 'xaxis': {'visible': False}, 'yaxis': {'visible': False}})

    # Convert to numpy if tensors
    embeddings_np = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else np.array(embeddings)
    predictions_np = predictions.cpu().numpy() if isinstance(predictions, torch.Tensor) else np.array(predictions)
    true_labels_np = true_labels.cpu().numpy() if isinstance(true_labels, torch.Tensor) else np.array(true_labels) if true_labels is not None else None

    num_nodes = embeddings_np.shape[0]
    if num_nodes == 0 or (predictions_np is not None and embeddings_np.shape[0] != predictions_np.shape[0]) or \
       (true_labels_np is not None and embeddings_np.shape[0] != true_labels_np.shape[0]):
        print("Warning: Embeddings, predictions, or labels size mismatch or zero nodes.")
        return go.Figure(layout={'title': "Embeddings (Invalid Data)", 'xaxis': {'visible': False}, 'yaxis': {'visible': False}})

    if num_nodes < 2:
         return go.Figure(layout={'title': "Embeddings (Need >= 2 nodes)", 'xaxis': {'visible': False}, 'yaxis': {'visible': False}})

    # Dimensionality Reduction
    n_components = 2
    if num_nodes < n_components + 1 and dim_reduction == 'TSNE':
        print("Not enough samples for t-SNE, switching to PCA")
        dim_reduction = 'PCA'

    if dim_reduction == 'TSNE':
        perplexity = min(30.0, max(5.0, float(num_nodes - 1) / 3.0))
        try:
            reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto', n_iter=300)
        except TypeError:
             print("Using older TSNE initialization (no learning_rate='auto').")
             reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, init='pca', n_iter=300)
    else: # PCA
        reducer = PCA(n_components=n_components)

    try:
        embeddings_2d = reducer.fit_transform(embeddings_np)
    except Exception as e:
        print(f"Error during dimensionality reduction ({dim_reduction}): {e}")
        embeddings_2d = np.random.rand(num_nodes, n_components) # Dummy data

    # Create DataFrame for Plotly
    df = pd.DataFrame(embeddings_2d, columns=['Dim 1', 'Dim 2'])
    df['Node ID'] = [str(i) for i in range(num_nodes)] # Crucial for linking clicks

    # Add Predicted Class
    if predictions_np is not None:
        if class_map:
            df['Predicted Class'] = [class_map.get(int(p), f'Class {int(p)}') for p in predictions_np]
        else:
            df['Predicted Class'] = [f'Class {int(p)}' for p in predictions_np]
        color_column = 'Predicted Class'
    else:
        df['Predicted Class'] = 'N/A'
        color_column = None # No color if no predictions

    # Add True Label
    if true_labels_np is not None:
        if class_map:
            df['True Label'] = [class_map.get(int(t), f'Class {int(t)}') if t != -1 else 'N/A' for t in true_labels_np]
        else:
            df['True Label'] = [f'Class {int(t)}' if t != -1 else 'N/A' for t in true_labels_np]
    else:
        df['True Label'] = 'N/A'

    # Prepare hover data
    hover_data = ['Node ID', 'Predicted Class', 'True Label']

    # Generate plot
    try:
        fig = px.scatter(df, x='Dim 1', y='Dim 2', color=color_column,
                         hover_data=hover_data,
                         custom_data=['Node ID'], # Pass Node ID for click events
                         title=f'Node Embeddings ({dim_reduction})',
                         color_discrete_sequence=px.colors.qualitative.Plotly)

        fig.update_layout(legend_title_text='Predicted Class', clickmode='event+select') # Enable click events
        fig.update_traces(marker=dict(size=8, opacity=0.8),
                          unselected=dict(marker=dict(opacity=0.5)), # Dim unselected points if selection occurs
                          selected=dict(marker=dict(size=12, opacity=1.0))) # Highlight selected point

        # Manually highlight the selected node if provided
        if selected_node_id is not None:
             try:
                 selected_point_index = df[df['Node ID'] == selected_node_id].index[0]
                 fig.update_traces(selectedpoints=[selected_point_index], selector=dict(type='scatter'))
             except IndexError:
                 print(f"Warning: Selected node ID {selected_node_id} not found in embedding plot DataFrame.")


    except Exception as e:
        print(f"Error creating embeddings plot: {e}")
        fig = go.Figure(layout={'title': f"Embeddings Plot Error ({dim_reduction})", 'xaxis': {'visible': False}, 'yaxis': {'visible': False}})

    return fig

def plot_feature_histogram(features, node_id):
    """Creates a histogram (bar chart) of NON-ZERO features for a single node."""
    if features is None:
        # Return an empty figure with a message
        return go.Figure(layout={'title': "Feature Histogram (Invalid Data)", 'xaxis': {'visible': False}, 'yaxis': {'visible': False},
                                'annotations': [{'text': 'Invalid feature data.', 'xref': 'paper', 'yref': 'paper', 'showarrow': False}]})

    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()
    # Ensure it's a 1D array
    if features.ndim != 1:
         return go.Figure(layout={'title': f"Feature Histogram for Node {node_id} (Error: Expected 1D features)",
                                 'xaxis': {'visible': False}, 'yaxis': {'visible': False},
                                 'annotations': [{'text': 'Feature data has incorrect dimensions.', 'xref': 'paper', 'yref': 'paper', 'showarrow': False}]})


    # Find indices of non-zero features (use a small threshold for floating point)
    threshold = 1e-6
    non_zero_indices = np.where(np.abs(features) > threshold)[0]

    if len(non_zero_indices) == 0:
        # Handle case where all features are zero or below threshold
        return go.Figure(layout={'title': f"Feature Histogram for Node {node_id}",
                                'xaxis': {'visible': False}, 'yaxis': {'visible': False},
                                'annotations': [{'text': 'Node has no non-zero features.', 'xref': 'paper', 'yref': 'paper', 'showarrow': False}]})

    # Get the values and labels for non-zero features
    non_zero_values = features[non_zero_indices]
    non_zero_labels = [f'F_{i}' for i in non_zero_indices]

    # Create DataFrame for Plotly
    df = pd.DataFrame({
        'Feature Index': non_zero_labels,
        'Value': non_zero_values
    })

    # Optional: Sort by index number or value if desired
    # df['IndexInt'] = non_zero_indices
    # df = df.sort_values('IndexInt')
    df = df.sort_values('Value', ascending=False) # Sort by value might be more insightful


    title = f"Non-Zero Features for Node {node_id} ({len(non_zero_indices)} features)"
    fig = px.bar(df, x='Feature Index', y='Value', title=title)

    # Improve layout: rotate labels if many features are shown
    if len(non_zero_indices) > 20: # Rotate if more than 20 non-zero features
         fig.update_layout(xaxis_tickangle=-45)
    elif len(non_zero_indices) > 50: # Further adjustments if extremely crowded
         fig.update_layout(xaxis_tickangle=-60, xaxis={'tickmode': 'linear', 'dtick': 5}) # Show fewer ticks


    fig.update_layout(xaxis_title="Feature Index", yaxis_title="Feature Value")
    # Ensure y-axis range is appropriate, e.g., if all values are 1 for binary features
    if np.all(np.isclose(non_zero_values, 1.0)):
        fig.update_yaxes(range=[0, 1.2]) # Adjust range slightly above 1

    return fig

# --- Dash App Initialization ---
app = dash.Dash(__name__, suppress_callback_exceptions=True,
                external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'], # Basic CSS
                prevent_initial_callbacks='initial_duplicate')
server = app.server # Expose server for deployment

# --- App Layout ---
app.layout = html.Div([
    # --- Stores ---
    dcc.Store(id='current-graph-store'), # Stores graph structure (serializable: lists)
    dcc.Store(id='current-model-output-store'), # Stores model outputs (predictions, embeddings as lists)
    dcc.Store(id='selected-node-store'), # Stores {'id': str} or None
    dcc.Store(id='selected-edge-store'), # Stores {'id': str, 'source': str, 'target': str} or None
    dcc.Store(id='dataset-info-store'), # Stores {'class_map': dict, 'num_features': int, ...}

    # --- Header ---
    html.H1("Interactive GNN Explainer", style={'textAlign': 'center', 'marginBottom': '20px'}),

    # --- Control Panel ---
    html.Div([
        # Dataset and Model Selection
        html.Div([
            html.Label("Select Dataset:", style={'marginRight': '10px'}),
            dcc.Dropdown(
                id='dataset-dropdown',
                options=[{'label': ds, 'value': ds} for ds in AVAILABLE_DATASETS],
                value=DEFAULT_DATASET,
                clearable=False,
                style={'width': '150px', 'marginRight': '20px'}
            ),
            html.Label("Select Model:", style={'marginRight': '10px'}),
            dcc.Dropdown(
                id='model-dropdown',
                options=[{'label': m, 'value': m} for m in AVAILABLE_MODELS],
                value=DEFAULT_MODEL,
                clearable=False,
                style={'width': '150px'}
            ),
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),

        # Graph Editing Controls
        html.Div([
            html.Button('Add Node', id='add-node-button', n_clicks=0, style={'marginRight': '5px'}),
            html.Button('Remove Selected Node', id='remove-node-button', n_clicks=0, style={'marginRight': '20px'}),
            dcc.Input(id='new-edge-source', type='text', placeholder='Source Node ID', style={'width': '120px', 'marginRight': '5px'}),
            dcc.Input(id='new-edge-target', type='text', placeholder='Target Node ID', style={'width': '120px', 'marginRight': '5px'}),
            html.Button('Add Edge', id='add-edge-button', n_clicks=0, style={'marginRight': '5px'}),
            html.Button('Remove Selected Edge', id='remove-edge-button', n_clicks=0),
        ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '10px'})

    ], id='control-panel', style={'padding': '15px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'marginBottom': '20px', 'backgroundColor': '#f9f9f9'}),

    # --- Status Message Area ---
    html.Div(id='status-message-area', style={'padding': '10px', 'marginBottom': '10px', 'border': '1px dashed #ccc', 'borderRadius': '5px', 'minHeight': '40px', 'backgroundColor': '#f0f0f0', 'whiteSpace': 'pre-wrap'}),

    # --- Main Visualization Area (Split View) ---
    html.Div([
        # --- Left Column ---
        html.Div([
            html.H3("Graph View"),
            # --- ADDED Node Input ---
            html.Div([
                dcc.Input(
                    id='select-node-input',
                    type='number', # Use number input for node IDs
                    placeholder='Enter Node ID...',
                    min=0, # Minimum node ID is 0
                    step=1,
                    style={'width': '150px', 'marginRight': '5px'}
                ),
                html.Button('Select Node', id='select-node-button', n_clicks=0)
            ], style={'marginBottom': '10px'}), # Add some spacing

            cyto.Cytoscape(
                id='graph-view',
                layout={'name': 'cose', 'idealEdgeLength': 100, 'nodeOverlap': 20, 'refresh': 20, 'fit': True, 'padding': 30, 'randomize': False, 'componentSpacing': 100, 'nodeRepulsion': 400000, 'edgeElasticity': 100, 'nestingFactor': 5, 'gravity': 80, 'numIter': 1000, 'initialTemp': 200, 'coolingFactor': 0.95, 'minTemp': 1.0},
                style={'width': '100%', 'height': '450px', 'border': '1px solid #ddd', 'borderRadius': '5px'},
                # --- UPDATED STYLESHEET ---
                stylesheet=[
                    # Default Node Style (applied to all nodes initially)
                    {'selector': 'node',
                    'style': {
                        'label': 'data(label)',
                        'font-size': '10px',
                        'text-valign': 'center',
                        'text-halign': 'center',
                        'background-color': '#808080', # Default color
                        'width': 15,
                        'height': 15,
                        'color': '#fff', # Label color
                        'text-outline-width': 2, # Make label readable
                        'text-outline-color': '#555', # Dark outline for label
                        'border-width': 0,
                        'shape': 'ellipse' # Default shape
                    }
                    },
                    # Style for SELECTED Nodes (overrides defaults)
                    {'selector': 'node.selected', # Target nodes with class 'selected'
                    'style': {
                        'background-color': '#FFA500', # Orange fill
                        'width': 100,                   # Larger size
                        'height': 100,
                        'border-color': '#FF0000',     # Red border
                        'border-width': 20,             # Thick border
                        'shape': 'ellipse',            # Keep shape or change e.g. 'star'
                        'z-index': 9999                # Bring to front
                    }
                    },
                                    # --- ADDED Neighbor Style ---
                    {'selector': 'node.neighbor', # Target nodes with class 'neighbor'
                    'style': {
                        'border-color': '#007bff',     # Blue border for neighbors
                        'border-width': 3,
                        'border-style': 'dashed' # Dashed border to distinguish from selected solid border
                        # Keep size same as default unless desired otherwise
                    }
                    },
                    # --- END ADDED Neighbor Style ---
                    # Default Edge Style
                    {'selector': 'edge',
                    'style': {
                        'curve-style': 'bezier',
                        'target-arrow-shape': 'triangle',
                        'line-color': '#cccccc',
                        'target-arrow-color': '#cccccc',
                        'opacity': 0.7,
                        'width': 1.5
                    }
                    },
                    # Style for SELECTED Edges
                    {'selector': 'edge.selected-edge', # Target edges with class 'selected-edge'
                    'style': {
                        'line-color': '#FF0000',    # Red line color
                        'target-arrow-color': '#FF0000',
                        'width': 3,                # Thicker line
                        'z-index': 9998            # Bring edge forward
                    }
                    }
                    # Optional: Add selectors for prediction classes if desired
                    # {'selector': 'node[class_pred=0]', 'style': {'background-color': '#...'}},
                    # {'selector': 'node[class_pred=1]', 'style': {'background-color': '#...'}},
                ]
                # --- END UPDATED STYLESHEET ---
            ),
            html.H3("Node Embeddings"),
            dcc.Dropdown(
                id='dim-reduction-dropdown',
                options=[{'label': 't-SNE', 'value': 'TSNE'}, {'label': 'PCA', 'value': 'PCA'}],
                value=NODE_DIM_REDUCTION,
                clearable=False,
                style={'width': '100px', 'marginBottom': '10px'}
            ),
            # Embeddings plot - clicks trigger 'select-node-embedding-callback'
            dcc.Graph(id='embeddings-view', style={'height': '400px'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '2%'}),

        # --- Right Column ---
        html.Div([
            # Feature view - now a graph for histogram
            html.H3("Feature Histogram (Selected Node)"),
            dcc.Graph(id='feature-histogram-view', style={'height': '250px'}),

            html.H3("Selected Node/Edge Info"),
            html.Div(id='selected-info-view', style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '10px', 'minHeight': '100px', 'marginBottom': '10px', 'backgroundColor': '#f9f9f9'}),

                # --- ADDED Neighborhood Info View ---
            html.H3("Neighbor Analysis"),
            html.Div(id='neighborhood-info-view', style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '10px', 'minHeight': '150px', 'maxHeight': '250px', 'overflowY': 'auto', 'marginBottom': '10px', 'backgroundColor': '#f9f9f9', 'fontSize': 'small'}),
            # --- END ADDED Neighborhood Info View ---

                # --- ADDED Reasoning Output View ---
            html.H3("Classification Reasoning"),
            html.Div(id='reasoning-output', style={'padding': '10px', 'marginTop': '5px', 'fontStyle': 'italic', 'color': '#333'}),
            # --- END ADDED Reasoning Output View ---
            html.H3("Attention Weights (GAT Only)"),
            dcc.Graph(id='attention-view', style={'height': '250px'}) # Placeholder for attention vis
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ])
], style={'padding': '20px'})

# --- Callbacks ---

# Callback to load dataset and initialize graph store
@callback(
    Output('current-graph-store', 'data'),
    Output('dataset-info-store', 'data'),
    Output('status-message-area', 'children', allow_duplicate=True),
    Input('dataset-dropdown', 'value'),
    prevent_initial_call='initial_duplicate' # Runs on startup and dropdown change
)
def load_dataset_callback(dataset_name):
    """Loads dataset, stores its basic structure and info."""
    print(f"Loading dataset: {dataset_name}")
    status_message = f"Loading dataset '{dataset_name}'..."
    dataset = load_dataset(dataset_name) # Function defined earlier

    empty_graph_store = {
        'edge_index': [[], []], 'features': [], 'labels': [], 'num_nodes': 0,
        'train_mask': [], 'val_mask': [], 'test_mask': []
    }
    empty_dataset_info = {'class_map': {}, 'num_node_features': 0, 'num_classes': 0}

    if dataset is None:
        error_message = f"Error: Failed to load dataset '{dataset_name}'. Check console and data directory."
        return empty_graph_store, empty_dataset_info, error_message

    try:
        # Get the graph object (works for both Jazz and Planetoid datasets)
        data = dataset[0]

        # Validate required attributes
        required_attrs = ['edge_index', 'x', 'y', 'num_nodes'] # Masks are optional for basic viz
        missing_attrs = [attr for attr in required_attrs if not hasattr(data, attr)]
        if missing_attrs:
            error_message = f"Error: Dataset '{dataset_name}' missing attributes: {missing_attrs}"
            return empty_graph_store, empty_dataset_info, error_message

        # Prepare data for storage (convert tensors to lists/serializable types)
        graph_store_data = {
            'edge_index': data.edge_index.cpu().tolist() if data.edge_index is not None else [[],[]],
            'features': data.x.cpu().tolist() if data.x is not None else [],
            'labels': data.y.cpu().tolist() if data.y is not None else [],
            'num_nodes': data.num_nodes,
            # Include masks if they exist, otherwise empty lists
            'train_mask': data.train_mask.cpu().tolist() if hasattr(data, 'train_mask') and data.train_mask is not None else [False]*data.num_nodes,
            'val_mask': data.val_mask.cpu().tolist() if hasattr(data, 'val_mask') and data.val_mask is not None else [False]*data.num_nodes,
            'test_mask': data.test_mask.cpu().tolist() if hasattr(data, 'test_mask') and data.test_mask is not None else [False]*data.num_nodes,
        }

        # Create class map
        class_map = {}
        if hasattr(dataset, 'num_classes') and data.y is not None:
            num_classes = dataset.num_classes
            # Create a generic map first
            class_map = {i: f'Class {i}' for i in range(num_classes)}
            # Add specific names if known (adjust as needed)
            try:
                ds_name_lower = dataset_name.lower()
                if ds_name_lower == 'cora' and num_classes == 7:
                    class_map = {0: 'Theory', 1: 'Reinforcement Learning', 2: 'Genetic Algorithms', 3: 'Neural Networks', 4: 'Probabilistic Methods', 5: 'Case Based', 6: 'Rule Learning'}
                elif ds_name_lower == 'citeseer' and num_classes == 6:
                    class_map = {0: 'Agents', 1: 'AI', 2: 'DB', 3: 'IR', 4: 'ML', 5: 'HCI'}
                elif ds_name_lower == 'pubmed' and num_classes == 3:
                     class_map = {0: 'Diabetes Mellitus Experimental', 1: 'Diabetes Mellitus Type 1', 2: 'Diabetes Mellitus Type 2'}
                # Add Jazz map if applicable (assuming labels 0..N-1 exist)
                elif ds_name_lower == 'jazz':
                     unique_labels = sorted(torch.unique(data.y).tolist())
                     class_map = {i: f'Community {i}' for i in unique_labels if i != -1} # Assuming -1 is not a valid community

            except Exception as map_err:
                print(f"Warning: Could not apply specific class names for {dataset_name}: {map_err}")

        dataset_info = {
            'class_map': class_map,
            'num_node_features': dataset.num_node_features if hasattr(dataset, 'num_node_features') else 0,
            'num_classes': dataset.num_classes if hasattr(dataset, 'num_classes') else 0
        }

        status_message = f"Dataset '{dataset_name}' loaded. Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1] if data.edge_index is not None else 0}, Features: {dataset_info['num_node_features']}."
        print(status_message)
        return graph_store_data, dataset_info, status_message

    except Exception as e:
        error_message = f"Error processing data from '{dataset_name}': {e}"
        import traceback
        traceback.print_exc()
        return empty_graph_store, empty_dataset_info, error_message


# Callback to run inference when graph data or model changes
@callback(
    Output('current-model-output-store', 'data'),
    Output('status-message-area', 'children', allow_duplicate=True),
    Input('current-graph-store', 'data'), # Triggered when dataset/graph changes
    Input('model-dropdown', 'value'),     # Triggered when model changes
    State('dataset-dropdown', 'value'),   # Need dataset name for loading model
    prevent_initial_call=True # Prevent initial run until graph_store is populated
)
def run_inference_callback(graph_store_data, model_type, dataset_name):
    """Runs inference when the graph data or model selection changes."""
    triggered_input = ctx.triggered_id
    print(f"Running inference callback. Trigger: {triggered_input}")

    if not graph_store_data or graph_store_data.get('num_nodes', 0) == 0:
        print("Inference skipped: Graph data is empty or invalid.")
        # Return empty results, but don't necessarily show an error message yet
        return {'predictions': None, 'embeddings': None}, "Load a dataset and model to run inference."

    if not model_type or not dataset_name:
        print("Inference skipped: Missing model type or dataset name.")
        return no_update, "Select both a dataset and a model."

    status_message = f"Loading {model_type} model for {dataset_name} and running inference on {device}..."

    # Load the selected model onto the target device
    model = load_model(model_type, dataset_name, device)
    if model is None:
        error_message = f"Inference skipped: {model_type} model for {dataset_name} could not be loaded."
        print(error_message)
        return {'predictions': None, 'embeddings': None}, error_message

    # Reconstruct PyG Data object from stored components (on CPU first)
    try:
        current_data = Data(
            x=torch.tensor(graph_store_data['features'], dtype=torch.float),
            edge_index=torch.tensor(graph_store_data['edge_index'], dtype=torch.long),
            y=torch.tensor(graph_store_data['labels'], dtype=torch.long), # Include labels
            num_nodes=graph_store_data['num_nodes'] # Set num_nodes explicitly
        )
        # Add masks if needed (usually not critical for basic inference)
        # current_data.train_mask = torch.tensor(graph_store_data['train_mask'], dtype=torch.bool)
        # current_data.val_mask = torch.tensor(graph_store_data['val_mask'], dtype=torch.bool)
        # current_data.test_mask = torch.tensor(graph_store_data['test_mask'], dtype=torch.bool)
    except Exception as e:
        error_message = f"Error reconstructing graph data for inference: {e}"
        print(error_message)
        return {'predictions': None, 'embeddings': None}, error_message

    # Run inference (handles moving data to device and results back to CPU)
    predictions, embeddings = run_inference(model, current_data, model_type, device)

    if predictions is None or embeddings is None:
        error_message = f"Inference failed for {model_type} on {dataset_name} using {device}. Check graph/model compatibility."
        print(error_message)
        return {'predictions': None, 'embeddings': None}, error_message

    print(f"Inference complete. Predictions shape: {predictions.shape}, Embeddings shape: {embeddings.shape}")
    status_message = f"Inference complete for {model_type} on {dataset_name} (using {device})."

    # Store results (already on CPU, convert to lists for JSON)
    output_data = {
        'predictions': predictions.tolist(),
        'embeddings': embeddings.tolist()
    }
    return output_data, status_message

# --- Callbacks for Graph Editing ---
# (Add Node, Remove Node, Add Edge, Remove Edge - Keep these similar, ensuring they update 'current-graph-store')
# ... (Keep existing add_node, remove_node, add_edge, remove_edge callbacks) ...
# Important: Ensure these callbacks return the updated graph_store_data dictionary
# with lists, not tensors, to trigger the run_inference_callback correctly.

# Example: Add Node (ensure features are handled correctly)
# --- ADDED CALLBACK for Node Input Selection ---
@callback(
    Output('selected-node-store', 'data', allow_duplicate=True),
    Output('status-message-area', 'children', allow_duplicate=True),
    Output('select-node-input', 'value'), # Clear input on success
    Input('select-node-button', 'n_clicks'),
    State('select-node-input', 'value'),
    State('current-graph-store', 'data'),
    prevent_initial_call=True
)
def select_node_from_input(n_clicks, input_value, graph_store_data):
    """Updates the selected node store based on input box value."""
    if n_clicks == 0 or input_value is None:
        # Don't trigger on initial load or if input is empty
        return no_update, no_update, no_update

    status_message = ""
    selected_node_update = no_update
    input_reset = no_update # Don't reset input on error

    if graph_store_data is None or graph_store_data.get('num_nodes', 0) == 0:
        status_message = "Error: Graph data not loaded. Cannot select node."
        return selected_node_update, status_message, input_reset

    num_nodes = graph_store_data['num_nodes']

    try:
        node_id_to_select = int(input_value)
        if 0 <= node_id_to_select < num_nodes:
            # Valid node ID entered
            node_id_str = str(node_id_to_select)
            selected_node_update = {'id': node_id_str}
            status_message = f"Selected Node {node_id_str} via input."
            print(status_message)
            input_reset = None # Clear the input box on success
        else:
            # Node ID out of range
            status_message = f"Error: Node ID {node_id_to_select} is out of range (0 to {num_nodes - 1})."
            print(status_message)
    except (ValueError, TypeError):
        # Input was not a valid integer
        status_message = f"Error: Invalid input '{input_value}'. Please enter a valid integer Node ID."
        print(status_message)

    # Return the update for the store and the status message
    return selected_node_update, status_message, input_reset




@callback(
    Output('current-graph-store', 'data', allow_duplicate=True),
    Output('status-message-area', 'children', allow_duplicate=True),
    Input('add-node-button', 'n_clicks'),
    State('current-graph-store', 'data'),
    State('dataset-info-store', 'data'), # Need num_features
    prevent_initial_call=True
)
def add_node_callback(n_clicks, graph_store_data, dataset_info):
    """Adds a new node with zero features initially."""
    if n_clicks == 0 or not graph_store_data or not dataset_info:
        return no_update, no_update

    print("Add Node button clicked.")
    status_message = "Adding node..."
    try:
        num_nodes = graph_store_data['num_nodes']
        num_features = dataset_info.get('num_node_features', 0) # Get from dataset info

        if num_features == 0 and num_nodes > 0:
             # Try to infer from existing features if info store failed
             if graph_store_data['features']:
                 num_features = len(graph_store_data['features'][0])
             else:
                 status_message = "Cannot add node: Feature dimension is unknown."
                 print(status_message)
                 return no_update, status_message
        elif num_features == 0 and num_nodes == 0:
             status_message = "Cannot add node: Load a dataset first to determine feature dimension."
             print(status_message)
             return no_update, status_message


        new_node_id = num_nodes # ID of the new node
        # Create zero features for the new node
        new_features = [0.0] * num_features

        # Update graph_store_data (use deepcopy)
        new_graph_data = copy.deepcopy(graph_store_data)
        new_graph_data['num_nodes'] += 1
        new_graph_data['features'].append(new_features)
        # Add dummy label and mask values
        new_graph_data['labels'].append(-1) # Assign dummy label -1
        new_graph_data['train_mask'].append(False)
        new_graph_data['val_mask'].append(False)
        new_graph_data['test_mask'].append(False)

        status_message = f"Added Node {new_node_id} (with zero features). Total nodes: {new_graph_data['num_nodes']}"
        print(status_message)
        # Return updated graph_store_data to trigger re-inference
        return new_graph_data, status_message
    except Exception as e:
        error_message = f"Error adding node: {e}"
        print(error_message)
        import traceback
        traceback.print_exc()
        return no_update, error_message

# Remove Node Callback (ensure remapping logic is robust)
@callback(
    Output('current-graph-store', 'data', allow_duplicate=True),
    Output('status-message-area', 'children', allow_duplicate=True),
    Output('selected-node-store', 'data', allow_duplicate=True), # Clear selection after removal
    Input('remove-node-button', 'n_clicks'),
    State('selected-node-store', 'data'),
    State('current-graph-store', 'data'),
    prevent_initial_call=True
)
def remove_node_callback(n_clicks, selected_node_data, graph_store_data):
    """Removes the selected node and its incident edges, re-indexes remaining nodes/edges."""
    if n_clicks == 0 or not graph_store_data:
        return no_update, no_update, no_update
    if not selected_node_data or 'id' not in selected_node_data:
        return no_update, "Select a node to remove.", no_update

    try:
        node_id_to_remove = int(selected_node_data['id'])
    except (ValueError, TypeError):
         return no_update, "Invalid selected node ID.", no_update

    print(f"Attempting to remove node: {node_id_to_remove}")
    status_message = f"Removing node {node_id_to_remove}..."

    try:
        num_nodes = graph_store_data['num_nodes']
        if not (0 <= node_id_to_remove < num_nodes):
            status_message = f"Invalid node ID to remove: {node_id_to_remove}"
            return no_update, status_message, no_update
        if num_nodes <= 1:
             status_message = "Cannot remove the last node."
             return no_update, status_message, no_update

        # Use deepcopy
        new_graph_data = copy.deepcopy(graph_store_data)

        # --- Update Node-related Lists ---
        node_mask = [i != node_id_to_remove for i in range(num_nodes)]
        new_graph_data['features'] = [feat for i, feat in enumerate(new_graph_data['features']) if node_mask[i]]
        new_graph_data['labels'] = [lab for i, lab in enumerate(new_graph_data['labels']) if node_mask[i]]
        new_graph_data['train_mask'] = [m for i, m in enumerate(new_graph_data['train_mask']) if node_mask[i]]
        new_graph_data['val_mask'] = [m for i, m in enumerate(new_graph_data['val_mask']) if node_mask[i]]
        new_graph_data['test_mask'] = [m for i, m in enumerate(new_graph_data['test_mask']) if node_mask[i]]

        # --- Update Edge Index and Remap ---
        edge_index = torch.tensor(new_graph_data['edge_index'], dtype=torch.long)
        remapped_edge_index_list = [[], []] # Default to empty

        if edge_index.numel() > 0:
            # Mask edges connected to the removed node
            edge_mask = (edge_index[0] != node_id_to_remove) & (edge_index[1] != node_id_to_remove)
            new_edge_index = edge_index[:, edge_mask]

            if new_edge_index.numel() > 0:
                # Create mapping from old indices (0..N-1) to new indices (0..N-2)
                node_mapping = torch.full((num_nodes,), -1, dtype=torch.long)
                kept_node_indices = torch.arange(num_nodes)[torch.tensor(node_mask, dtype=torch.bool)]
                new_indices = torch.arange(len(kept_node_indices))
                node_mapping[kept_node_indices] = new_indices

                # Apply mapping to the filtered edge index
                remapped_edge_index = node_mapping[new_edge_index]

                # Filter out any invalid edges (shouldn't happen with correct logic)
                valid_edge_mask = torch.all(remapped_edge_index != -1, dim=0)
                remapped_edge_index = remapped_edge_index[:, valid_edge_mask]
                remapped_edge_index_list = remapped_edge_index.tolist()

        # Update graph data store
        new_graph_data['num_nodes'] = len(new_graph_data['features']) # New number of nodes
        new_graph_data['edge_index'] = remapped_edge_index_list

        status_message = f"Removed Node {node_id_to_remove}. New node count: {new_graph_data['num_nodes']}"
        print(status_message)
        # Return updated data and clear selection
        return new_graph_data, status_message, None # Clear selected node store

    except Exception as e:
        error_message = f"Error removing node {node_id_to_remove}: {e}"
        print(error_message)
        import traceback
        traceback.print_exc()
        return no_update, error_message, no_update


# Add Edge Callback
@callback(
    Output('current-graph-store', 'data', allow_duplicate=True),
    Output('status-message-area', 'children', allow_duplicate=True),
    Input('add-edge-button', 'n_clicks'),
    State('new-edge-source', 'value'),
    State('new-edge-target', 'value'),
    State('current-graph-store', 'data'),
    prevent_initial_call=True
)
def add_edge_callback(n_clicks, source_str, target_str, graph_store_data):
    """Adds a new edge (and its reverse if undirected visually) between specified nodes."""
    if n_clicks == 0 or not graph_store_data:
        return no_update, no_update
    if not source_str or not target_str:
        return no_update, "Enter both source and target node IDs."

    try:
        source_id = int(source_str)
        target_id = int(target_str)
    except (ValueError, TypeError):
        return no_update, "Invalid source or target ID (must be integers)."

    print(f"Attempting to add edge: {source_id} -> {target_id}")
    status_message = f"Adding edge {source_id} -> {target_id}..."

    try:
        num_nodes = graph_store_data['num_nodes']
        if not (0 <= source_id < num_nodes and 0 <= target_id < num_nodes):
            status_message = f"Invalid node ID(s). Must be between 0 and {num_nodes-1}."
            return no_update, status_message
        if source_id == target_id:
            # Optionally allow self-loops if desired, but GNN layers often handle them implicitly
            return no_update, "Self-loops are not added via this button."

        # Use deepcopy
        new_graph_data = copy.deepcopy(graph_store_data)
        edge_index_list = new_graph_data['edge_index']
        if not edge_index_list: edge_index_list = [[], []] # Ensure structure

        # Check if edge already exists (consider both directions for undirected graphs)
        exists = False
        if edge_index_list[0]: # Check only if list is not empty
            edge_set = set(zip(edge_index_list[0], edge_index_list[1]))
            if (source_id, target_id) in edge_set or (target_id, source_id) in edge_set:
                exists = True

        if exists:
            status_message = f"Edge between {source_id} and {target_id} already exists."
            return no_update, status_message

        # Append new edge (source, target) and its reverse for undirected representation
        edge_index_list[0].extend([source_id, target_id])
        edge_index_list[1].extend([target_id, source_id])

        new_graph_data['edge_index'] = edge_index_list
        status_message = f"Added edge between {source_id} and {target_id}."
        print(status_message)
        return new_graph_data, status_message

    except Exception as e:
        error_message = f"Error adding edge ({source_id} -> {target_id}): {e}"
        print(error_message)
        import traceback
        traceback.print_exc()
        return no_update, error_message

# Remove Edge Callback
@callback(
    Output('current-graph-store', 'data', allow_duplicate=True),
    Output('status-message-area', 'children', allow_duplicate=True),
    Output('selected-edge-store', 'data', allow_duplicate=True), # Clear selection
    Input('remove-edge-button', 'n_clicks'),
    State('selected-edge-store', 'data'),
    State('current-graph-store', 'data'),
    prevent_initial_call=True
)
def remove_edge_callback(n_clicks, selected_edge_data, graph_store_data):
    """Removes the selected edge (and its reverse)."""
    if n_clicks == 0 or not graph_store_data:
        return no_update, no_update, no_update
    if not selected_edge_data or 'source' not in selected_edge_data or 'target' not in selected_edge_data:
        return no_update, "Select an edge to remove.", no_update

    try:
        source_id = int(selected_edge_data['source'])
        target_id = int(selected_edge_data['target'])
    except (ValueError, TypeError, KeyError):
        return no_update, "Invalid selected edge data.", no_update

    print(f"Attempting to remove edge between {source_id} and {target_id}")
    status_message = f"Removing edge {source_id} <-> {target_id}..."

    try:
        # Use deepcopy
        new_graph_data = copy.deepcopy(graph_store_data)
        edge_index_list = new_graph_data['edge_index']

        if not edge_index_list or not edge_index_list[0]:
             status_message = f"Edge between {source_id} and {target_id} not found (no edges)."
             return no_update, status_message, None # Clear selection

        edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long)
        # Create mask to identify the edge(s) to remove (both directions)
        mask_forward = (edge_index_tensor[0] == source_id) & (edge_index_tensor[1] == target_id)
        mask_backward = (edge_index_tensor[0] == target_id) & (edge_index_tensor[1] == source_id)
        remove_mask = mask_forward | mask_backward

        if not torch.any(remove_mask):
            status_message = f"Edge between {source_id} and {target_id} not found."
            return no_update, status_message, None # Clear selection

        # Keep edges that are NOT the one(s) to be removed
        keep_mask = ~remove_mask
        new_edge_index = edge_index_tensor[:, keep_mask]

        # Update graph_store_data
        new_graph_data['edge_index'] = new_edge_index.tolist()
        status_message = f"Removed edge between {source_id} and {target_id}. New edge count: {new_edge_index.shape[1]}"
        print(status_message)
        # Return updated data and clear selection
        return new_graph_data, status_message, None

    except Exception as e:
        error_message = f"Error removing edge ({source_id} <-> {target_id}): {e}"
        print(error_message)
        import traceback
        traceback.print_exc()
        return no_update, error_message, no_update


# --- Callbacks for Storing Selections ---

# Store node selection from Graph View
@callback(
    Output('selected-node-store', 'data'),
    Output('selected-edge-store', 'data', allow_duplicate=True), # Clear edge selection
    Input('graph-view', 'tapNodeData'),
    prevent_initial_call=True
)
def store_selected_node_graph(tapNodeData):
    if tapNodeData and 'id' in tapNodeData:
        print(f"Node selected from Graph View: {tapNodeData['id']}")
        return {'id': tapNodeData['id']}, None # Store node ID, clear edge store
    return no_update, no_update # Don't clear if click is not on a node

# Store node selection from Embedding View
@callback(
    Output('selected-node-store', 'data', allow_duplicate=True),
    Output('selected-edge-store', 'data', allow_duplicate=True), # Clear edge selection
    Input('embeddings-view', 'clickData'), # Listen to clicks on embedding plot
    prevent_initial_call=True
)
def select_node_embedding_callback(clickData):
    if clickData and 'points' in clickData and clickData['points']:
        # Extract node ID from customdata associated with the clicked point
        point_info = clickData['points'][0]
        if 'customdata' in point_info and point_info['customdata']:
            selected_node_id = str(point_info['customdata'][0]) # Assuming Node ID is the first element
            print(f"Node selected from Embedding View: {selected_node_id}")
            return {'id': selected_node_id}, None # Update node store, clear edge store
    return no_update, no_update # No change if click is invalid

# Store edge selection from Graph View
@callback(
    Output('selected-edge-store', 'data'),
    Output('selected-node-store', 'data', allow_duplicate=True), # Clear node selection
    Input('graph-view', 'tapEdgeData'),
    prevent_initial_call=True
)
def store_selected_edge_graph(tapEdgeData):
    if tapEdgeData and 'source' in tapEdgeData and 'target' in tapEdgeData:
        print(f"Edge selected from Graph View: {tapEdgeData.get('id', 'N/A')} (Source: {tapEdgeData['source']}, Target: {tapEdgeData['target']})")
        return {'id': tapEdgeData.get('id'), 'source': tapEdgeData['source'], 'target': tapEdgeData['target']}, None # Store edge data, clear node store
    return no_update, no_update # Don't clear if click is not on an edge


def generate_reasoning_sentence(selected_node_idx, true_class, pred_class, neighbor_ids, neighbor_true_classes, neighbor_pred_classes, class_map, model_type, attention_data=None):
    """Generates a heuristic reasoning sentence for the classification."""
    if true_class is None or pred_class is None:
        return "Reasoning requires true and predicted labels for the selected node."

    true_label = class_map.get(true_class, f"Class {true_class}") if true_class != -1 else "N/A"
    pred_label = class_map.get(pred_class, f"Class {pred_class}")
    is_correct = true_class == pred_class
    num_neighbors = len(neighbor_ids)

    # Start the sentence
    verdict = "Correctly" if is_correct else "Incorrectly"
    reasoning = f"Node {selected_node_idx} was {verdict} classified as '{pred_label}' (True Label: '{true_label}'). "

    if num_neighbors == 0:
        reasoning += "The node has no neighbors, so the classification is based solely on its own features."
        return reasoning

    # --- Neighborhood Analysis ---
    true_class_neighbor_count = 0
    pred_class_neighbor_count = 0
    for neighbor_tc, neighbor_pc in zip(neighbor_true_classes, neighbor_pred_classes):
        if neighbor_tc is not None and neighbor_tc == true_class:
            true_class_neighbor_count += 1
        if neighbor_pc is not None and neighbor_pc == pred_class:
            pred_class_neighbor_count += 1

    neighborhood_factor = ""
    if pred_class_neighbor_count > num_neighbors / 2:
         # Majority of neighbors match prediction
         neighborhood_factor = f"A majority ({pred_class_neighbor_count}/{num_neighbors}) of its neighbors were predicted as the same class ('{pred_label}')."
         if not is_correct and true_class_neighbor_count <= num_neighbors / 2:
              neighborhood_factor += " This strong neighborhood agreement towards the wrong class might be a key factor."
         elif is_correct:
              neighborhood_factor += " This neighborhood agreement likely supported the correct classification."
    elif true_class_neighbor_count > num_neighbors / 2:
         # Majority of neighbors match true class (if prediction was wrong)
         if not is_correct:
             neighborhood_factor = f"Interestingly, a majority ({true_class_neighbor_count}/{num_neighbors}) of its neighbors belong to the node's true class ('{true_label}'), suggesting the misclassification might stem more from the node's own features or specific influential neighbors."
         # If prediction is correct, majority matching true class isn't the primary explanation for *why* it's correct (that would be majority matching predicted)
    else:
         neighborhood_factor = f"The neighborhood classes are mixed (Predicted: {pred_class_neighbor_count}/{num_neighbors}, True: {true_class_neighbor_count}/{num_neighbors})."
         if not is_correct:
              neighborhood_factor += " The lack of strong neighborhood consensus might make the classification harder."


    # --- Attention Analysis (GAT Only) ---
    attention_factor = ""
    if model_type == 'GAT' and attention_data is not None:
        att_neighbor_indices, att_scores = attention_data
        if len(att_neighbor_indices) > 0:
            # Find top attended neighbors
            top_k = min(3, len(att_neighbor_indices)) # Look at top 3
            sorted_indices = np.argsort(att_scores)[::-1] # Indices of scores sorted descending
            top_att_neighbors = att_neighbor_indices[sorted_indices[:top_k]]
            top_att_scores = att_scores[sorted_indices[:top_k]]

            # Get classes of top attended neighbors
            top_neighbor_details = []
            for idx, score in zip(top_att_neighbors, top_att_scores):
                 # Find this neighbor's predicted class from the main neighbor list
                 neighbor_pred_class = None
                 try:
                      list_idx = neighbor_ids.index(idx) # Find index in the original neighbor list
                      neighbor_pred_class = neighbor_pred_classes[list_idx]
                 except ValueError:
                      pass # Neighbor from attention not found in main list (shouldn't happen often)

                 n_pred_label = class_map.get(neighbor_pred_class, f"Class {neighbor_pred_class}") if neighbor_pred_class is not None else "N/A"
                 top_neighbor_details.append(f"Node {idx} (Pred: '{n_pred_label}', Score: {score:.2f})")

            attention_factor = f" The model paid highest attention to neighbors: {', '.join(top_neighbor_details)}."

            # Add simple interpretation based on top attended predicted classes
            top_attended_pred_classes = [neighbor_pred_classes[neighbor_ids.index(idx)] for idx in top_att_neighbors if idx in neighbor_ids]
            if top_attended_pred_classes:
                 # Count how many top attended neighbors match the node's predicted class
                 match_pred_count = sum(1 for c in top_attended_pred_classes if c == pred_class)
                 if match_pred_count >= top_k / 2 and match_pred_count > 0: # If at least half match prediction
                      attention_factor += f" High attention towards nodes predicted as '{pred_label}' could explain the result."
                 elif not is_correct:
                     # If incorrect, and attention was NOT primarily on predicted class nodes
                     attention_factor += " Attention was not primarily focused on neighbors matching the incorrect prediction."


    # --- Combine factors ---
    if model_type == 'GAT' and attention_factor:
         reasoning += neighborhood_factor + attention_factor
    else:
         reasoning += neighborhood_factor # GCN or GAT w/o attention info

    return reasoning


# --- Callback to Update ALL Visualizations ---
@callback(
    Output('graph-view', 'elements'),
    Output('embeddings-view', 'figure'),
    # Make sure the ID matches the dcc.Graph component in the layout
    Output('feature-histogram-view', 'figure'), # ID for the feature plot Graph component
    Output('selected-info-view', 'children'),
    Output('attention-view', 'figure'),
    Output('neighborhood-info-view', 'children'),
    Output('reasoning-output', 'children'),
    # Triggers:
    Input('current-model-output-store', 'data'),
    Input('selected-node-store', 'data'),
    Input('selected-edge-store', 'data'),
    Input('dim-reduction-dropdown', 'value'),
    # States:
    State('current-graph-store', 'data'),
    State('dataset-info-store', 'data'),
    State('model-dropdown', 'value'),
    State('dataset-dropdown', 'value'),
    prevent_initial_call=True
)
def update_visualizations(model_output, selected_node_data, selected_edge_data,
                          dim_reduction_method,
                          graph_store_data, dataset_info, model_type, dataset_name):
    """Updates all visualization components based on current state."""
    trigger = ctx.triggered_id
    print(f"\n--- Debug: update_visualizations triggered by: {trigger} ---") # DEBUG

    reasoning_content = "" # Default empty reasoning
    # Define default empty/placeholder states
    empty_elements = []
    empty_fig = go.Figure(layout={'xaxis': {'visible': False}, 'yaxis': {'visible': False}, 'annotations': [{'text': 'N/A', 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 12}}]})
    default_info_text = "Load data and select a model, then click on a node or edge."
    default_attn_text = "Attention view requires a GAT model and a selected node."
    default_feature_text = "Select a single node to view its feature histogram."
    attn_fig = go.Figure(layout={'title': default_attn_text, 'xaxis': {'visible': False}, 'yaxis': {'visible': False}})
    feature_fig = go.Figure(layout={'title': default_feature_text, 'xaxis': {'visible': False}, 'yaxis': {'visible': False}})
    default_neighbor_text = "Select a node to analyze its neighborhood."
    neighbor_info_content = default_neighbor_text # Default value


    # --- Guard Clauses ---
    if graph_store_data is None or graph_store_data.get('num_nodes', 0) == 0:
        print("Update visualizations skipped: No graph data.")
        return empty_elements, empty_fig, feature_fig, default_info_text, attn_fig

    # --- Prepare data ---
    try:
        # Reconstruct PyG Data object (on CPU) for visualization logic
        # Important: Use the lists directly from the store
        current_data = Data(
            x=torch.tensor(graph_store_data['features'], dtype=torch.float),
            edge_index=torch.tensor(graph_store_data['edge_index'], dtype=torch.long),
            y=torch.tensor(graph_store_data['labels'], dtype=torch.long), # Include labels
            num_nodes=graph_store_data['num_nodes']
        )
    except Exception as e:
        print(f"Error reconstructing graph data for visualization: {e}")
        return empty_elements, empty_fig, feature_fig, "Error loading graph data.", attn_fig

    class_map = dataset_info.get('class_map', {}) if dataset_info else {}
    selected_node_id = selected_node_data.get('id') if selected_node_data else None

    # Check if model output is valid
    predictions = None
    embeddings = None
    valid_model_output = False
    if model_output and model_output.get('predictions') is not None and model_output.get('embeddings') is not None:
         # Convert back to tensors for potential processing, keep on CPU for plotting
         try:
             predictions_list = model_output['predictions']
             embeddings_list = model_output['embeddings']
             if len(predictions_list) == current_data.num_nodes and len(embeddings_list) == current_data.num_nodes:
                 predictions = torch.tensor(predictions_list)
                 embeddings = torch.tensor(embeddings_list)
                 valid_model_output = True
             else:
                 print("Warning: Model output size mismatch with current graph nodes. Ignoring model output.")
         except Exception as e:
             print(f"Error processing model output for visualization: {e}")

    # --- Node/Edge Selection & Neighbor Calculation ---
    selected_node_id_str = selected_node_data.get('id') if selected_node_data else None
    neighbor_ids = []
    node_true_class = None
    node_pred_class = None
    neighbor_true_classes = [] # Store neighbor true classes
    neighbor_pred_classes = [] # Store neighbor predicted classes
    attention_data_for_reasoning = None # Store attention data if GAT

    if selected_node_id_str is not None:
        try:
            selected_node_idx = int(selected_node_id_str)
            if 0 <= selected_node_idx < current_data.num_nodes:
                # Get selected node's labels
                has_true_labels = hasattr(current_data, 'y') and current_data.y is not None
                has_predictions = predictions is not None

                if has_true_labels and selected_node_idx < len(current_data.y):
                    node_true_class = int(current_data.y[selected_node_idx].item())
                if has_predictions and selected_node_idx < len(predictions):
                    node_pred_class = int(predictions[selected_node_idx].item())

                # --- Find neighbors and their labels ---
                edge_index = current_data.edge_index
                # ... (find neighbor_ids - same logic as before) ...
                mask_source = edge_index[0] == selected_node_idx
                mask_target = edge_index[1] == selected_node_idx
                neighbors_from_source = edge_index[1][mask_source].tolist()
                neighbors_from_target = edge_index[0][mask_target].tolist()
                all_neighbors = set(neighbors_from_source + neighbors_from_target)
                all_neighbors.discard(selected_node_idx)
                neighbor_ids = sorted(list(all_neighbors)) # Now neighbor_ids is populated


                # --- Generate Neighbor Info Content & Store Neighbor Labels ---
                neighbor_details = [] # For display
                if neighbor_ids:
                    for neighbor_idx in neighbor_ids:
                        if 0 <= neighbor_idx < current_data.num_nodes:
                            n_true_class, n_pred_class = None, None
                            true_label = "N/A"
                            if has_true_labels and neighbor_idx < len(current_data.y):
                                n_true_class = int(current_data.y[neighbor_idx].item())
                                true_label = class_map.get(n_true_class, f"C{n_true_class}") if n_true_class != -1 else "N/A"

                            pred_label = "N/A"
                            if has_predictions and neighbor_idx < len(predictions):
                                n_pred_class = int(predictions[neighbor_idx].item())
                                pred_label = class_map.get(n_pred_class, f"C{n_pred_class}")

                            neighbor_details.append(html.Li(f"Node {neighbor_idx}: True={true_label}, Pred={pred_label}"))
                            neighbor_true_classes.append(n_true_class)
                            neighbor_pred_classes.append(n_pred_class)
                        else:
                             neighbor_details.append(html.Li(f"Node {neighbor_idx}: Invalid Index!"))
                             neighbor_true_classes.append(None) # Keep lists aligned
                             neighbor_pred_classes.append(None)

                    if neighbor_details:
                         neighbor_info_content = [html.Strong(f"Neighbors of Node {selected_node_idx} ({len(neighbor_ids)}):"), html.Ul(neighbor_details)]
                    else:
                         neighbor_info_content = f"Node {selected_node_idx} error getting neighbor details."
                else:
                    neighbor_info_content = f"Node {selected_node_idx} has no neighbors."


                # --- Get Attention Data (if GAT) ---
                # (This part should ideally reuse the attention calculation done for the plot)
                # For simplicity here, we might recalculate or assume attn_fig holds the data if needed
                # Let's assume we can get it if model_type is GAT
                if model_type == 'GAT' and valid_model_output:
                     # Simplified: Ideally, extract data used for attn_fig
                     # Placeholder: Fetch attention data again if needed for reasoning
                     gat_model = load_model(model_type, dataset_name, device)
                     if gat_model:
                          att_data = get_attention_weights(gat_model, current_data, selected_node_idx, device)
                          if att_data:
                              attention_data_for_reasoning = att_data # Store (neighbor_indices, attention_scores) tuple

                # --- Generate Reasoning Sentence ---
                if node_true_class is not None and node_pred_class is not None:
                    reasoning_content = generate_reasoning_sentence(
                        selected_node_idx, node_true_class, node_pred_class,
                        neighbor_ids, neighbor_true_classes, neighbor_pred_classes,
                        class_map, model_type, attention_data_for_reasoning
                    )
                else:
                     reasoning_content = "Cannot generate reasoning without node's true and predicted class."


            else: # selected_node_idx out of bounds
                 neighbor_info_content = f"Selected node ID {selected_node_idx} is invalid."
                 selected_node_id_str = None
                 neighbor_ids = []

        except (ValueError, TypeError, IndexError) as e: # Catch errors during processing
            print(f"Error processing selected node or finding neighbors/labels: {e}")
            import traceback
            traceback.print_exc()
            neighbor_info_content = "Error analyzing neighborhood."
            reasoning_content = "Error generating reasoning due to processing error."
            selected_node_id_str = None
            neighbor_ids = []


    # --- Generate Graph View Elements ---
    print(f"Debug update_visualizations: Calling data_to_cytoscape with selected_node_id='{selected_node_id_str}', {len(neighbor_ids)} neighbors.") # DEBUG
    cyto_elements = data_to_cytoscape(current_data, predictions, class_map, selected_node_id_str, selected_edge_data, neighbor_ids=neighbor_ids)

    # --- Generate Embeddings Plot ---
    if valid_model_output:
        # Pass true labels (current_data.y) and selected node ID for highlighting
        embeddings_fig = plot_embeddings(embeddings, predictions, current_data.y, class_map, dim_reduction_method, selected_node_id)
    else:
        # Show embeddings plot even without predictions if data exists
        # Generate dummy embeddings if none exist from model yet
        if current_data.num_nodes > 1:
             print("Generating dummy embeddings for visualization as model output is invalid/missing.")
             dummy_embeddings = torch.randn(current_data.num_nodes, 16) # Example size
             embeddings_fig = plot_embeddings(dummy_embeddings, None, current_data.y, class_map, dim_reduction_method, selected_node_id)
             embeddings_fig.update_layout(title=f'Node Embeddings ({dim_reduction_method}) - Using Dummy Data')
        else:
             embeddings_fig = go.Figure(layout={'title': "Embeddings (Not Available)", 'xaxis': {'visible': False}, 'yaxis': {'visible': False}})


    # --- DEBUGGING PRINT for Final Element Data ---
    if selected_node_id_str:
        found = False
        for element in cyto_elements:
             # Check if 'data' exists and 'id' matches
             if isinstance(element.get('data'), dict) and element['data'].get('id') == selected_node_id_str:
                print(f"Debug update_visualizations: FINAL element data for selected node {selected_node_id_str}: {element.get('data')}") # DEBUG
                found = True
                break
        if not found:
             print(f"Debug update_visualizations: Selected node ID {selected_node_id_str} not found in FINAL elements list.") # DEBUG
    elif trigger == 'selected-node-store' and selected_node_data is None:
         print("Debug update_visualizations: Node selection cleared.") # DEBUG
    # --- END DEBUGGING PRINT ---

    if selected_node_id:
        try:
            node_idx = int(selected_node_id)
            if 0 <= node_idx < current_data.num_nodes:
                node_features = current_data.x[node_idx]
                # Call the updated plotting function
                feature_fig = plot_feature_histogram(node_features, selected_node_id)
            else:
                # Update placeholder if ID is invalid
                 feature_fig = go.Figure(layout={'title': f"Feature Histogram (Invalid Node ID: {selected_node_id})", 'xaxis': {'visible': False}, 'yaxis': {'visible': False},
                                                'annotations': [{'text': f"Invalid node ID.", 'xref': 'paper', 'yref': 'paper', 'showarrow': False}]})
        except (ValueError, TypeError, IndexError) as e:
             print(f"Error getting features for node {selected_node_id}: {e}")
             # Update placeholder on error
             feature_fig = go.Figure(layout={'title': f"Feature Histogram Error", 'xaxis': {'visible': False}, 'yaxis': {'visible': False},
                                             'annotations': [{'text': 'Error retrieving features.', 'xref': 'paper', 'yref': 'paper', 'showarrow': False}]})
    # else: feature_fig remains the default placeholder ("Select a single node...")


    # --- Generate Selected Info ---
    selected_info_content = []
    if selected_node_id:
        try:
            node_idx = int(selected_node_id)
            if 0 <= node_idx < current_data.num_nodes:
                selected_info_content.append(html.Strong(f"Selected Node: {node_idx}"))

                # Show prediction if valid
                if valid_model_output and node_idx < len(predictions):
                    pred_class = int(predictions[node_idx].item())
                    pred_label = class_map.get(pred_class, f'Class {pred_class}')
                    selected_info_content.append(html.P(f"Predicted Class: {pred_label} ({pred_class})"))
                else:
                     selected_info_content.append(html.P("Predicted Class: N/A"))

                # Display true label
                if hasattr(current_data, 'y') and current_data.y is not None and node_idx < len(current_data.y):
                    true_class = int(current_data.y[node_idx].item())
                    if true_class == -1: # Handle dummy label for added nodes
                        selected_info_content.append(html.P(f"True Class: N/A (New Node)"))
                    else:
                        true_label = class_map.get(true_class, f'Class {true_class}')
                        selected_info_content.append(html.P(f"True Class: {true_label} ({true_class})"))
                else:
                     selected_info_content.append(html.P("True Class: N/A"))
            else:
                 selected_info_content = f"Invalid Node ID selected: {selected_node_id}"

        except (ValueError, TypeError, IndexError) as e:
             selected_info_content = f"Error displaying info for node {selected_node_id}: {e}"

    elif selected_edge_data:
        source = selected_edge_data.get('source', 'N/A')
        target = selected_edge_data.get('target', 'N/A')
        selected_info_content.append(html.Strong(f"Selected Edge: {source} -> {target}"))
        # Could add info about connected nodes here if needed
    else:
        selected_info_content = default_info_text

    # --- Generate Attention View (if GAT and node selected) ---
    if model_type == 'GAT' and selected_node_id and valid_model_output:
        try:
            node_idx = int(selected_node_id)
            if 0 <= node_idx < current_data.num_nodes:
                # Reload the GAT model specifically for attention (ensure it's on the correct device)
                # Use the global device variable
                gat_model = load_model(model_type, dataset_name, device)
                if gat_model:
                    # Pass the reconstructed current_data and target device
                    att_data = get_attention_weights(gat_model, current_data, node_idx, device)
                    if att_data:
                        neighbor_indices, attention_scores = att_data
                        if len(neighbor_indices) > 0:
                            # Create bar chart
                            att_df = pd.DataFrame({'Neighbor Node': [str(n) for n in neighbor_indices], 'Attention Score': attention_scores})
                            att_df = att_df.sort_values('Attention Score', ascending=False).head(20) # Limit neighbors shown
                            attn_fig = px.bar(att_df, x='Neighbor Node', y='Attention Score',
                                              title=f'Attention Scores to Node {node_idx} (Top 20 Neighbors)',
                                              labels={'Attention Score': 'Attention'})
                            attn_fig.update_layout(xaxis_title="Source Neighbor Node", yaxis_title="Attention Score", title_font_size=14)
                        else:
                             attn_fig = go.Figure(layout={'title': f'Node {node_idx} has no incoming attention', 'xaxis': {'visible': False}, 'yaxis': {'visible': False}})
                    else:
                         # get_attention_weights returned None (e.g., error or no weights)
                         attn_fig = go.Figure(layout={'title': 'Could not retrieve attention weights.', 'xaxis': {'visible': False}, 'yaxis': {'visible': False}})
                else:
                     attn_fig = go.Figure(layout={'title': 'GAT Model could not be loaded for attention.', 'xaxis': {'visible': False}, 'yaxis': {'visible': False}})
            else:
                 attn_fig = go.Figure(layout={'title': f'Attention (Invalid Node ID: {selected_node_id})', 'xaxis': {'visible': False}, 'yaxis': {'visible': False}})

        except (ValueError, TypeError, IndexError) as e:
             attn_fig = go.Figure(layout={'title': f'Error getting attention: {e}', 'xaxis': {'visible': False}, 'yaxis': {'visible': False}})
             print(f"Error getting attention: {e}")
             import traceback
             traceback.print_exc()

    elif model_type != 'GAT':
         attn_fig = go.Figure(layout={'title': 'Attention view only available for GAT models.', 'xaxis': {'visible': False}, 'yaxis': {'visible': False}})
    # Else (GAT selected but no node selected, or model output invalid): keep the default message

    return cyto_elements, embeddings_fig, feature_fig, selected_info_content, attn_fig, neighbor_info_content, reasoning_content


# --- Run App ---
if __name__ == '__main__':
    print("--- Starting Interactive GNN Explainer ---")
    print(f"Using device: {device}")
    print(f"Models expected in: '{os.path.abspath(MODEL_DIR)}'")
    print(f"Datasets expected in: '{os.path.abspath(DATA_DIR)}'")
    print(f"IMPORTANT: Ensure pre-trained models (.pkl files) exist in the '{MODEL_DIR}' directory.")
    print(f"IMPORTANT: Ensure datasets (e.g., Cora, CiteSeer folders, jazz files) exist in the '{DATA_DIR}' directory.")
    print("If datasets are missing, PyG might attempt to download Planetoid datasets.")
    print("If models are missing, please run the training script first.")

    # Check if default model exists as a basic check
    script_dir = os.path.dirname(__file__) if '__file__' in locals() else '.'
    default_model_path = os.path.join(script_dir, MODEL_DIR, f"{DEFAULT_MODEL}_{DEFAULT_DATASET}.pkl")
    if not os.path.exists(default_model_path):
         print(f"\nWarning: Default model '{default_model_path}' not found.")

    print("\nDash app starting... Access at http://127.0.0.1:8050/ (or your configured host/port)")
    app.run(debug=True, host='0.0.0.0', port=8050) # Use debug=False for production

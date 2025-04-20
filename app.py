
import dash
from dash import dcc, html, Input, Output, State, callback, dash_table, ctx, no_update, ALL # Import ALL for pattern matching if needed later
import dash_cytoscape as cyto
import plotly.express as px
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
# FutureWarning can be noisy with pandas/numpy, suppress if needed
# warnings.simplefilter(action='ignore', category=FutureWarning)


# 2. Global Variables / Setup
MODEL_DIR = 'models' # Directory where trained models are saved
DEFAULT_DATASET = 'Cora'
DEFAULT_MODEL = 'GCN'
AVAILABLE_DATASETS = ['Cora', 'CiteSeer', 'PubMed', 'Jazz']
AVAILABLE_MODELS = ['GCN', 'GAT']
NODE_DIM_REDUCTION = 'TSNE' # 'PCA' or 'TSNE'

# Ensure model directory exists (important for loading)
# os.makedirs(MODEL_DIR, exist_ok=True) # Training script should create this

# --- Model Definitions (Must match the ones in the training script and Dash app) ---
class GCNNet(torch.nn.Module):
    """Basic GCN Network Definition."""
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        # Store args for saving/reconstruction
        self.init_args = {'hidden_channels': hidden_channels}

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # Note: Dropout is applied differently during eval vs train
        # For inference (eval mode), dropout is typically off.
        # If dropout was used during training, ensure model.eval() is called.
        # x = F.dropout(x, p=0.5, training=self.training)
        embeddings = x # Get embeddings from the hidden layer
        x = self.conv2(x, edge_index)
        # Return log_softmax for consistency with training loss and embeddings
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
        # Store args for saving/reconstruction
        self.init_args = {'hidden_channels': hidden_channels, 'heads': heads}

    def forward(self, data, return_attention_weights=False):
        x, edge_index = data.x, data.edge_index
        # Dropout layers are automatically handled by model.train() / model.eval()
        # x = F.dropout(x, p=0.6, training=self.training) # Input dropout handled by eval mode

        # Pass return_attention_weights to GATConv layers
        x, att1 = self.conv1(x, edge_index, return_attention_weights=return_attention_weights)
        x = F.elu(x)
        embeddings = x # Use embeddings after first layer + activation

        # x = F.dropout(x, p=0.6, training=self.training) # Dropout handled by eval mode
        x, att2 = self.conv2(x, edge_index, return_attention_weights=return_attention_weights)

        # Combine attention weights if needed
        attention_weights = (att1, att2) if return_attention_weights else None

        # Return format depends on whether attention is requested
        if return_attention_weights:
            # Return log_softmax for consistency, embeddings, and attention
            return F.log_softmax(x, dim=1), embeddings, attention_weights
        else:
            # Return log_softmax and embeddings
            return F.log_softmax(x, dim=1), embeddings


# --- Helper Functions ---

def load_dataset(name):
    """Loads a dataset, handling both Planetoid and custom Jazz dataset."""
    try:
        script_dir = os.path.dirname(__file__) if '__file__' in locals() else '.'
        data_dir = 'data'
        path = os.path.join(script_dir, data_dir, name)
        if name != 'Jazz':
            dataset = Planetoid(root=path, name=name)
            print(f"Dataset '{name}' loaded successfully.")
            return dataset
        else:
            return load_jazz_dataset(path)
    except Exception as e:
        print(f"Error loading dataset {name}: {e}")
        return None

def load_jazz_dataset(path):
    """Loads the Jazz dataset with Node2Vec embeddings and makes it compatible with Planetoid datasets."""
    try:
        edge_file = os.path.join(path, 'jazz.cites')
        feature_file = os.path.join(path, 'jazz.content')

        # Check if files exist
        if not os.path.exists(edge_file):
            print(f"Edge file not found: {edge_file}")
            return None
        if not os.path.exists(feature_file):
            print(f"Feature file not found: {feature_file}")
            return None

        # Load edges
        edges = np.loadtxt(edge_file, dtype=int, delimiter='\t')
        edge_index = torch.tensor(edges.T, dtype=torch.long)

        # Load content
        content = np.loadtxt(feature_file, dtype=float)
        node_ids = content[:, 0].astype(int)
        features = content[:, 1:-1]  # Features (e.g., degree), flexible for multiple columns
        labels = content[:, -1].astype(int)

        # Remap node IDs to consecutive integers
        node_map = {old_id: new_id for new_id, old_id in enumerate(sorted(node_ids))}
        edge_index = torch.tensor([[node_map[e[0]], node_map[e[1]]] for e in edges], dtype=torch.long).T

        # Generate Node2Vec embeddings
        G = nx.from_edgelist(edges)
        node2vec = Node2Vec(G, dimensions=16, walk_length=30, num_walks=200, workers=4, quiet=True)
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        embeddings = np.array([model.wv[str(node)] for node in sorted(G.nodes())])
        # Note: No need to realign embeddings with node_map here; they are already in the correct order
        # because sorted(G.nodes()) matches sorted(node_ids), and remapping ensures consistency

        # Concatenate base features with Node2Vec embeddings
        x = torch.tensor(np.hstack((features, embeddings)), dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.long)

        # Create train/val/test masks (60/20/20 split)
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

        # Create a dataset-like object with required attributes and indexing support
        dataset = type('JazzDataset', (), {
            'data': data,
            'num_node_features': x.size(1),
            'num_classes': len(torch.unique(y)),
            '__getitem__': lambda self, idx: self.data if idx == 0 else None,
            '__len__': lambda self: 1
        })()
        print("Dataset 'Jazz' loaded successfully.")
        return dataset
    except Exception as e:
        print(f"Error loading Jazz dataset: {e}")
        return None

def load_model(model_type, dataset_name):
    """Loads a pre-trained model from a .pkl file."""
    model_filename = f"{model_type}_{dataset_name}.pkl"
    # Determine path relative to the script file if possible, otherwise use current directory
    script_dir = os.path.dirname(__file__) if '__file__' in locals() else '.'
    model_path = os.path.join(script_dir, MODEL_DIR, model_filename)

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        print(f"Please run the training script (e.g., train_models.py) first to generate '{model_filename}'.")
        return None

    try:
        with open(model_path, 'rb') as f:
            # Load both the state dict and the initialization arguments
            model_state_dict, model_args = pickle.load(f)

        # Need dataset info to instantiate the model correctly
        dataset = load_dataset(dataset_name)
        if dataset is None: return None
        in_channels = dataset.num_node_features
        out_channels = dataset.num_classes

        # Recreate model instance using saved arguments
        if model_type == 'GCN':
            # Use saved args, provide defaults if missing (though training script should save them)
            hidden_channels = model_args.get('hidden_channels', 16)
            model = GCNNet(in_channels, hidden_channels, out_channels)
        elif model_type == 'GAT':
            hidden_channels = model_args.get('hidden_channels', 8)
            heads = model_args.get('heads', 8)
            model = GATNet(in_channels, hidden_channels, out_channels, heads=heads)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        model.load_state_dict(model_state_dict)
        model.eval() # Set model to evaluation mode *immediately* after loading
        print(f"Loaded model {model_filename} with args: {model_args}")
        return model
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None

@torch.no_grad() # Disable gradient calculations for inference
def run_inference(model, data, model_type):
    """Runs GNN inference and returns predictions and embeddings."""
    if model is None or data is None:
        return None, None

    model.eval() # Ensure model is in evaluation mode
    try:
        if model_type == 'GAT':
            # GAT forward method returns log_probs, embeddings (and optionally attention)
            log_probs, embeddings, *attention = model(data, return_attention_weights=False)
        else: # GCN
            log_probs, embeddings = model(data)

        predictions = log_probs.argmax(dim=1)
        return predictions, embeddings
    except Exception as e:
        print(f"Error during model inference: {e}")
        # This could happen if graph structure is incompatible after edits
        return None, None


@torch.no_grad() # Disable gradients for attention calculation too
def get_attention_weights(model, data, node_idx):
    """Extracts attention weights for a specific node (GAT only)."""
    if not isinstance(model, GATNet) or data is None or node_idx is None:
        return None

    model.eval() # Ensure model is in eval mode
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

        # Attention output is likely a tuple: (att_layer1, att_layer2)
        # Each element contains (edge_index, alpha_scores)
        # Let's visualize attention from the *last* GAT layer (att2)
        edge_index_att, alpha_att = attention_output[-1] # Get edge_index and scores from last GAT layer

        # Find edges pointing TO the selected node (target node = node_idx)
        mask = edge_index_att[1] == node_idx
        if not torch.any(mask):
            print(f"Node {node_idx} has no incoming edges in the attention mechanism.")
            return None # No attention scores to show if no incoming edges

        # Get the source nodes of these edges and their corresponding attention scores
        neighbor_indices = edge_index_att[0][mask]
        # alpha_att shape is typically [num_edges, num_heads] or [num_edges] if heads=1/concat=False
        # Average over heads if multiple heads exist in the final layer (should be 1 if concat=False)
        attention_scores = alpha_att[mask].mean(dim=1).squeeze().cpu().numpy()

        # Return source node indices and their attention scores towards the target node
        return neighbor_indices.cpu().numpy(), attention_scores

    except Exception as e:
        print(f"Error getting attention weights: {e}")
        # This might happen if graph structure changed drastically
        return None


def data_to_cytoscape(data, predictions=None, class_map=None, selected_node_id=None, selected_edge_data=None): # Changed selected_edge_id to selected_edge_data
    """Converts PyG Data object to Cytoscape elements list."""
    if data is None or data.num_nodes == 0:
        return [] # Return empty list if no data or no nodes

    nodes = []
    default_color = '#808080' # Grey
    selected_color = '#FFA500' # Orange
    # Define a color palette for classes
    color_palette = px.colors.qualitative.Plotly

    # Ensure predictions is a numpy array or list of the correct length
    preds_list = None
    if predictions is not None:
        if isinstance(predictions, torch.Tensor):
            preds_list = predictions.cpu().numpy()
        elif isinstance(predictions, (list, np.ndarray)):
            preds_list = np.array(predictions) # Ensure numpy array for consistent indexing
        # Ensure length matches num_nodes, otherwise ignore predictions
        if len(preds_list) != data.num_nodes:
            print(f"Warning: Predictions length ({len(preds_list)}) doesn't match num_nodes ({data.num_nodes}). Ignoring predictions for coloring.")
            preds_list = None


    for i in range(data.num_nodes): # Iterate up to num_nodes
        node_data = {'id': str(i), 'label': f'Node {i}'}
        color = default_color
        size = 15 # Default size

        if preds_list is not None:
            pred_class = preds_list[i]
            node_data['class'] = int(pred_class) # Store class as int
            # Assign color based on predicted class
            color = color_palette[int(pred_class) % len(color_palette)]
            if class_map and int(pred_class) in class_map:
                node_data['label'] += f'\n({class_map[int(pred_class)]})' # Add class label if map provided

        if selected_node_id is not None and str(i) == selected_node_id:
            color = selected_color # Highlight selected node
            size = 25 # Make selected node larger

        nodes.append({'data': node_data, 'style': {'background-color': color, 'width': size, 'height': size}})

    edges = []
    default_edge_color = '#cccccc'
    selected_edge_color = '#FFA500' # Orange

    if data.edge_index is not None and data.edge_index.shape[1] > 0:
        edge_index_display = data.edge_index.cpu().numpy()
        sel_source = selected_edge_data.get('source') if selected_edge_data else None
        sel_target = selected_edge_data.get('target') if selected_edge_data else None

        for i in range(edge_index_display.shape[1]):
            source = str(edge_index_display[0, i])
            target = str(edge_index_display[1, i])
            # Ensure source and target nodes exist (can happen after node removal)
            if int(source) >= data.num_nodes or int(target) >= data.num_nodes:
                continue # Skip edges connected to non-existent nodes

            edge_id = f"{source}_{target}_{i}" # Unique edge ID for Cytoscape
            edge_data = {'id': edge_id, 'source': source, 'target': target}
            edge_color = default_edge_color
            edge_width = 1

            # Check if this edge (or its reverse) matches the selected edge source/target
            if sel_source is not None and sel_target is not None:
                 if sel_source == source and sel_target == target:
                      edge_color = selected_edge_color
                      edge_width = 3
                 # Consider highlighting reverse edge too if graph is treated as undirected visually
                 elif sel_source == target and sel_target == source:
                      edge_color = selected_edge_color
                      edge_width = 3

            edges.append({'data': edge_data, 'style': {'line-color': edge_color, 'width': edge_width}})

    return nodes + edges # Combine nodes and edges for Cytoscape elements


def plot_embeddings(embeddings, predictions, class_map=None, dim_reduction='TSNE'):
    """Generates Plotly scatter plot for node embeddings."""
    if embeddings is None or predictions is None:
        return px.scatter(title="Embeddings (Not Available)")

    # Convert to numpy if they are tensors
    if isinstance(embeddings, torch.Tensor):
        embeddings_np = embeddings.cpu().numpy()
    else:
        embeddings_np = np.array(embeddings) # Assume list of lists or numpy array

    if isinstance(predictions, torch.Tensor):
        predictions_np = predictions.cpu().numpy()
    else:
        predictions_np = np.array(predictions) # Assume list or numpy array

    num_nodes = embeddings_np.shape[0]
    if num_nodes == 0 or embeddings_np.shape[0] != predictions_np.shape[0]:
        print("Warning: Embeddings and predictions size mismatch or zero nodes.")
        return px.scatter(title="Embeddings (Invalid Data)")

    if num_nodes < 2:
         return px.scatter(title="Embeddings (Need >= 2 nodes)")

    # Dimensionality Reduction
    n_components = 2
    if num_nodes < n_components + 1 and dim_reduction == 'TSNE':
        print("Not enough samples for t-SNE, switching to PCA")
        dim_reduction = 'PCA' # Fallback if TSNE requirements not met

    if dim_reduction == 'TSNE':
        # Adjust perplexity based on number of samples
        perplexity = min(30.0, max(5.0, float(num_nodes - 1) / 3.0)) # Heuristic
        try:
            # Try newer sklearn syntax first
            reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto', n_iter=300) # Shorter iter for speed
        except TypeError: # Fallback for older sklearn versions
             print("Using older TSNE initialization (no learning_rate='auto').")
             reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, init='pca', n_iter=300)
    else: # PCA
        reducer = PCA(n_components=n_components)

    try:
        embeddings_2d = reducer.fit_transform(embeddings_np)
    except Exception as e:
        print(f"Error during dimensionality reduction ({dim_reduction}): {e}")
        # Create dummy data if reduction fails
        embeddings_2d = np.random.rand(num_nodes, n_components)


    # Create DataFrame for Plotly
    df = pd.DataFrame(embeddings_2d, columns=['Dim 1', 'Dim 2'])
    df['Node ID'] = [str(i) for i in range(num_nodes)]

    # Map predictions to labels if class_map is available
    if class_map:
        df['Predicted Class'] = [class_map.get(int(p), f'Class {int(p)}') for p in predictions_np]
    else:
        df['Predicted Class'] = [f'Class {int(p)}' for p in predictions_np]

    # Generate plot
    try:
        fig = px.scatter(df, x='Dim 1', y='Dim 2', color='Predicted Class',
                         hover_data=['Node ID', 'Predicted Class'], title=f'Node Embeddings ({dim_reduction})',
                         color_discrete_sequence=px.colors.qualitative.Plotly) # Use consistent colors
        fig.update_layout(legend_title_text='Predicted Class')
        fig.update_traces(marker=dict(size=8, opacity=0.8))
    except Exception as e:
        print(f"Error creating embeddings plot: {e}")
        fig = px.scatter(title=f"Embeddings Plot Error ({dim_reduction})")


    return fig

def create_feature_table(features, node_indices=None):
    """Creates data and columns for the Dash DataTable."""
    if features is None:
        return [], []

    # Convert to numpy if tensor
    if isinstance(features, torch.Tensor):
        features_np = features.cpu().numpy()
    elif isinstance(features, list):
         # Handle potential empty list after node removal
         if not features:
             return [], []
         try:
             features_np = np.array(features)
         except ValueError as e: # Handle case where sublists might have different lengths after error
             print(f"Warning: Could not convert features list to numpy array: {e}")
             return [], []

    else: # Assume numpy already
        features_np = features

    if features_np.ndim == 1: # Handle case of single feature vector
        features_np = features_np.reshape(1, -1)
    elif features_np.ndim == 0 or features_np.size == 0: # Handle empty features
         return [], []


    num_total_nodes, num_features = features_np.shape

    # If specific nodes are selected, filter features
    if node_indices is not None and len(node_indices) > 0:
        # Ensure indices are valid integers
        valid_indices = [idx for idx in node_indices if isinstance(idx, int) and 0 <= idx < num_total_nodes]
        if not valid_indices:
             print("Warning: No valid node indices provided for feature table.")
             # Show all features if selected node is invalid? Or empty table? Let's show empty.
             return [], []
        features_to_display = features_np[valid_indices, :]
        index_labels = [f'Node {idx}' for idx in valid_indices]
    else:
        # Show all nodes if none are specifically selected
        features_to_display = features_np
        index_labels = [f'Node {i}' for i in range(num_total_nodes)]

    # Handle case where features_to_display becomes empty after filtering
    if features_to_display.size == 0:
        return [], []

    # Create DataFrame
    df = pd.DataFrame(features_to_display, index=index_labels)
    # Ensure consistent number of columns even if empty
    df.columns = [f'F_{i}' for i in range(num_features)] # Shorter names for table
    df.reset_index(inplace=True) # Make index a column called 'index'
    df.rename(columns={'index': 'Node'}, inplace=True)

    columns = [{"name": i, "id": i} for i in df.columns]
    data = df.to_dict('records')
    return columns, data

# --- Dash App Initialization ---
app = dash.Dash(__name__, suppress_callback_exceptions=True,
                external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'], prevent_initial_callbacks='initial_duplicate') # Basic CSS
server = app.server # Expose server for deployment

# --- App Layout ---
app.layout = html.Div([
    # --- Stores ---
    # Store for the main graph data (nodes, edges, features) - allows modification
    # Storing components as lists/basic types for JSON serialization
    dcc.Store(id='current-graph-store'), # Stores {'edge_index': [[...],[...]], 'features': [[...],...], 'labels': [...], 'num_nodes': int}
    # Store for model outputs (predictions, embeddings)
    dcc.Store(id='current-model-output-store'), # Stores {'predictions': [...], 'embeddings': [[...],...]}
    # Store for selected node/edge IDs from Cytoscape
    dcc.Store(id='selected-node-store'), # Stores {'id': str}
    dcc.Store(id='selected-edge-store'), # Stores {'id': str, 'source': str, 'target': str}
    # Store dataset info (like class map)
    dcc.Store(id='dataset-info-store'), # Stores {'class_map': dict}

    # --- Header ---
    html.H1("Interactive GNN Explainer", style={'textAlign': 'center', 'marginBottom': '20px'}),

    # --- Control Panel ---
    html.Div([
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

        html.Div([
            html.Button('Add Node', id='add-node-button', n_clicks=0, style={'marginRight': '5px'}),
            html.Button('Remove Selected Node', id='remove-node-button', n_clicks=0, style={'marginRight': '20px'}),

            dcc.Input(id='new-edge-source', type='text', placeholder='Source Node ID', style={'width': '120px', 'marginRight': '5px'}),
            dcc.Input(id='new-edge-target', type='text', placeholder='Target Node ID', style={'width': '120px', 'marginRight': '5px'}),
            html.Button('Add Edge', id='add-edge-button', n_clicks=0, style={'marginRight': '5px'}),
            html.Button('Remove Selected Edge', id='remove-edge-button', n_clicks=0),
        ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '10px'}) # Added margin top

    ], id='control-panel', style={'padding': '15px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'marginBottom': '20px', 'backgroundColor': '#f9f9f9'}),

    # --- Status Message Area ---
    html.Div(id='status-message-area', style={'padding': '10px', 'marginBottom': '10px', 'border': '1px dashed #ccc', 'borderRadius': '5px', 'minHeight': '40px', 'backgroundColor': '#f0f0f0'}),


    # --- Main Visualization Area (Split View) ---
    html.Div([
        # --- Left Column ---
        html.Div([
            html.H3("Graph View"),
            cyto.Cytoscape(
                id='graph-view',
                layout={'name': 'cose', 'idealEdgeLength': 100, 'nodeOverlap': 20, 'refresh': 20, 'fit': True, 'padding': 30, 'randomize': False, 'componentSpacing': 100, 'nodeRepulsion': 400000, 'edgeElasticity': 100, 'nestingFactor': 5, 'gravity': 80, 'numIter': 1000, 'initialTemp': 200, 'coolingFactor': 0.95, 'minTemp': 1.0},
                style={'width': '100%', 'height': '450px', 'border': '1px solid #ddd', 'borderRadius': '5px'},
                stylesheet=[ # Default stylesheet
                    {'selector': 'node', 'style': {'label': 'data(label)', 'font-size': '10px', 'text-valign': 'center', 'text-halign': 'center', 'background-color': '#808080', 'width': 15, 'height': 15, 'color': '#fff', 'text-outline-width': 2, 'text-outline-color': '#888'}}, # Added text outline for better readability
                    {'selector': 'edge', 'style': {'curve-style': 'bezier', 'target-arrow-shape': 'triangle', 'line-color': '#ccc', 'target-arrow-color': '#ccc', 'opacity': 0.7, 'width': 1.5}} # Slightly thicker edges
                ]
            ),
            html.H3("Node Embeddings"),
            dcc.Dropdown(
                id='dim-reduction-dropdown',
                options=[{'label': 't-SNE', 'value': 'TSNE'}, {'label': 'PCA', 'value': 'PCA'}],
                value=NODE_DIM_REDUCTION,
                clearable=False,
                style={'width': '100px', 'marginBottom': '10px'}
            ),
            dcc.Graph(id='embeddings-view', style={'height': '400px'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '2%'}),

        # --- Right Column ---
        html.Div([
            html.H3("Feature Matrix (Selected Node or All)"),
            dash_table.DataTable(
                id='feature-matrix-view',
                page_size=8, # Adjust page size
                style_table={'overflowX': 'auto', 'height': '250px', 'overflowY': 'auto', 'border': '1px solid #ddd'},
                style_cell={'minWidth': '40px', 'width': '60px', 'maxWidth': '80px', 'textAlign': 'center', 'padding': '5px'},
                style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
                style_data={'border': '1px solid #eee'},
                filter_action="native",
                sort_action="native",
            ),
            html.H3("Selected Node/Edge Info"),
            html.Div(id='selected-info-view', style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '10px', 'minHeight': '100px', 'marginBottom': '10px', 'backgroundColor': '#f9f9f9'}),
            html.H3("Attention Weights (GAT Only)"),
            dcc.Graph(id='attention-view', style={'height': '250px'}) # Placeholder for attention vis
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ])
], style={'padding': '20px'})


# --- Callbacks ---

# Callback to load dataset and initialize graph store
# Runs on startup because prevent_initial_call=False
@callback(
    Output('current-graph-store', 'data', allow_duplicate=True),  # Allow duplicate as other callbacks target this
    Output('dataset-info-store', 'data'),  # Only output from here
    Output('status-message-area', 'children', allow_duplicate=True),  # Allow duplicate as other callbacks target this
    Input('dataset-dropdown', 'value'),
    prevent_initial_call='initial_duplicate'  # Runs on startup
)
def load_dataset_callback(dataset_name):
    """Loads dataset when dropdown changes or on startup, stores its basic structure."""
    # Use ctx.triggered_id to check if it's the initial call or a dropdown change
    trigger_id = ctx.triggered_id
    is_initial_call = trigger_id is None  # If None, it's likely the initial call

    print(f"Loading dataset: {dataset_name} (Trigger: {trigger_id if trigger_id else 'Initial Load'})")
    status_message = f"Loading dataset '{dataset_name}'..."
    dataset = load_dataset(dataset_name)
    
    if dataset is None:
        error_message = f"Error: Failed to load dataset '{dataset_name}'. Check console and data directory."
        empty_graph_store = {
            'edge_index': [[], []],
            'features': [],
            'labels': [],
            'num_nodes': 0,
            'train_mask': [],
            'val_mask': [],
            'test_mask': []
        }
        return empty_graph_store, {}, error_message

    # Get the graph object (now works for both Jazz and Planetoid datasets)
    data = dataset[0]

    # Validate that data has required attributes
    required_attrs = ['edge_index', 'x', 'y', 'num_nodes', 'train_mask', 'val_mask', 'test_mask']
    missing_attrs = [attr for attr in required_attrs if not hasattr(data, attr)]
    if missing_attrs:
        error_message = f"Error: Dataset '{dataset_name}' missing attributes: {missing_attrs}"
        empty_graph_store = {
            'edge_index': [[], []],
            'features': [],
            'labels': [],
            'num_nodes': 0,
            'train_mask': [],
            'val_mask': [],
            'test_mask': []
        }
        return empty_graph_store, {}, error_message

    # Prepare data for storage (convert tensors to lists/serializable types)
    try:
        graph_store_data = {
            'edge_index': data.edge_index.cpu().tolist(),
            'features': data.x.cpu().tolist(),  # Store features as list
            'labels': data.y.cpu().tolist(),    # Store original labels
            'num_nodes': data.num_nodes,
            'train_mask': data.train_mask.cpu().tolist(),
            'val_mask': data.val_mask.cpu().tolist(),
            'test_mask': data.test_mask.cpu().tolist(),
        }
        # Update dataset info store with basic metadata
        dataset_info = {
            'num_node_features': dataset.num_node_features,
            'num_classes': dataset.num_classes
        }
        status_message = f"Dataset '{dataset_name}' loaded successfully."
        return graph_store_data, dataset_info, status_message
    except Exception as e:
        error_message = f"Error processing data from '{dataset_name}': {e}"
        empty_graph_store = {
            'edge_index': [[], []],
            'features': [],
            'labels': [],
            'num_nodes': 0,
            'train_mask': [],
            'val_mask': [],
            'test_mask': []
        }
        return empty_graph_store, {}, error_message


    # Store class map if available
    class_map = None
    # Standard PyG way for Planetoid:
    if hasattr(dataset, 'num_classes'):
         # Create a generic map if names aren't stored explicitly
         class_map = {i: f'Class {i}' for i in range(dataset.num_classes)}
         # Try to get actual names (less reliable across PyG versions)
         try:
              if dataset_name == 'Cora' and dataset.num_classes == 7:
                   class_map = {0: 'Theory', 1: 'Reinforcement Learning', 2: 'Genetic Algorithms', 3: 'Neural Networks', 4: 'Probabilistic Methods', 5: 'Case Based', 6: 'Rule Learning'}
              elif dataset_name == 'CiteSeer' and dataset.num_classes == 6:
                   class_map = {0: 'Agents', 1: 'AI', 2: 'DB', 3: 'IR', 4: 'ML', 5: 'HCI'}
              # Add PubMed map if known and needed
         except Exception:
              pass # Stick with generic map if specific names fail


    dataset_info = {'class_map': class_map}

    status_message = f"Dataset '{dataset_name}' loaded. Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1]}."
    print(status_message)
    # This update triggers the run_inference_callback
    return graph_store_data, dataset_info, status_message


# Callback to run inference when graph data changes (loading or editing)
@callback(
    Output('current-model-output-store', 'data'),
    Output('status-message-area', 'children', allow_duplicate=True), # Allow duplicate
    Input('current-graph-store', 'data'), # Triggered when dataset/graph changes
    State('model-dropdown', 'value'),     # Get current model selection
    State('dataset-dropdown', 'value'),   # Need dataset name for context
    prevent_initial_call=True # IMPORTANT: Prevent this from running until graph_store is populated by the initial load
)
def run_inference_callback(graph_store_data, model_type, dataset_name):
    """Runs inference when the graph data changes."""
    triggered_input = ctx.triggered_id
    print(f"Running inference. Trigger: {triggered_input}")
    status_message = f"Running {model_type} inference on {dataset_name}..."

    # Add checks for empty graph data which might happen on initial load failure
    if not graph_store_data or graph_store_data.get('num_nodes', 0) == 0:
        print("Inference skipped: Graph data is empty or invalid.")
        return {'predictions': None, 'embeddings': None}, "Inference skipped: No graph data."
    if not model_type or not dataset_name:
        print("Inference skipped: Missing model type or dataset name.")
        return no_update, "Inference skipped: Missing model/dataset selection."


    # Reconstruct PyG Data object from stored components
    try:
        # Ensure tensors are created on CPU initially
        device = 'cpu' # Inference usually done on CPU unless GPU is explicitly managed
        current_data = Data(
            x=torch.tensor(graph_store_data['features'], dtype=torch.float, device=device),
            edge_index=torch.tensor(graph_store_data['edge_index'], dtype=torch.long, device=device),
            y=torch.tensor(graph_store_data['labels'], dtype=torch.long, device=device) # Include labels if stored
        )
        # Important: Set num_nodes explicitly, as it might change during edits
        current_data.num_nodes = graph_store_data['num_nodes']
        # Add masks back if needed by model/evaluation (not strictly needed for basic inference)
        current_data.train_mask = torch.tensor(graph_store_data['train_mask'], dtype=torch.bool, device=device)
        current_data.val_mask = torch.tensor(graph_store_data['val_mask'], dtype=torch.bool, device=device)
        current_data.test_mask = torch.tensor(graph_store_data['test_mask'], dtype=torch.bool, device=device)

    except Exception as e:
        error_message = f"Error reconstructing graph data for inference: {e}"
        print(error_message)
        # Return empty results if reconstruction fails
        return {'predictions': None, 'embeddings': None}, error_message

    # Load the selected model
    model = load_model(model_type, dataset_name)
    if model is None:
        error_message = f"Inference skipped: {model_type} model for {dataset_name} could not be loaded."
        print(error_message)
        return {'predictions': None, 'embeddings': None}, error_message # Return empty output

    # Run inference
    predictions, embeddings = run_inference(model, current_data, model_type)

    if predictions is None or embeddings is None:
        error_message = f"Inference failed for {model_type} on {dataset_name}. Check graph structure/model compatibility."
        print(error_message)
        # Return empty output store data on failure
        return {'predictions': None, 'embeddings': None}, error_message

    print(f"Inference complete. Predictions shape: {predictions.shape}, Embeddings shape: {embeddings.shape}")
    status_message = f"Inference complete for {model_type} on {dataset_name}."

    # Store results (convert back to lists for JSON serialization)
    output_data = {
        'predictions': predictions.cpu().tolist(),
        'embeddings': embeddings.cpu().tolist()
    }
    return output_data, status_message


# --- Callbacks for Graph Editing ---

@callback(
    Output('current-graph-store', 'data', allow_duplicate=True), # Allow duplicate
    Output('status-message-area', 'children', allow_duplicate=True), # Allow duplicate
    Input('add-node-button', 'n_clicks'),
    State('current-graph-store', 'data'),
    prevent_initial_call=True
)
def add_node_callback(n_clicks, graph_store_data):
    """Adds a new node with average features."""
    if n_clicks == 0 or not graph_store_data:
        return no_update, no_update

    print("Add Node button clicked.")
    status_message = "Adding node..."

    try:
        features_list = graph_store_data['features']
        num_nodes = graph_store_data['num_nodes']
        new_node_id = num_nodes # ID of the new node

        # Calculate average features
        if features_list: # Check if features_list is not empty
            features_tensor = torch.tensor(features_list, dtype=torch.float)
            if features_tensor.numel() > 0 and features_tensor.ndim == 2: # Ensure it's a 2D tensor
                 new_features = features_tensor.mean(dim=0, keepdim=True)
            elif features_tensor.ndim == 1: # Handle case if only one node existed before
                 new_features = features_tensor.unsqueeze(0) # Keep original features
                 print("Warning: Only one node existed, using its features for the new node.")
            else: # Handle case where list exists but is empty or invalid shape
                 status_message = "Cannot add node: Feature list is empty or has invalid shape."
                 print(status_message)
                 return no_update, status_message
        else:
            # Cannot determine feature dimension if no nodes exist
            status_message = "Cannot add node: No existing nodes to determine feature dimension."
            print(status_message)
            return no_update, status_message

        # Update graph_store_data (use deepcopy to avoid modifying the original state directly before returning)
        new_graph_data = copy.deepcopy(graph_store_data)
        new_graph_data['num_nodes'] += 1
        new_graph_data['features'].append(new_features.squeeze().tolist())
        # Add dummy label and mask values for the new node
        new_graph_data['labels'].append(-1) # Assign a dummy label (e.g., -1)
        new_graph_data['train_mask'].append(False)
        new_graph_data['val_mask'].append(False)
        new_graph_data['test_mask'].append(False)


        status_message = f"Added Node {new_node_id}. Total nodes: {new_graph_data['num_nodes']}"
        print(status_message)
        # Return updated graph_store_data to trigger re-inference
        return new_graph_data, status_message

    except Exception as e:
        error_message = f"Error adding node: {e}"
        print(error_message)
        return no_update, error_message


@callback(
    Output('current-graph-store', 'data', allow_duplicate=True), # Allow duplicate
    Output('status-message-area', 'children', allow_duplicate=True), # Allow duplicate
    Input('remove-node-button', 'n_clicks'),
    State('selected-node-store', 'data'), # Get the ID of the node to remove
    State('current-graph-store', 'data'),
    prevent_initial_call=True
)
def remove_node_callback(n_clicks, selected_node_data, graph_store_data):
    """Removes the selected node and its incident edges."""
    if n_clicks == 0 or not graph_store_data:
        return no_update, no_update
    if not selected_node_data or 'id' not in selected_node_data:
        return no_update, "Select a node to remove."

    try:
        node_id_to_remove = int(selected_node_data['id'])
    except (ValueError, TypeError):
         return no_update, "Invalid selected node ID."

    print(f"Attempting to remove node: {node_id_to_remove}")
    status_message = f"Removing node {node_id_to_remove}..."

    try:
        num_nodes = graph_store_data['num_nodes']
        if not (0 <= node_id_to_remove < num_nodes):
            status_message = f"Invalid node ID to remove: {node_id_to_remove}"
            print(status_message)
            return no_update, status_message
        if num_nodes <= 1:
             status_message = "Cannot remove the last node."
             print(status_message)
             return no_update, status_message


        # Use deepcopy to work on a new version
        new_graph_data = copy.deepcopy(graph_store_data)

        # --- Update Node-related Lists ---
        # Create mask for nodes to keep
        node_mask = [i != node_id_to_remove for i in range(num_nodes)]

        # Filter features, labels, and masks
        new_graph_data['features'] = [feat for i, feat in enumerate(new_graph_data['features']) if node_mask[i]]
        new_graph_data['labels'] = [lab for i, lab in enumerate(new_graph_data['labels']) if node_mask[i]]
        new_graph_data['train_mask'] = [m for i, m in enumerate(new_graph_data['train_mask']) if node_mask[i]]
        new_graph_data['val_mask'] = [m for i, m in enumerate(new_graph_data['val_mask']) if node_mask[i]]
        new_graph_data['test_mask'] = [m for i, m in enumerate(new_graph_data['test_mask']) if node_mask[i]]

        # --- Update Edge Index ---
        edge_index = torch.tensor(new_graph_data['edge_index'], dtype=torch.long)

        # Handle empty edge_index case
        if edge_index.numel() == 0:
            remapped_edge_index = edge_index # Keep it empty
        else:
            # Create mask for edges to keep (both source and target must not be the removed node)
            edge_mask = (edge_index[0] != node_id_to_remove) & (edge_index[1] != node_id_to_remove)
            new_edge_index = edge_index[:, edge_mask]

            # --- Remap Edge Indices ---
            # Create a mapping from old indices to new indices
            node_mapping = torch.full((num_nodes,), -1, dtype=torch.long)
            # Calculate new indices for nodes that are kept
            kept_node_indices = torch.arange(num_nodes)[torch.tensor(node_mask)]
            new_indices = torch.arange(len(kept_node_indices))
            node_mapping[kept_node_indices] = new_indices

            # Apply mapping to the filtered edge index
            # Important check: ensure new_edge_index is not empty before mapping
            if new_edge_index.numel() > 0:
                 remapped_edge_index = node_mapping[new_edge_index]
                 # Filter out any edges that might map to -1 (shouldn't happen with correct logic, but safety check)
                 valid_edge_mask = torch.all(remapped_edge_index != -1, dim=0)
                 remapped_edge_index = remapped_edge_index[:, valid_edge_mask]
            else:
                 remapped_edge_index = new_edge_index # Keep it empty if no edges remained


        # Update graph data store
        new_graph_data['num_nodes'] = len(new_graph_data['features']) # New number of nodes
        new_graph_data['edge_index'] = remapped_edge_index.tolist()

        status_message = f"Removed Node {node_id_to_remove}. New node count: {new_graph_data['num_nodes']}"
        print(status_message)
        # Return updated data to trigger re-inference
        return new_graph_data, status_message

    except Exception as e:
        error_message = f"Error removing node {node_id_to_remove}: {e}"
        print(error_message)
        return no_update, error_message


@callback(
    Output('current-graph-store', 'data', allow_duplicate=True), # Allow duplicate
    Output('status-message-area', 'children', allow_duplicate=True), # Allow duplicate
    Input('add-edge-button', 'n_clicks'),
    State('new-edge-source', 'value'),
    State('new-edge-target', 'value'),
    State('current-graph-store', 'data'),
    prevent_initial_call=True
)
def add_edge_callback(n_clicks, source_str, target_str, graph_store_data):
    """Adds a new edge (and its reverse) between specified nodes."""
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
        # Validate node IDs
        if not (0 <= source_id < num_nodes and 0 <= target_id < num_nodes):
            status_message = f"Invalid node ID(s). Must be between 0 and {num_nodes-1}."
            print(status_message)
            return no_update, status_message
        if source_id == target_id:
            return no_update, "Self-loops are not added via this button." # GNN layers might handle them internally

        # Use deepcopy to work on a new version
        new_graph_data = copy.deepcopy(graph_store_data)
        edge_index_list = new_graph_data['edge_index']

        # Check if edge already exists (check both directions)
        exists1 = False
        exists2 = False
        # Handle empty edge index list
        if edge_index_list and edge_index_list[0]:
            for i in range(len(edge_index_list[0])):
                if edge_index_list[0][i] == source_id and edge_index_list[1][i] == target_id:
                    exists1 = True
                if edge_index_list[0][i] == target_id and edge_index_list[1][i] == source_id:
                    exists2 = True
                if exists1 and exists2: # Found both directions
                    break

        if exists1 and exists2:
            status_message = f"Edge between {source_id} and {target_id} already exists."
            print(status_message)
            return no_update, status_message

        # Ensure edge_index_list has the correct structure [[], []] if empty
        if not edge_index_list:
            edge_index_list = [[], []]

        # Append new edge (source, target) and its reverse
        if not exists1:
            edge_index_list[0].append(source_id)
            edge_index_list[1].append(target_id)
        if not exists2:
            edge_index_list[0].append(target_id)
            edge_index_list[1].append(source_id)

        new_graph_data['edge_index'] = edge_index_list
        status_message = f"Added edge between {source_id} and {target_id}."
        print(status_message)
        # Return updated data to trigger re-inference
        return new_graph_data, status_message

    except Exception as e:
        error_message = f"Error adding edge ({source_id} -> {target_id}): {e}"
        print(error_message)
        return no_update, error_message


@callback(
    Output('current-graph-store', 'data', allow_duplicate=True), # Allow duplicate
    Output('status-message-area', 'children', allow_duplicate=True), # Allow duplicate
    Input('remove-edge-button', 'n_clicks'),
    State('selected-edge-store', 'data'), # Get the ID/data of the edge to remove
    State('current-graph-store', 'data'),
    prevent_initial_call=True
)
def remove_edge_callback(n_clicks, selected_edge_data, graph_store_data):
    """Removes the selected edge (and its reverse)."""
    if n_clicks == 0 or not graph_store_data:
        return no_update, no_update
    if not selected_edge_data or 'source' not in selected_edge_data or 'target' not in selected_edge_data:
        return no_update, "Select an edge to remove."

    try:
        # Get source and target from the selected edge data (provided by Cytoscape)
        source_id = int(selected_edge_data['source'])
        target_id = int(selected_edge_data['target'])
    except (ValueError, TypeError, KeyError):
        return no_update, "Invalid selected edge data."

    print(f"Attempting to remove edge between {source_id} and {target_id}")
    status_message = f"Removing edge {source_id} <-> {target_id}..."

    try:
        # Use deepcopy to work on a new version
        new_graph_data = copy.deepcopy(graph_store_data)
        edge_index_list = new_graph_data['edge_index']

        # Handle empty edge case
        if not edge_index_list or not edge_index_list[0]:
             status_message = f"Edge between {source_id} and {target_id} not found (no edges)."
             print(status_message)
             return no_update, status_message

        edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long)

        # Create mask to identify the edge(s) to remove (both directions)
        mask_forward = (edge_index_tensor[0] == source_id) & (edge_index_tensor[1] == target_id)
        mask_backward = (edge_index_tensor[0] == target_id) & (edge_index_tensor[1] == source_id)
        remove_mask = mask_forward | mask_backward

        if not torch.any(remove_mask):
            status_message = f"Edge between {source_id} and {target_id} not found."
            print(status_message)
            return no_update, status_message

        # Keep edges that are NOT the one(s) to be removed
        keep_mask = ~remove_mask
        new_edge_index = edge_index_tensor[:, keep_mask]

        # Update graph_store_data
        new_graph_data['edge_index'] = new_edge_index.tolist()
        status_message = f"Removed edge between {source_id} and {target_id}. New edge count: {new_edge_index.shape[1]}"
        print(status_message)
        # Return updated data to trigger re-inference
        return new_graph_data, status_message

    except Exception as e:
        error_message = f"Error removing edge ({source_id} <-> {target_id}): {e}"
        print(error_message)
        return no_update, error_message


# --- Callbacks for Storing Selections ---

@callback(
    Output('selected-node-store', 'data'),
    Output('selected-edge-store', 'data', allow_duplicate=True), # Clear edge selection when node is clicked
    Input('graph-view', 'tapNodeData'),
    prevent_initial_call=True
)
def store_selected_node(tapNodeData):
    if tapNodeData and 'id' in tapNodeData:
        print(f"Node selected: {tapNodeData['id']}")
        return {'id': tapNodeData['id']}, None # Store node ID, clear edge store
    # If tap is not on a node, don't change anything
    return no_update, no_update

@callback(
    Output('selected-edge-store', 'data'),
    Output('selected-node-store', 'data', allow_duplicate=True), # Clear node selection when edge is clicked
    Input('graph-view', 'tapEdgeData'),
    prevent_initial_call=True
)
def store_selected_edge(tapEdgeData):
    if tapEdgeData and 'source' in tapEdgeData and 'target' in tapEdgeData:
        print(f"Edge selected: {tapEdgeData.get('id', 'N/A')} (Source: {tapEdgeData['source']}, Target: {tapEdgeData['target']})")
        # Store source and target, maybe the unique ID if needed
        return {'id': tapEdgeData.get('id'), 'source': tapEdgeData['source'], 'target': tapEdgeData['target']}, None # Store edge data, clear node store
    # If tap is not on an edge, don't change anything
    return no_update, no_update


# --- Callback to Update Visualizations ---

@callback(
    Output('graph-view', 'elements'),
    Output('embeddings-view', 'figure'),
    Output('feature-matrix-view', 'columns'),
    Output('feature-matrix-view', 'data'),
    Output('selected-info-view', 'children'),
    Output('attention-view', 'figure'),
    # Main trigger: Model output changes after inference
    Input('current-model-output-store', 'data'),
    # Secondary triggers: Selections change highlights/info panel
    Input('selected-node-store', 'data'),
    Input('selected-edge-store', 'data'),
    # Trigger for embedding plot type change
    Input('dim-reduction-dropdown', 'value'),
    # States needed for generating visualizations:
    State('current-graph-store', 'data'), # Need current graph structure
    State('dataset-info-store', 'data'), # Needed for class labels
    State('model-dropdown', 'value'), # Need model type for attention logic
    State('dataset-dropdown', 'value'), # Need dataset name for loading GAT model for attention
    prevent_initial_call=True # Visualizations updated after first inference automatically
)
def update_visualizations(model_output, selected_node_data, selected_edge_data,
                          dim_reduction_method,
                          graph_store_data, dataset_info, model_type, dataset_name):

    # Determine which input triggered the callback (useful for debugging)
    trigger = ctx.triggered_id
    print(f"Updating visualizations. Trigger: {trigger}")

    # Define default empty states for outputs
    empty_elements = []
    empty_fig = {'layout': {'xaxis': {'visible': False}, 'yaxis': {'visible': False}, 'annotations': [{'text': 'N/A', 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 12}}]}}
    empty_table_cols = []
    empty_table_data = []
    default_info_text = "Load data and select a model."
    default_attn_text = "Attention view requires a GAT model and a selected node."
    attn_fig = {'layout': {'xaxis': {'visible': False}, 'yaxis': {'visible': False}, 'annotations': [{'text': default_attn_text, 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 12}}]}}


    # --- Guard Clauses ---
    if graph_store_data is None or graph_store_data.get('num_nodes', 0) == 0:
        print("Update visualizations skipped: No graph data in store.")
        return empty_elements, empty_fig, empty_table_cols, empty_table_data, default_info_text, attn_fig

    # Check if model output is valid before proceeding
    predictions = None
    embeddings = None
    valid_model_output = False
    if model_output and model_output.get('predictions') is not None and model_output.get('embeddings') is not None:
         # Check if lengths match current number of nodes
         if len(model_output['predictions']) == graph_store_data['num_nodes'] and \
            len(model_output['embeddings']) == graph_store_data['num_nodes']:
              predictions = model_output['predictions'] # Should be list
              embeddings = model_output['embeddings']   # Should be list of lists
              valid_model_output = True
         else:
              print("Warning: Model output size mismatch with current graph nodes. Ignoring model output.")


    # --- Prepare data ---
    try:
        # Reconstruct PyG Data object for processing (needed for features, etc.)
        current_data = Data(
            x=torch.tensor(graph_store_data['features'], dtype=torch.float),
            edge_index=torch.tensor(graph_store_data['edge_index'], dtype=torch.long),
            num_nodes=graph_store_data['num_nodes']
        )
        # Add labels if they exist in the store
        if 'labels' in graph_store_data:
             current_data.y = torch.tensor(graph_store_data['labels'], dtype=torch.long)

    except Exception as e:
        print(f"Error reconstructing graph data for visualization: {e}")
        return empty_elements, empty_fig, empty_table_cols, empty_table_data, "Error loading graph data.", attn_fig


    class_map = dataset_info.get('class_map', {}) if dataset_info else {}
    selected_node_id = selected_node_data.get('id') if selected_node_data else None
    # Pass full selected_edge_data for highlighting check in data_to_cytoscape
    selected_edge_info_for_cyto = selected_edge_data

    # --- Generate Graph View Elements ---
    # Pass predictions only if they are valid
    cyto_elements = data_to_cytoscape(current_data, predictions if valid_model_output else None, class_map, selected_node_id, selected_edge_info_for_cyto)

    # --- Generate Embeddings Plot ---
    # Generate plot only if embeddings are valid
    if valid_model_output:
        embeddings_fig = plot_embeddings(embeddings, predictions, class_map, dim_reduction_method)
    else:
        embeddings_fig = empty_fig
        embeddings_fig['layout']['annotations'][0]['text'] = 'Embeddings (Not Available)'


    # --- Generate Feature Table ---
    # Show features for selected node, or all if none selected
    node_idx_for_features = None
    if selected_node_id:
        try:
             node_idx_for_features = [int(selected_node_id)]
        except (ValueError, TypeError):
             node_idx_for_features = None # Handle invalid ID case

    feature_cols, feature_data = create_feature_table(current_data.x, node_idx_for_features)

    # --- Generate Selected Info ---
    selected_info_content = []
    if selected_node_id:
        try:
            node_idx = int(selected_node_id)
            selected_info_content.append(html.Strong(f"Selected Node: {node_idx}"))
            # Show prediction only if valid
            if valid_model_output and node_idx < len(predictions):
                pred_class = int(predictions[node_idx])
                pred_label = class_map.get(pred_class, f'Class {pred_class}')
                selected_info_content.append(html.P(f"Predicted Class: {pred_label} ({pred_class})"))
            # Display true label if available
            if hasattr(current_data, 'y') and current_data.y is not None and node_idx < len(current_data.y):
                 true_class = int(current_data.y[node_idx].item())
                 # Handle potential dummy label (-1)
                 if true_class == -1:
                      selected_info_content.append(html.P(f"True Class: N/A (New Node)"))
                 else:
                      true_label = class_map.get(true_class, f'Class {true_class}')
                      selected_info_content.append(html.P(f"True Class: {true_label} ({true_class})"))

        except (ValueError, TypeError, IndexError) as e:
             selected_info_content = f"Error displaying info for node {selected_node_id}: {e}"

    elif selected_edge_data:
        source = selected_edge_data.get('source', 'N/A')
        target = selected_edge_data.get('target', 'N/A')
        selected_info_content.append(html.Strong(f"Selected Edge: {source} -> {target}"))
        # Could add info about connected nodes here

    else:
        selected_info_content = "Click on a node or edge to see details."


    # --- Generate Attention View (if GAT and node selected) ---
    # Generate attention only if valid model output exists
    if model_type == 'GAT' and selected_node_id and valid_model_output:
        try:
            node_idx = int(selected_node_id)
            # Reload the GAT model specifically for attention calculation
            gat_model = load_model(model_type, dataset_name)
            if gat_model:
                # Pass the reconstructed current_data
                att_data = get_attention_weights(gat_model, current_data, node_idx)
                if att_data:
                    neighbor_indices, attention_scores = att_data
                    if len(neighbor_indices) > 0:
                        # Create a bar chart of attention scores from neighbors
                        att_df = pd.DataFrame({'Neighbor Node': [str(n) for n in neighbor_indices], 'Attention Score': attention_scores})
                        att_df = att_df.sort_values('Attention Score', ascending=False)
                        attn_fig = px.bar(att_df, x='Neighbor Node', y='Attention Score',
                                          title=f'Attention Scores to Node {node_idx}',
                                          labels={'Attention Score': 'Attention'})
                        attn_fig.update_layout(xaxis_title="Source Neighbor Node", yaxis_title="Attention Score", title_font_size=14)
                    else:
                         attn_fig = {'layout': {'xaxis': {'visible': False}, 'yaxis': {'visible': False}, 'annotations': [{'text': f'Node {node_idx} has no incoming edges/attention', 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 12}}]}}
                else:
                     # Keep default message if get_attention_weights returns None
                     attn_fig = {'layout': {'xaxis': {'visible': False}, 'yaxis': {'visible': False}, 'annotations': [{'text': 'Could not retrieve attention weights.', 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 12}}]}}
            else:
                 attn_fig = {'layout': {'xaxis': {'visible': False}, 'yaxis': {'visible': False}, 'annotations': [{'text': 'GAT Model could not be loaded for attention.', 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 12}}]}}
        except (ValueError, TypeError, IndexError) as e:
             attn_fig = {'layout': {'xaxis': {'visible': False}, 'yaxis': {'visible': False}, 'annotations': [{'text': f'Error getting attention: {e}', 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 12}}]}}

    elif model_type != 'GAT':
         attn_fig = {'layout': {'xaxis': {'visible': False}, 'yaxis': {'visible': False}, 'annotations': [{'text': 'Attention view only available for GAT models.', 'xref': 'paper', 'yref': 'paper', 'showarrow': False, 'font': {'size': 12}}]}}
    # Else (GAT selected but no node selected, or model output invalid): keep the default message


    return cyto_elements, embeddings_fig, feature_cols, feature_data, selected_info_content, attn_fig


# --- Run App ---
if __name__ == '__main__':
    print("Starting Dash app...")
    print(f"IMPORTANT: Ensure pre-trained models exist in the '{MODEL_DIR}' directory.")
    print(f"IMPORTANT: Ensure datasets exist in the 'data' directory.")
    # Check if default model exists as a basic check
    # Determine path relative to the script file if possible, otherwise use current directory
    script_dir = os.path.dirname(__file__) if '__file__' in locals() else '.'
    default_model_path = os.path.join(script_dir, MODEL_DIR, f"{DEFAULT_MODEL}_{DEFAULT_DATASET}.pkl")

    if not os.path.exists(default_model_path):
         print(f"Warning: Default model '{default_model_path}' not found. Please run the training script.")

    # The initial load is now handled by setting prevent_initial_call=False
    # on the main load_dataset_callback definition.
    # No need for the extra app.callback block here anymore.

    app.run(debug=True, host='0.0.0.0', port=8050) # Set debug=False for production, host='0.0.0.0' to access from network, choose a port

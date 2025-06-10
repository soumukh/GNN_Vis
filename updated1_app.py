# -*- coding: utf-8 -*-
import dash
from dash import dcc, html, Input, Output, State, callback, ctx, no_update, ALL
import dash_cytoscape as cyto
import plotly.express as px
import plotly.graph_objects as go # For histogram and empty figures

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv
# ---> Correct Import Path for Explainer framework <---
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx, remove_self_loops, add_self_loops, k_hop_subgraph

try:
    from node2vec import Node2Vec
except ImportError:
    print("Node2Vec library not found for app. Please install it: pip install node2vec")
    # Optionally exit or handle the absence of node2vec gracefully
    # import sys
    # sys.exit(1) # Example: exit if node2vec is essential for all datasets


import networkx as nx
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap.umap_ as umap
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
NODE_DIM_REDUCTION = 'TSNE' # 'PCA', 'TSNE', or 'UMAP'

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

    # The standard forward method for GCN usually returns log_softmax,
    # which is often suitable for GNNExplainer's default 'log_probs' return type.
    # This forward is designed to be compatible with Explainer.
    def forward(self, x, edge_index, **kwargs): # Accept potential extra kwargs from Explainer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1) # Return only scores for explainer

    # Separate method for inference if you need embeddings alongside predictions
    def inference(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        embeddings = x
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1), embeddings


class GATNet(torch.nn.Module):
    """Basic GAT Network Definition."""
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        num_heads_out = 1
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=num_heads_out,
                             concat=False, dropout=0.6)
        self.init_args = {'hidden_channels': hidden_channels, 'heads': heads} # Store args

    # Standard forward for explainer (only returns scores)
    def forward(self, x, edge_index, **kwargs): # Accept potential extra kwargs from Explainer
        # Ensure dropout is off by calling model.eval() before using the explainer
        x, _ = self.conv1(x, edge_index, return_attention_weights=False)
        x = F.elu(x)
        x, _ = self.conv2(x, edge_index, return_attention_weights=False)
        return F.log_softmax(x, dim=1)

    # Separate method for inference needing embeddings or attention
    def inference(self, data, return_attention_weights=False):
        x, edge_index = data.x, data.edge_index
        x, att1 = self.conv1(x, edge_index, return_attention_weights=return_attention_weights)
        x = F.elu(x)
        embeddings = x
        x, att2 = self.conv2(x, edge_index, return_attention_weights=return_attention_weights)
        attention_weights = (att1, att2) if return_attention_weights else None
        log_probs = F.log_softmax(x, dim=1)

        if return_attention_weights:
            return log_probs, embeddings, attention_weights
        else:
            return log_probs, embeddings


class JazzDatasetWrapper:
    """Wraps the Jazz dataset PyG Data object to mimic Planetoid structure."""
    def __init__(self, data, num_node_features, num_classes):
        self.data = data # The actual PyG Data object
        self.num_node_features = num_node_features
        self.num_classes = num_classes
        self.name = 'Jazz' # Keep the name attribute if needed elsewhere

    def __getitem__(self, idx):
        if idx == 0:
            return self.data
        else:
            raise IndexError("JazzDatasetWrapper index out of range")

    def __len__(self):
        return 1

# --- Helper Functions ---
def load_jazz_dataset(path, n2v_dim=16, walk_len=10, num_walk=50, workers=4):
    """Loads the Jazz dataset, generates Node2Vec embeddings, and prepares PyG Data."""
    try:
        edge_file = os.path.join(path, 'jazz.cites')
        content_file = os.path.join(path, 'jazz.content')
        print(f"--- Debugging load_jazz_dataset ---")
        print(f"Attempting to load from base path: {path}")
        print(f"Checking for edge file at: {os.path.abspath(edge_file)}")
        print(f"Checking for content file at: {os.path.abspath(content_file)}")
        if not os.path.exists(edge_file) or not os.path.exists(content_file):
            print(f"*** ERROR: Jazz dataset files not found.")
            return None
        else:
            print(f"Files found. Proceeding...")
        try:
            content_df = pd.read_csv(content_file, sep='\t', header=None)
            node_ids = content_df.iloc[:, 0].astype(int).values
            if content_df.shape[1] > 2:
                original_features = content_df.iloc[:, 1:-1].values
                print(f"  App: Found original features shape: {original_features.shape}")
            else:
                original_features = None
                print("  App: No original features found.")
            labels = content_df.iloc[:, -1].astype(int).values
            node_ids_list = sorted(list(np.unique(node_ids)))
            num_nodes = len(node_ids_list)
        except Exception as e:
            print(f"App: Error reading Jazz content file: {e}")
            return None
        node_map = {old_id: new_id for new_id, old_id in enumerate(node_ids_list)}
        G = nx.Graph()
        G.add_nodes_from(node_map.keys())
        edges_added_count = 0
        with open(edge_file, 'r') as f:
            for line in f:
                if line.startswith('#'): continue
                try:
                    u, v = map(int, line.strip().split('\t'))
                    if u in node_map and v in node_map:
                        G.add_edge(u, v)
                        edges_added_count += 1
                except ValueError: pass
        print(f"  App: Built graph with {G.number_of_nodes()} nodes and {edges_added_count} edges.")
        print(f"  App: Generating Node2Vec (dim={n2v_dim}, len={walk_len}, num={num_walk})...")
        node2vec = Node2Vec(G, dimensions=n2v_dim, walk_length=walk_len, num_walks=num_walk, workers=workers, quiet=True)
        model_n2v = node2vec.fit(window=5, min_count=1, batch_words=4, epochs=1)
        print("  App: Node2Vec fitting complete.")
        n2v_embeddings = np.zeros((num_nodes, n2v_dim))
        for old_id, new_id in node_map.items():
            try: n2v_embeddings[new_id] = model_n2v.wv[str(old_id)]
            except KeyError: pass
        if original_features is not None:
            aligned_original_features = np.zeros((num_nodes, original_features.shape[1]))
            feature_map = dict(zip(content_df.iloc[:, 0].astype(int).values, original_features))
            for old_id, new_id in node_map.items():
                 if old_id in feature_map: aligned_original_features[new_id] = feature_map[old_id]
            final_features = np.hstack((aligned_original_features, n2v_embeddings))
        else:
            final_features = n2v_embeddings
        x = torch.tensor(final_features, dtype=torch.float)
        num_node_features = x.shape[1]
        print(f"  App: Final feature dimension: {num_node_features}")
        aligned_labels = np.zeros(num_nodes, dtype=int)
        label_map = dict(zip(content_df.iloc[:, 0].astype(int).values, labels))
        for old_id, new_id in node_map.items():
            if old_id in label_map: aligned_labels[new_id] = label_map[old_id]
        y = torch.tensor(aligned_labels, dtype=torch.long)
        num_classes = len(torch.unique(y))
        mapped_edges = []
        for u, v in G.edges(): mapped_edges.append([node_map[u], node_map[v]])
        edge_index_mapped = torch.tensor(mapped_edges, dtype=torch.long).t().contiguous()
        from torch_geometric.utils import to_undirected
        edge_index = to_undirected(edge_index_mapped)
        data = Data(x=x, edge_index=edge_index, y=y)
        data.num_nodes = num_nodes
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        wrapper_instance = JazzDatasetWrapper(data, num_node_features, num_classes)
        print("App: Dataset 'Jazz' loaded successfully using faster Node2Vec.")
        return wrapper_instance
    except Exception as e:
        print(f"App: Error loading or processing Jazz dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_dataset(name):
    """Loads a dataset for the dashboard, handling Planetoid and custom Jazz."""
    global DATA_DIR
    try:
        script_dir = os.path.dirname(__file__) if '__file__' in locals() else '.'
        base_data_dir = os.path.join(script_dir, DATA_DIR)
        path = os.path.join(base_data_dir, name)
        print(f"App: Attempting to load dataset '{name}'...")
        if name in ['Cora', 'CiteSeer', 'PubMed']:
             dataset = Planetoid(root=base_data_dir, name=name)
             print(f"App: Planetoid dataset '{name}' loaded successfully.")
             return dataset
        elif name == 'Jazz':
             print(f"App: Calling load_jazz_dataset for path: {path}")
             return load_jazz_dataset(path)
        else:
            print(f"App: Dataset '{name}' is not recognized by the load_dataset function.")
            return None
    except FileNotFoundError:
         print(f"App: Error loading dataset {name}. Required files not found in expected location: {path}")
         return None
    except Exception as e:
        print(f"App: Error loading dataset {name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_model(model_type, dataset_name, device):
    """Loads a pre-trained model from a .pkl file onto the specified device."""
    model_filename = f"{model_type}_{dataset_name}.pkl"
    script_dir = os.path.dirname(__file__) if '__file__' in locals() else '.'
    model_path = os.path.join(script_dir, MODEL_DIR, model_filename)
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    try:
        dataset = load_dataset(dataset_name)
        if dataset is None: return None
        in_channels = dataset.num_node_features
        out_channels = dataset.num_classes
        with open(model_path, 'rb') as f:
            model_state_dict, model_args = pickle.load(f)
        if model_type == 'GCN':
            hidden_channels = model_args.get('hidden_channels', 16)
            model = GCNNet(in_channels, hidden_channels, out_channels)
        elif model_type == 'GAT':
            hidden_channels = model_args.get('hidden_channels', 8)
            heads = model_args.get('heads', 8)
            model = GATNet(in_channels, hidden_channels, out_channels, heads=heads)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()
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

@torch.no_grad()
def run_inference(model, data, model_type, device):
    """Runs GNN inference on the specified device."""
    if model is None or data is None: return None, None
    model.eval()
    model.to(device)
    data = data.to(device)
    try:
        # Use the dedicated inference method if available
        if hasattr(model, 'inference'):
            log_probs, embeddings = model.inference(data)
        else: # Fallback to standard forward, assuming it returns scores and embeddings
             # This might fail if standard forward was changed for explainer
             # Re-evaluate if this fallback is needed or if models should always have .inference()
             print("Warning: Using standard model forward for inference, ensure it returns (log_probs, embeddings).")
             log_probs, embeddings = model(data.x, data.edge_index) # Assuming standard forward takes x, edge_index

        predictions = log_probs.argmax(dim=1)
        return predictions.cpu(), embeddings.cpu()
    except Exception as e:
        print(f"Error during model inference on device {device}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

@torch.no_grad()
def get_attention_weights(model, data, node_idx, device):
    """Extracts attention weights for a specific node (GAT only) on the specified device."""
    if not isinstance(model, GATNet) or data is None or node_idx is None: return None
    model.eval()
    model.to(device)
    data = data.to(device)
    num_nodes = data.num_nodes
    if not (0 <= node_idx < num_nodes):
        print(f"Invalid node index {node_idx} for attention calculation.")
        return None
    try:
        # Use the dedicated inference method if available and it returns attention
        if hasattr(model, 'inference'):
             _, _, attention_output = model.inference(data, return_attention_weights=True)
        else: # Assume standard forward can return attention (might need adjustment)
             # This path is unlikely if GATNet always has .inference()
             print("Warning: Trying to get attention from standard GAT forward.")
             _, _, attention_output = model(data.x, data.edge_index, return_attention_weights=True) # Hypothetical call

        if attention_output is None:
             print("Model did not return attention weights.")
             return None
        edge_index_att, alpha_att = attention_output[-1]
        mask = edge_index_att[1] == node_idx
        if not torch.any(mask): return None
        neighbor_indices = edge_index_att[0][mask]
        attention_scores = alpha_att[mask].mean(dim=1).squeeze()
        return neighbor_indices.cpu().numpy(), attention_scores.cpu().numpy()
    except Exception as e:
        print(f"Error getting attention weights: {e}")
        import traceback
        traceback.print_exc()
        return None

def data_to_cytoscape(data, predictions=None, class_map=None, selected_node_id=None, selected_edge_data=None, neighbor_ids=None, explanation_masks=None):
    """Converts PyG Data object to Cytoscape elements list."""
    if data is None or data.num_nodes == 0: return []
    if neighbor_ids is None: neighbor_ids = set()
    else: neighbor_ids = set(neighbor_ids)

    explained_edge_mask = None
    edge_importance_threshold = 0.5 # Example threshold
    if explanation_masks and 'edge_mask' in explanation_masks and explanation_masks['edge_mask'] is not None:
        explained_edge_mask = explanation_masks['edge_mask']
        # Convert to numpy if it's a tensor, handle list case
        if isinstance(explained_edge_mask, torch.Tensor):
            explained_edge_mask = explained_edge_mask.cpu().numpy()
        elif isinstance(explained_edge_mask, list):
            explained_edge_mask = np.array(explained_edge_mask)
        print(f"Using edge importance threshold: {edge_importance_threshold}")


    nodes = []
    default_color = '#808080'
    color_palette = px.colors.qualitative.Plotly
    preds_list = None
    if predictions is not None:
        if isinstance(predictions, torch.Tensor): preds_list = predictions.cpu().numpy()
        elif isinstance(predictions, (list, np.ndarray)): preds_list = np.array(predictions)
        if len(preds_list) != data.num_nodes:
            print(f"Warning: Predictions length mismatch. Ignoring.")
            preds_list = None

    for i in range(data.num_nodes):
        node_id_str = str(i)
        node_data = {'id': node_id_str, 'label': f'Node {i}'}
        node_classes = []
        node_base_color = default_color
        if preds_list is not None and i < len(preds_list):
             pred_class = int(preds_list[i])
             node_data['class_pred'] = pred_class
             node_base_color = color_palette[pred_class % len(color_palette)]
             if class_map and pred_class in class_map: node_data['label'] += f'\n({class_map[pred_class]})'
        if hasattr(data, 'y') and data.y is not None and i < len(data.y):
             true_class = int(data.y[i].item())
             node_data['true_class'] = true_class
             if class_map and true_class in class_map: node_data['true_label'] = class_map[true_class]
             elif true_class != -1: node_data['true_label'] = f'Class {true_class}'
             else: node_data['true_label'] = 'N/A'
        is_selected = False
        if selected_node_id is not None and node_id_str == selected_node_id:
            is_selected = True
            node_classes.append('selected')
        is_neighbor = i in neighbor_ids
        if is_neighbor and not is_selected: node_classes.append('neighbor')
        node_data['classes'] = ' '.join(node_classes)
        node_style = {'background-color': node_base_color}
        nodes.append({'data': node_data, 'style': node_style})

    edges = []
    if data.edge_index is not None and data.edge_index.shape[1] > 0:
        edge_index_display = data.edge_index.cpu().numpy()
        sel_source = selected_edge_data.get('source') if selected_edge_data else None
        sel_target = selected_edge_data.get('target') if selected_edge_data else None
        edge_tuple_to_idx = {(u, v): i for i, (u, v) in enumerate(zip(edge_index_display[0], edge_index_display[1]))}

        for i in range(edge_index_display.shape[1]):
            source = str(edge_index_display[0, i])
            target = str(edge_index_display[1, i])
            source_int = edge_index_display[0, i]
            target_int = edge_index_display[1, i]
            if source_int >= data.num_nodes or target_int >= data.num_nodes: continue
            edge_id = f"{source}_{target}_{i}"
            edge_data = {'id': edge_id, 'source': source, 'target': target}
            edge_classes = []
            edge_style = {}
            is_selected_edge = False
            if sel_source is not None and sel_target is not None:
                 if (sel_source == source and sel_target == target) or (sel_source == target and sel_target == source):
                      is_selected_edge = True
            if is_selected_edge: edge_classes.append('selected-edge')

            is_explained_edge = False
            if explained_edge_mask is not None and isinstance(explained_edge_mask, np.ndarray): # Check if mask is valid numpy array
                edge_idx_forward = edge_tuple_to_idx.get((source_int, target_int))
                edge_idx_backward = edge_tuple_to_idx.get((target_int, source_int))
                score = 0.0
                if edge_idx_forward is not None and edge_idx_forward < len(explained_edge_mask):
                    score = max(score, explained_edge_mask[edge_idx_forward])
                if edge_idx_backward is not None and edge_idx_backward < len(explained_edge_mask):
                    score = max(score, explained_edge_mask[edge_idx_backward])

                if score > edge_importance_threshold:
                    is_explained_edge = True
                    edge_classes.append('explained-edge')
                    if edge_importance_threshold < 1.0:
                        scale_factor = max(0, (score - edge_importance_threshold) / (1.0 - edge_importance_threshold)) # Ensure non-negative
                        opacity = 0.3 + 0.7 * scale_factor
                        width = 1 + 3 * scale_factor
                        edge_style['opacity'] = min(max(opacity, 0.3), 1.0)
                        edge_style['width'] = min(max(width, 1.0), 4.0)
                    else:
                        edge_style['opacity'] = 1.0
                        edge_style['width'] = 4.0

            edge_data['classes'] = ' '.join(edge_classes)
            edges.append({'data': edge_data, 'style': edge_style})
    return nodes + edges


def plot_embeddings(embeddings, predictions, true_labels=None, class_map=None, dim_reduction='TSNE', selected_node_id=None):
    """Generates Plotly scatter plot for node embeddings."""
    if embeddings is None: return go.Figure(layout={'title': "Embeddings (Not Available)", 'xaxis': {'visible': False}, 'yaxis': {'visible': False}})
    embeddings_np = embeddings.cpu().numpy() if isinstance(embeddings, torch.Tensor) else np.array(embeddings)
    predictions_np = predictions.cpu().numpy() if isinstance(predictions, torch.Tensor) else np.array(predictions) if predictions is not None else None
    true_labels_np = true_labels.cpu().numpy() if isinstance(true_labels, torch.Tensor) else np.array(true_labels) if true_labels is not None else None
    num_nodes = embeddings_np.shape[0]
    if num_nodes == 0 or \
       (predictions_np is not None and embeddings_np.shape[0] != predictions_np.shape[0]) or \
       (true_labels_np is not None and embeddings_np.shape[0] != true_labels_np.shape[0]):
        print("Warning: Embeddings/predictions/labels size mismatch or zero nodes.")
        return go.Figure(layout={'title': "Embeddings (Invalid Data)", 'xaxis': {'visible': False}, 'yaxis': {'visible': False}})
    if num_nodes < 2: return go.Figure(layout={'title': "Embeddings (Need >= 2 nodes)", 'xaxis': {'visible': False}, 'yaxis': {'visible': False}})

    n_components = 2
    embeddings_2d = None
    reduction_method_used = dim_reduction
    n_neighbors = 15
    min_dist = 0.1
    try:
        if dim_reduction == 'TSNE':
            if num_nodes < n_components + 1:
                 print(f"Not enough samples ({num_nodes}) for t-SNE, switching to PCA.")
                 reduction_method_used = 'PCA'; reducer = PCA(n_components=n_components)
            else:
                 perplexity = min(30.0, max(5.0, float(num_nodes - 1) / 3.0))
                 try: reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto', n_iter=300)
                 except TypeError: print("Using older TSNE initialization."); reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42, init='pca', n_iter=300)
        elif dim_reduction == 'UMAP':
             effective_n_neighbors = min(n_neighbors, num_nodes - 1)
             if effective_n_neighbors < 2:
                  print(f"Not enough samples ({num_nodes}) for UMAP, switching to PCA.")
                  reduction_method_used = 'PCA'; reducer = PCA(n_components=n_components)
             else:
                  print(f"Running UMAP with n_neighbors={effective_n_neighbors}, min_dist={min_dist}")
                  reducer = umap.UMAP(n_neighbors=effective_n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
        elif dim_reduction == 'PCA': reduction_method_used = 'PCA'; reducer = PCA(n_components=n_components)
        else: print(f"Unknown reduction method '{dim_reduction}', defaulting to PCA."); reduction_method_used = 'PCA'; reducer = PCA(n_components=n_components)
        embeddings_2d = reducer.fit_transform(embeddings_np)
    except Exception as e:
        print(f"Error during dimensionality reduction ({reduction_method_used}): {e}")
        embeddings_2d = np.random.rand(num_nodes, n_components)
        reduction_method_used = f"{reduction_method_used} (Error - Random Fallback)"

    df = pd.DataFrame(embeddings_2d, columns=['Dim 1', 'Dim 2'])
    df['Node ID'] = [str(i) for i in range(num_nodes)]
    color_column = None
    if predictions_np is not None:
        df['Predicted Class'] = [class_map.get(int(p), f'Class {int(p)}') for p in predictions_np] if class_map else [f'Class {int(p)}' for p in predictions_np]
        color_column = 'Predicted Class'
    else:
        df['Predicted Class'] = 'N/A'
        if true_labels_np is not None:
             df['True Label Color'] = [class_map.get(int(t), f'Class {int(t)}') if t != -1 else 'N/A' for t in true_labels_np] if class_map else [f'Class {int(t)}' if t != -1 else 'N/A' for t in true_labels_np]
             color_column = 'True Label Color'
    if true_labels_np is not None:
        df['True Label'] = [class_map.get(int(t), f'Class {int(t)}') if t != -1 else 'N/A' for t in true_labels_np] if class_map else [f'Class {int(t)}' if t != -1 else 'N/A' for t in true_labels_np]
    else: df['True Label'] = 'N/A'
    hover_data = ['Node ID', 'Predicted Class', 'True Label']
    fig_title = f'Node Embeddings ({reduction_method_used})'
    try:
        fig = px.scatter(df, x='Dim 1', y='Dim 2', color=color_column, hover_data=hover_data, custom_data=['Node ID'], title=fig_title, color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.update_layout(legend_title_text=color_column if color_column else 'Class', clickmode='event+select')
        fig.update_traces(marker=dict(size=8, opacity=0.8), unselected=dict(marker=dict(opacity=0.5)), selected=dict(marker=dict(size=12, opacity=1.0)))
        if selected_node_id is not None:
             try:
                 selected_point_index = df[df['Node ID'] == selected_node_id].index[0]
                 fig.update_traces(selectedpoints=[selected_point_index], selector=dict(type='scatter'))
             except IndexError: print(f"Warning: Selected node ID {selected_node_id} not found in embedding plot.")
    except Exception as e:
        print(f"Error creating embeddings plot: {e}")
        fig = go.Figure(layout={'title': f"Embeddings Plot Error ({reduction_method_used})", 'xaxis': {'visible': False}, 'yaxis': {'visible': False}})
    return fig

def plot_feature_histogram(features, node_id, feature_mask=None, top_k=10):
    """Creates a histogram of features, highlighting important ones if mask is provided."""
    if features is None: return go.Figure(layout={'title': "Feature Histogram (Invalid Data)"})
    if isinstance(features, torch.Tensor): features = features.cpu().numpy()
    if features.ndim != 1: return go.Figure(layout={'title': f"Feature Histogram Error (Node {node_id}): Expected 1D features"})

    feature_indices = np.arange(len(features))
    feature_values = features
    title = f"Feature Histogram for Node {node_id}"
    indices_to_plot = feature_indices
    plot_colors = ['blue'] * len(indices_to_plot)

    if feature_mask is not None:
        title = f"Top {top_k} Important Features for Node {node_id} (GNNExplainer)"
        if isinstance(feature_mask, torch.Tensor): feature_mask = feature_mask.cpu().numpy()
        if isinstance(feature_mask, (list, np.ndarray)) and len(feature_mask) == len(features):
            feature_mask_np = np.array(feature_mask)
            important_indices = np.argsort(feature_mask_np)[::-1][:top_k]
            indices_to_plot = important_indices
            plot_colors = ['red'] * len(indices_to_plot)
        else:
            print("Warning: Feature mask invalid. Showing non-zero features instead.")
            feature_mask = None # Invalidate mask for sorting logic below

    if feature_mask is None: # Show non-zero if no valid mask
        threshold = 1e-6
        non_zero_indices = np.where(np.abs(features) > threshold)[0]
        if len(non_zero_indices) == 0: return go.Figure(layout={'title': title + " (No non-zero features)"})
        indices_to_plot = non_zero_indices
        plot_colors = ['blue'] * len(indices_to_plot)
        title = f"Non-Zero Features for Node {node_id} ({len(non_zero_indices)} features)"

    plot_labels = [f'F_{i}' for i in indices_to_plot]
    plot_values = feature_values[indices_to_plot]
    df = pd.DataFrame({'Feature Index': plot_labels, 'Value': plot_values, 'IndexInt': indices_to_plot})

    if feature_mask is not None and 'feature_mask_np' in locals(): # Sort by importance if mask was valid
         df['Importance'] = feature_mask_np[indices_to_plot]
         df = df.sort_values('Importance', ascending=False)
    else: # Sort by value otherwise
         df = df.sort_values('Value', ascending=False)

    fig = px.bar(df, x='Feature Index', y='Value', title=title)
    fig.update_traces(marker_color=plot_colors)
    num_features_shown = len(df)
    if num_features_shown > 20: fig.update_layout(xaxis_tickangle=-45)
    elif num_features_shown > 50: fig.update_layout(xaxis_tickangle=-60, xaxis={'tickmode': 'linear', 'dtick': 5})
    fig.update_layout(xaxis_title="Feature Index", yaxis_title="Feature Value")
    if np.all(np.isclose(plot_values, 1.0)): fig.update_yaxes(range=[0, 1.2])
    return fig

# --- Dash App Initialization ---
app = dash.Dash(__name__, suppress_callback_exceptions=True,
                external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'],
                prevent_initial_callbacks='initial_duplicate')
server = app.server

# --- App Layout ---
# (Layout remains the same as the previous version - no changes needed here)
app.layout = html.Div([
    dcc.Store(id='current-graph-store'),
    dcc.Store(id='current-model-output-store'),
    dcc.Store(id='selected-node-store'),
    dcc.Store(id='selected-edge-store'),
    dcc.Store(id='dataset-info-store'),
    dcc.Store(id='explanation-store'),
    html.H1("Interactive GNN Explainer", style={'textAlign': 'center', 'marginBottom': '20px'}),
    html.Div([
        html.Div([
            html.Label("Select Dataset:", style={'marginRight': '10px'}),
            dcc.Dropdown(id='dataset-dropdown', options=[{'label': ds, 'value': ds} for ds in AVAILABLE_DATASETS], value=DEFAULT_DATASET, clearable=False, style={'width': '150px', 'marginRight': '20px'}),
            html.Label("Select Model:", style={'marginRight': '10px'}),
            dcc.Dropdown(id='model-dropdown', options=[{'label': m, 'value': m} for m in AVAILABLE_MODELS], value=DEFAULT_MODEL, clearable=False, style={'width': '150px'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
        html.Div([
            html.Button('Add Node (Copy Selected)', id='add-node-button', n_clicks=0, style={'marginRight': '5px'}),
            html.Button('Remove Selected Node', id='remove-node-button', n_clicks=0, style={'marginRight': '20px'}),
            dcc.Input(id='new-edge-source', type='text', placeholder='Source Node ID', style={'width': '120px', 'marginRight': '5px'}),
            dcc.Input(id='new-edge-target', type='text', placeholder='Target Node ID', style={'width': '120px', 'marginRight': '5px'}),
            html.Button('Add Edge', id='add-edge-button', n_clicks=0, style={'marginRight': '5px'}),
            html.Button('Remove Selected Edge', id='remove-edge-button', n_clicks=0),
        ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '10px'})
    ], id='control-panel', style={'padding': '15px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'marginBottom': '20px', 'backgroundColor': '#f9f9f9'}),
    html.Div(id='status-message-area', style={'padding': '10px', 'marginBottom': '10px', 'border': '1px dashed #ccc', 'borderRadius': '5px', 'minHeight': '40px', 'backgroundColor': '#f0f0f0', 'whiteSpace': 'pre-wrap'}),
    html.Div([
        html.Div([
            html.H3("Graph View"),
            html.Div([
                dcc.Input(id='select-node-input', type='number', placeholder='Enter Node ID...', min=0, step=1, style={'width': '150px', 'marginRight': '5px'}),
                html.Button('Select Node', id='select-node-button', n_clicks=0)
            ], style={'marginBottom': '10px'}),
            cyto.Cytoscape(
                id='graph-view',
                layout={'name': 'cose', 'idealEdgeLength': 100, 'nodeOverlap': 20, 'refresh': 20, 'fit': True, 'padding': 30, 'randomize': False, 'componentSpacing': 100, 'nodeRepulsion': 400000, 'edgeElasticity': 100, 'nestingFactor': 5, 'gravity': 80, 'numIter': 1000, 'initialTemp': 200, 'coolingFactor': 0.95, 'minTemp': 1.0},
                style={'width': '100%', 'height': '450px', 'border': '1px solid #ddd', 'borderRadius': '5px'},
                stylesheet=[
                    {'selector': 'node', 'style': { 'label': 'data(label)', 'font-size': '10px', 'text-valign': 'center', 'text-halign': 'center', 'background-color': '#808080', 'width': 15, 'height': 15, 'color': '#fff', 'text-outline-width': 2, 'text-outline-color': '#555', 'border-width': 0, 'shape': 'ellipse'}},
                    {'selector': 'node.selected', 'style': {'background-color': '#FFA500', 'width': 25, 'height': 25, 'border-color': '#FF0000', 'border-width': 3, 'shape': 'ellipse', 'z-index': 9999 }},
                    {'selector': 'node.neighbor', 'style': {'border-color': '#007bff', 'border-width': 3, 'border-style': 'dashed' }},
                    {'selector': 'edge', 'style': {'curve-style': 'bezier', 'target-arrow-shape': 'none', 'line-color': '#cccccc', 'target-arrow-color': '#cccccc', 'opacity': 0.5, 'width': 1.5 }},
                    {'selector': 'edge.selected-edge', 'style': {'line-color': '#FF0000', 'target-arrow-color': '#FF0000', 'width': 3, 'opacity': 1.0, 'z-index': 9998 }},
                    {'selector': 'edge.explained-edge', 'style': { 'line-color': '#39FF14', 'target-arrow-color': '#39FF14', 'width': 4, 'opacity': 0.9, 'z-index': 9997 }}
                ]
            ),
            html.H3("Node Embeddings"),
            dcc.Dropdown(id='dim-reduction-dropdown', options=[{'label': 't-SNE', 'value': 'TSNE'}, {'label': 'PCA', 'value': 'PCA'}, {'label': 'UMAP', 'value': 'UMAP'} ], value='UMAP', clearable=False, style={'width': '100px', 'marginBottom': '10px'}),
            dcc.Graph(id='embeddings-view', style={'height': '400px'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '2%'}),
        html.Div([
            html.H3("Feature Importance (Selected Node)"),
            dcc.Graph(id='feature-histogram-view', style={'height': '250px'}),
            html.H3("Selected Node/Edge Info"),
            html.Div(id='selected-info-view', style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '10px', 'minHeight': '100px', 'marginBottom': '10px', 'backgroundColor': '#f9f9f9'}),
            html.H3("Neighbor Analysis"),
            html.Div(id='neighborhood-info-view', style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '10px', 'minHeight': '150px', 'maxHeight': '250px', 'overflowY': 'auto', 'marginBottom': '10px', 'backgroundColor': '#f9f9f9', 'fontSize': 'small'}),
            html.H3("GNN Explanation (Selected Node)"),
            html.Div(id='reasoning-output', style={'padding': '10px', 'marginTop': '5px', 'fontStyle': 'italic', 'color': '#333', 'border': '1px solid #e0e0e0', 'borderRadius': '5px', 'backgroundColor': '#fafafa', 'minHeight': '80px'}),
            html.H3("Attention Weights (GAT Only)"),
            dcc.Graph(id='attention-view', style={'height': '250px'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ])
], style={'padding': '20px'})


# --- Callbacks ---

# Load Dataset
@callback(
    Output('current-graph-store', 'data'),
    Output('dataset-info-store', 'data'),
    Output('status-message-area', 'children', allow_duplicate=True),
    Output('explanation-store', 'data', allow_duplicate=True),
    Input('dataset-dropdown', 'value'),
    prevent_initial_call='initial_duplicate'
)
def load_dataset_callback(dataset_name):
    """Loads dataset, stores its basic structure and info."""
    print(f"Loading dataset: {dataset_name}")
    status_message = f"Loading dataset '{dataset_name}'..."
    dataset = load_dataset(dataset_name)
    empty_graph_store = {'edge_index': [[], []], 'features': [], 'labels': [], 'num_nodes': 0, 'train_mask': [], 'val_mask': [], 'test_mask': []}
    empty_dataset_info = {'class_map': {}, 'num_node_features': 0, 'num_classes': 0}
    if dataset is None:
        error_message = f"Error: Failed to load dataset '{dataset_name}'."
        return empty_graph_store, empty_dataset_info, error_message, None
    try:
        data = dataset[0]
        required_attrs = ['edge_index', 'x', 'y', 'num_nodes']
        missing_attrs = [attr for attr in required_attrs if not hasattr(data, attr)]
        if missing_attrs:
            error_message = f"Error: Dataset '{dataset_name}' missing attributes: {missing_attrs}"
            return empty_graph_store, empty_dataset_info, error_message, None
        graph_store_data = {
            'edge_index': data.edge_index.cpu().tolist() if data.edge_index is not None else [[],[]],
            'features': data.x.cpu().tolist() if data.x is not None else [],
            'labels': data.y.cpu().tolist() if data.y is not None else [],
            'num_nodes': data.num_nodes,
            'train_mask': data.train_mask.cpu().tolist() if hasattr(data, 'train_mask') and data.train_mask is not None else [False]*data.num_nodes,
            'val_mask': data.val_mask.cpu().tolist() if hasattr(data, 'val_mask') and data.val_mask is not None else [False]*data.num_nodes,
            'test_mask': data.test_mask.cpu().tolist() if hasattr(data, 'test_mask') and data.test_mask is not None else [False]*data.num_nodes,
        }
        class_map = {}
        if hasattr(dataset, 'num_classes') and data.y is not None:
            num_classes = dataset.num_classes
            class_map = {i: f'Class {i}' for i in range(num_classes)}
            try:
                ds_name_lower = dataset_name.lower()
                if ds_name_lower == 'cora' and num_classes == 7: class_map = {0: 'Theory', 1: 'RL', 2: 'GA', 3: 'NN', 4: 'Prob', 5: 'Case', 6: 'Rule'}
                elif ds_name_lower == 'citeseer' and num_classes == 6: class_map = {0: 'Agents', 1: 'AI', 2: 'DB', 3: 'IR', 4: 'ML', 5: 'HCI'}
                elif ds_name_lower == 'pubmed' and num_classes == 3: class_map = {0: 'Diabetes Exp', 1: 'Diabetes T1', 2: 'Diabetes T2'}
                elif ds_name_lower == 'jazz':
                     unique_labels = sorted(torch.unique(data.y).tolist())
                     class_map = {i: f'Community {i}' for i in unique_labels if i >= 0}
            except Exception as map_err: print(f"Warning: Could not apply specific class names: {map_err}")
        dataset_info = {
            'class_map': class_map,
            'num_node_features': dataset.num_node_features if hasattr(dataset, 'num_node_features') else 0,
            'num_classes': dataset.num_classes if hasattr(dataset, 'num_classes') else 0
        }
        status_message = f"Dataset '{dataset_name}' loaded. Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1] if data.edge_index is not None else 0}, Features: {dataset_info['num_node_features']}."
        print(status_message)
        return graph_store_data, dataset_info, status_message, None
    except Exception as e:
        error_message = f"Error processing data from '{dataset_name}': {e}"
        import traceback; traceback.print_exc()
        return empty_graph_store, empty_dataset_info, error_message, None

# Run Inference
@callback(
    Output('current-model-output-store', 'data'),
    Output('status-message-area', 'children', allow_duplicate=True),
    Output('explanation-store', 'data', allow_duplicate=True),
    Input('current-graph-store', 'data'),
    Input('model-dropdown', 'value'),
    State('dataset-dropdown', 'value'),
    prevent_initial_call=True
)
def run_inference_callback(graph_store_data, model_type, dataset_name):
    """Runs inference when the graph data or model selection changes."""
    triggered_input = ctx.triggered_id
    print(f"Running inference callback. Trigger: {triggered_input}")
    empty_output = {'predictions': None, 'embeddings': None}
    clear_explanation = None
    if not graph_store_data or graph_store_data.get('num_nodes', 0) == 0:
        print("Inference skipped: Graph data empty.")
        return empty_output, "Load a dataset and model.", clear_explanation
    if not model_type or not dataset_name:
        print("Inference skipped: Missing model/dataset name.")
        return no_update, "Select dataset and model.", clear_explanation
    status_message = f"Loading {model_type} model for {dataset_name} and running inference on {device}..."
    model = load_model(model_type, dataset_name, device)
    if model is None:
        error_message = f"Inference skipped: {model_type} model for {dataset_name} could not be loaded."
        print(error_message)
        return empty_output, error_message, clear_explanation
    try:
        current_data = Data(
            x=torch.tensor(graph_store_data['features'], dtype=torch.float),
            edge_index=torch.tensor(graph_store_data['edge_index'], dtype=torch.long),
            y=torch.tensor(graph_store_data['labels'], dtype=torch.long),
            num_nodes=graph_store_data['num_nodes']
        )
    except Exception as e:
        error_message = f"Error reconstructing graph data: {e}"
        print(error_message)
        return empty_output, error_message, clear_explanation
    predictions, embeddings = run_inference(model, current_data, model_type, device)
    if predictions is None or embeddings is None:
        error_message = f"Inference failed for {model_type} on {dataset_name} using {device}."
        print(error_message)
        return empty_output, error_message, clear_explanation
    print(f"Inference complete. Predictions shape: {predictions.shape}, Embeddings shape: {embeddings.shape}")
    status_message = f"Inference complete for {model_type} on {dataset_name} (using {device})."
    output_data = {'predictions': predictions.tolist(), 'embeddings': embeddings.tolist()}
    return output_data, status_message, clear_explanation

# --- Graph Editing Callbacks (Add Node, Remove Node, Add Edge, Remove Edge) ---
# Add Node
@callback(
    Output('current-graph-store', 'data', allow_duplicate=True),
    Output('status-message-area', 'children', allow_duplicate=True),
    Input('add-node-button', 'n_clicks'),
    State('selected-node-store', 'data'),
    State('current-graph-store', 'data'),
    prevent_initial_call=True
)
def add_node_like_selected_callback(n_clicks, selected_node_data, graph_store_data):
    if n_clicks == 0: return no_update, no_update
    if not selected_node_data or 'id' not in selected_node_data: return no_update, "Select a node to copy features."
    if not graph_store_data or graph_store_data.get('num_nodes', 0) == 0: return no_update, "Graph data not loaded."
    try:
        template_node_id = int(selected_node_data['id'])
        num_nodes = graph_store_data['num_nodes']
        if not (0 <= template_node_id < num_nodes): return no_update, f"Invalid template node ID: {template_node_id}"
        features_to_copy = graph_store_data['features'][template_node_id]
        new_node_id = num_nodes
        new_graph_data = copy.deepcopy(graph_store_data)
        new_graph_data['num_nodes'] += 1
        new_graph_data['features'].append(copy.deepcopy(features_to_copy))
        new_graph_data['labels'].append(-1)
        new_graph_data['train_mask'].append(False); new_graph_data['val_mask'].append(False); new_graph_data['test_mask'].append(False)
        status_message = f"Added Node {new_node_id} (copied from Node {template_node_id}). Total: {new_graph_data['num_nodes']}"
        print(status_message)
        return new_graph_data, status_message
    except Exception as e: error_message = f"Error adding node: {e}"; print(error_message); return no_update, error_message

# Remove Node
@callback(
    Output('current-graph-store', 'data', allow_duplicate=True),
    Output('status-message-area', 'children', allow_duplicate=True),
    Output('selected-node-store', 'data', allow_duplicate=True),
    Input('remove-node-button', 'n_clicks'),
    State('selected-node-store', 'data'),
    State('current-graph-store', 'data'),
    prevent_initial_call=True
)
def remove_node_callback(n_clicks, selected_node_data, graph_store_data):
    if n_clicks == 0 or not graph_store_data: return no_update, no_update, no_update
    if not selected_node_data or 'id' not in selected_node_data: return no_update, "Select a node to remove.", no_update
    try: node_id_to_remove = int(selected_node_data['id'])
    except (ValueError, TypeError): return no_update, "Invalid selected node ID.", no_update
    print(f"Attempting to remove node: {node_id_to_remove}")
    try:
        num_nodes = graph_store_data['num_nodes']
        if not (0 <= node_id_to_remove < num_nodes): return no_update, f"Invalid node ID: {node_id_to_remove}", no_update
        if num_nodes <= 1: return no_update, "Cannot remove last node.", no_update
        new_graph_data = copy.deepcopy(graph_store_data)
        node_mask = [i != node_id_to_remove for i in range(num_nodes)]
        new_graph_data['features'] = [f for i, f in enumerate(new_graph_data['features']) if node_mask[i]]
        new_graph_data['labels'] = [l for i, l in enumerate(new_graph_data['labels']) if node_mask[i]]
        new_graph_data['train_mask'] = [m for i, m in enumerate(new_graph_data['train_mask']) if node_mask[i]]
        new_graph_data['val_mask'] = [m for i, m in enumerate(new_graph_data['val_mask']) if node_mask[i]]
        new_graph_data['test_mask'] = [m for i, m in enumerate(new_graph_data['test_mask']) if node_mask[i]]
        edge_index = torch.tensor(new_graph_data['edge_index'], dtype=torch.long)
        remapped_edge_index_list = [[], []]
        if edge_index.numel() > 0:
            edge_mask = (edge_index[0] != node_id_to_remove) & (edge_index[1] != node_id_to_remove)
            new_edge_index = edge_index[:, edge_mask]
            if new_edge_index.numel() > 0:
                node_mapping = torch.full((num_nodes,), -1, dtype=torch.long)
                kept_node_indices = torch.arange(num_nodes)[torch.tensor(node_mask, dtype=torch.bool)]
                new_indices = torch.arange(len(kept_node_indices))
                node_mapping[kept_node_indices] = new_indices
                remapped_edge_index = node_mapping[new_edge_index]
                valid_edge_mask = torch.all(remapped_edge_index != -1, dim=0)
                remapped_edge_index = remapped_edge_index[:, valid_edge_mask]
                remapped_edge_index_list = remapped_edge_index.tolist()
        new_graph_data['num_nodes'] = len(new_graph_data['features'])
        new_graph_data['edge_index'] = remapped_edge_index_list
        status_message = f"Removed Node {node_id_to_remove}. New count: {new_graph_data['num_nodes']}"
        print(status_message)
        return new_graph_data, status_message, None
    except Exception as e: error_message = f"Error removing node: {e}"; print(error_message); return no_update, error_message, no_update

# Add Edge
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
    if n_clicks == 0 or not graph_store_data: return no_update, no_update
    if not source_str or not target_str: return no_update, "Enter source and target IDs."
    try: source_id, target_id = int(source_str), int(target_str)
    except (ValueError, TypeError): return no_update, "IDs must be integers."
    print(f"Attempting to add edge: {source_id} -> {target_id}")
    try:
        num_nodes = graph_store_data['num_nodes']
        if not (0 <= source_id < num_nodes and 0 <= target_id < num_nodes): return no_update, f"Invalid ID(s). Max is {num_nodes-1}."
        if source_id == target_id: return no_update, "Self-loops not added via button."
        new_graph_data = copy.deepcopy(graph_store_data)
        edge_index_list = new_graph_data['edge_index']
        if not edge_index_list: edge_index_list = [[], []]
        exists = False
        if edge_index_list[0]:
            edge_set = set(zip(edge_index_list[0], edge_index_list[1]))
            if (source_id, target_id) in edge_set or (target_id, source_id) in edge_set: exists = True
        if exists: return no_update, f"Edge {source_id}-{target_id} already exists."
        edge_index_list[0].extend([source_id, target_id])
        edge_index_list[1].extend([target_id, source_id])
        new_graph_data['edge_index'] = edge_index_list
        status_message = f"Added edge {source_id}-{target_id}."
        print(status_message)
        return new_graph_data, status_message
    except Exception as e: error_message = f"Error adding edge: {e}"; print(error_message); return no_update, error_message

# Remove Edge
@callback(
    Output('current-graph-store', 'data', allow_duplicate=True),
    Output('status-message-area', 'children', allow_duplicate=True),
    Output('selected-edge-store', 'data', allow_duplicate=True),
    Input('remove-edge-button', 'n_clicks'),
    State('new-edge-source', 'value'), State('new-edge-target', 'value'),
    State('selected-edge-store', 'data'), State('current-graph-store', 'data'),
    prevent_initial_call=True
)
def remove_edge_callback(n_clicks, source_input_str, target_input_str, selected_edge_data, graph_store_data):
    if n_clicks == 0 or not graph_store_data: return no_update, no_update, no_update if ctx.triggered_id == 'remove-edge-button' else selected_edge_data
    source_id_to_remove, target_id_to_remove, source_method = None, None, None
    if source_input_str and target_input_str:
        try: source_id_to_remove, target_id_to_remove = int(source_input_str), int(target_input_str); source_method = 'input'
        except (ValueError, TypeError): return no_update, "Invalid Node IDs in input.", selected_edge_data
    elif selected_edge_data and 'source' in selected_edge_data and 'target' in selected_edge_data:
        try: source_id_to_remove, target_id_to_remove = int(selected_edge_data['source']), int(selected_edge_data['target']); source_method = 'selection'
        except (ValueError, TypeError, KeyError): return no_update, "Invalid selected edge data.", None
    else: return no_update, "Specify edge via input or selection.", selected_edge_data
    print(f"Attempting edge removal: {source_id_to_remove}<->{target_id_to_remove} (Method: {source_method})")
    try:
        num_nodes = graph_store_data['num_nodes']
        if not (0 <= source_id_to_remove < num_nodes and 0 <= target_id_to_remove < num_nodes):
             status_message = f"Invalid node ID(s). Max is {num_nodes-1}."
             clear_selection_output = None if source_method == 'selection' else selected_edge_data
             return no_update, status_message, clear_selection_output
        if source_id_to_remove == target_id_to_remove: return no_update, "Cannot remove self-loops.", selected_edge_data
        new_graph_data = copy.deepcopy(graph_store_data)
        edge_index_list = new_graph_data['edge_index']
        if not edge_index_list or not edge_index_list[0]:
              status_message = f"Edge {source_id_to_remove}-{target_id_to_remove} not found (no edges)."
              clear_selection_output = None if source_method == 'selection' else selected_edge_data
              return no_update, status_message, clear_selection_output
        edge_index_tensor = torch.tensor(edge_index_list, dtype=torch.long)
        mask_forward = (edge_index_tensor[0] == source_id_to_remove) & (edge_index_tensor[1] == target_id_to_remove)
        mask_backward = (edge_index_tensor[0] == target_id_to_remove) & (edge_index_tensor[1] == source_id_to_remove)
        remove_mask = mask_forward | mask_backward
        if not torch.any(remove_mask):
            status_message = f"Edge {source_id_to_remove}-{target_id_to_remove} not found."
            clear_selection_output = None if source_method == 'selection' else selected_edge_data
            return no_update, status_message, clear_selection_output
        keep_mask = ~remove_mask
        new_edge_index = edge_index_tensor[:, keep_mask]
        new_graph_data['edge_index'] = new_edge_index.tolist()
        status_message = f"Removed edge {source_id_to_remove}<->{target_id_to_remove}. New edge count: {new_edge_index.shape[1]}"
        print(status_message)
        clear_selection_output = None if source_method == 'selection' else selected_edge_data
        return new_graph_data, status_message, clear_selection_output
    except Exception as e: error_message = f"Error removing edge: {e}"; print(error_message); clear_selection_output = None if source_method == 'selection' else selected_edge_data; return no_update, error_message, clear_selection_output

# --- Selection Callbacks ---
@callback(
    Output('selected-node-store', 'data', allow_duplicate=True),
    Output('selected-edge-store', 'data', allow_duplicate=True),
    Output('status-message-area', 'children', allow_duplicate=True),
    Output('select-node-input', 'value', allow_duplicate=True),
    Input('graph-view', 'tapNodeData'),
    Input('embeddings-view', 'clickData'),
    Input('select-node-button', 'n_clicks'),
    State('select-node-input', 'value'),
    State('current-graph-store', 'data'),
    prevent_initial_call=True
)
def store_selected_node(tapNodeData, clickData, n_clicks_input, input_value, graph_store_data):
    """Handles node selection from graph, embeddings plot, or input box."""
    trigger = ctx.triggered_id
    print(f"Store selected node triggered by: {trigger}")
    selected_node_update, clear_edge_update, status_message, input_reset = no_update, no_update, no_update, no_update
    if trigger == 'graph-view' and tapNodeData and 'id' in tapNodeData:
        selected_node_update = {'id': tapNodeData['id']}; clear_edge_update = None; status_message = f"Selected Node {tapNodeData['id']} from graph."
        print(status_message)
    elif trigger == 'embeddings-view' and clickData and 'points' in clickData and clickData['points']:
        point_info = clickData['points'][0]
        if 'customdata' in point_info and point_info['customdata']:
            selected_node_id = str(point_info['customdata'][0])
            selected_node_update = {'id': selected_node_id}; clear_edge_update = None; status_message = f"Selected Node {selected_node_id} from embeddings plot."
            print(status_message)
    elif trigger == 'select-node-button' and n_clicks_input > 0:
        if input_value is None: status_message = "Enter a Node ID."
        elif graph_store_data is None or graph_store_data.get('num_nodes', 0) == 0: status_message = "Error: Graph data not loaded."
        else:
            num_nodes = graph_store_data['num_nodes']
            try:
                node_id_to_select = int(input_value)
                if 0 <= node_id_to_select < num_nodes:
                    node_id_str = str(node_id_to_select)
                    selected_node_update = {'id': node_id_str}; clear_edge_update = None; status_message = f"Selected Node {node_id_str} via input."; input_reset = None
                    print(status_message)
                else: status_message = f"Error: Node ID {node_id_to_select} out of range (0 to {num_nodes - 1})."
            except (ValueError, TypeError): status_message = f"Error: Invalid input '{input_value}'. Enter integer ID."
            print(status_message)
    return selected_node_update, clear_edge_update, status_message, input_reset

@callback(
    Output('selected-edge-store', 'data', allow_duplicate=True),
    Output('selected-node-store', 'data', allow_duplicate=True),
    Input('graph-view', 'tapEdgeData'),
    prevent_initial_call=True
)
def store_selected_edge_graph(tapEdgeData):
    if tapEdgeData and 'source' in tapEdgeData and 'target' in tapEdgeData:
        print(f"Edge selected: {tapEdgeData.get('id', 'N/A')} ({tapEdgeData['source']} -> {tapEdgeData['target']})")
        return {'id': tapEdgeData.get('id'), 'source': tapEdgeData['source'], 'target': tapEdgeData['target']}, None
    return no_update, no_update

# --- Callback to Run GNNExplainer ---
@callback(
    Output('explanation-store', 'data'),
    Output('status-message-area', 'children', allow_duplicate=True),
    Input('selected-node-store', 'data'),
    Input('current-model-output-store', 'data'),
    State('current-graph-store', 'data'),
    State('model-dropdown', 'value'),
    State('dataset-dropdown', 'value'),
    prevent_initial_call=True
)
def run_gnn_explainer_callback(selected_node_data, model_output, graph_store_data, model_type, dataset_name):
    """Runs GNNExplainer for the selected node."""
    trigger = ctx.triggered_id
    print(f"GNNExplainer callback triggered by: {trigger}")

    if not selected_node_data or 'id' not in selected_node_data: return None, no_update
    if not model_output or model_output.get('predictions') is None: return no_update, no_update # Need predictions for target
    if not graph_store_data or graph_store_data.get('num_nodes', 0) == 0: return None, "Graph data not loaded."

    try:
        selected_node_idx = int(selected_node_data['id'])
        num_nodes = graph_store_data['num_nodes']
        if not (0 <= selected_node_idx < num_nodes): return None, f"Invalid node ID {selected_node_idx}."
    except (ValueError, TypeError): return None, "Invalid node ID format."

    # --- Get Target Prediction ---
    try:
        target_prediction_class = model_output['predictions'][selected_node_idx]
        target = torch.tensor([target_prediction_class], device=device) # Create tensor for target
    except (IndexError, KeyError, TypeError):
        return None, f"Could not get prediction for Node {selected_node_idx} to use as target."


    status_message = f"Running GNNExplainer for Node {selected_node_idx} (Target Class: {target_prediction_class})..."
    print(status_message)

    model = load_model(model_type, dataset_name, device)
    if model is None: return None, f"GNNExplainer failed: Could not load model."

    try:
        x = torch.tensor(graph_store_data['features'], dtype=torch.float).to(device)
        edge_index = torch.tensor(graph_store_data['edge_index'], dtype=torch.long).to(device)
    except Exception as e: return None, f"GNNExplainer failed: Error preparing data tensors - {e}"

    try:
        model.eval()

        # 1. Instantiate the explanation algorithm (GNNExplainer)
        gnn_explainer_algorithm = GNNExplainer(
            epochs=100,
            lr=0.01,
            coeffs={
               "edge_size": 0.05,
               "node_feat_size": 1.0,
               "edge_ent": 1.0,
               "node_feat_ent": 0.1,
            }
        )

        # 2. Configure the main Explainer
        explainer = Explainer(
            model=model,
            algorithm=gnn_explainer_algorithm,
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='log_probs', # ---> FIX: Use plural 'log_probs' <---
            ),
        )

        # 3. Generate the explanation by calling the Explainer instance
        explanation = explainer(x=x, edge_index=edge_index, target=target, index=selected_node_idx)

        print(f"GNNExplainer finished for Node {selected_node_idx}.")
        status_message = f"GNNExplainer explanation generated for Node {selected_node_idx}."

        # Extract masks from the Explanation object
        node_feat_mask = explanation.get('node_feat_mask', None)
        edge_mask = explanation.get('edge_mask', None)

        explanation_data = {
            'node_feat_mask': node_feat_mask.cpu().tolist() if node_feat_mask is not None else None,
            'edge_mask': edge_mask.cpu().tolist() if edge_mask is not None else None
        }
        return explanation_data, status_message

    except Exception as e:
        error_msg = f"Error running GNNExplainer for Node {selected_node_idx}: {e}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        return None, error_msg


# --- Callback to Update ALL Visualizations ---
@callback(
    Output('graph-view', 'elements'),
    Output('embeddings-view', 'figure'),
    Output('feature-histogram-view', 'figure'),
    Output('selected-info-view', 'children'),
    Output('attention-view', 'figure'),
    Output('neighborhood-info-view', 'children'),
    Output('reasoning-output', 'children'),
    Input('current-model-output-store', 'data'),
    Input('selected-node-store', 'data'),
    Input('selected-edge-store', 'data'),
    Input('dim-reduction-dropdown', 'value'),
    Input('explanation-store', 'data'),
    State('current-graph-store', 'data'),
    State('dataset-info-store', 'data'),
    State('model-dropdown', 'value'),
    State('dataset-dropdown', 'value'),
    prevent_initial_call=True
)
def update_visualizations(model_output, selected_node_data, selected_edge_data,
                          dim_reduction_method, explanation_data,
                          graph_store_data, dataset_info, model_type, dataset_name):
    """Updates all visualization components based on current state."""
    trigger = ctx.triggered_id
    print(f"\n--- Debug: update_visualizations triggered by: {trigger} ---")

    empty_elements = []
    empty_fig = go.Figure(layout={'xaxis': {'visible': False}, 'yaxis': {'visible': False}, 'annotations': [{'text': 'N/A'}]})
    default_info_text = "Load data/model, then select a node/edge."
    default_attn_text = "Attention view (GAT only)."
    default_feature_text = "Select node for features. Importance shown after explanation."
    default_neighbor_text = "Select node for neighbors."
    default_reasoning_text = "Select node for GNN explanation."

    attn_fig = go.Figure(layout={'title': default_attn_text, 'xaxis': {'visible': False}, 'yaxis': {'visible': False}})
    feature_fig = go.Figure(layout={'title': default_feature_text, 'xaxis': {'visible': False}, 'yaxis': {'visible': False}})
    neighbor_info_content = default_neighbor_text
    reasoning_content = default_reasoning_text

    if graph_store_data is None or graph_store_data.get('num_nodes', 0) == 0:
        print("Update visualizations skipped: No graph data.")
        return empty_elements, empty_fig, feature_fig, default_info_text, attn_fig, default_neighbor_text, default_reasoning_text

    try:
        current_data = Data(
            x=torch.tensor(graph_store_data['features'], dtype=torch.float),
            edge_index=torch.tensor(graph_store_data['edge_index'], dtype=torch.long),
            y=torch.tensor(graph_store_data['labels'], dtype=torch.long),
            num_nodes=graph_store_data['num_nodes']
        )
    except Exception as e:
        print(f"Error reconstructing graph data: {e}")
        return empty_elements, empty_fig, feature_fig, "Error loading graph.", attn_fig, default_neighbor_text, "Error loading graph."

    class_map = dataset_info.get('class_map', {}) if dataset_info else {}
    selected_node_id_str = selected_node_data.get('id') if selected_node_data else None
    predictions, embeddings, valid_model_output = None, None, False
    if model_output and model_output.get('predictions') is not None and model_output.get('embeddings') is not None:
         try:
             predictions_list, embeddings_list = model_output['predictions'], model_output['embeddings']
             if len(predictions_list) == current_data.num_nodes and len(embeddings_list) == current_data.num_nodes:
                 predictions, embeddings = torch.tensor(predictions_list), torch.tensor(embeddings_list)
                 valid_model_output = True
             else: print("Warning: Model output size mismatch.")
         except Exception as e: print(f"Error processing model output: {e}")

    neighbor_ids, selected_node_idx = [], None
    if selected_node_id_str is not None:
        try:
            node_idx_int = int(selected_node_id_str)
            if 0 <= node_idx_int < current_data.num_nodes:
                selected_node_idx = node_idx_int
                edge_index = current_data.edge_index
                mask_source = edge_index[0] == selected_node_idx; mask_target = edge_index[1] == selected_node_idx
                neighbors_from_source = edge_index[1][mask_source].tolist(); neighbors_from_target = edge_index[0][mask_target].tolist()
                all_neighbors = set(neighbors_from_source + neighbors_from_target)
                all_neighbors.discard(selected_node_idx)
                neighbor_ids = sorted(list(all_neighbors))
                neighbor_details = []
                has_true_labels = hasattr(current_data, 'y') and current_data.y is not None
                has_predictions = predictions is not None
                if neighbor_ids:
                    for neighbor_idx_int in neighbor_ids:
                        if 0 <= neighbor_idx_int < current_data.num_nodes:
                            true_label = "N/A"; pred_label = "N/A"
                            if has_true_labels and neighbor_idx_int < len(current_data.y): n_true_class = int(current_data.y[neighbor_idx_int].item()); true_label = class_map.get(n_true_class, f"C{n_true_class}") if n_true_class != -1 else "N/A"
                            if has_predictions and neighbor_idx_int < len(predictions): n_pred_class = int(predictions[neighbor_idx_int].item()); pred_label = class_map.get(n_pred_class, f"C{n_pred_class}")
                            neighbor_details.append(html.Li(f"Node {neighbor_idx_int}: True={true_label}, Pred={pred_label}"))
                        else: neighbor_details.append(html.Li(f"Node {neighbor_idx_int}: Invalid!"))
                    neighbor_info_content = [html.Strong(f"Neighbors of Node {selected_node_idx} ({len(neighbor_ids)}):"), html.Ul(neighbor_details)]
                else: neighbor_info_content = f"Node {selected_node_idx} has no neighbors."
            else: neighbor_info_content = f"Invalid node ID {selected_node_id_str}."; selected_node_id_str = None; neighbor_ids = []
        except Exception as e: print(f"Error processing neighbors: {e}"); neighbor_info_content = "Error analyzing neighbors."; selected_node_id_str = None; neighbor_ids = []

    explanation_masks, node_feat_mask, edge_mask = None, None, None
    if selected_node_idx is not None and explanation_data and isinstance(explanation_data, dict):
        try:
            node_feat_mask_list = explanation_data.get('node_feat_mask')
            edge_mask_list = explanation_data.get('edge_mask')
            explanation_masks = {}
            if node_feat_mask_list is not None: node_feat_mask = np.array(node_feat_mask_list); explanation_masks['node_feat_mask'] = node_feat_mask
            if edge_mask_list is not None: edge_mask = np.array(edge_mask_list); explanation_masks['edge_mask'] = edge_mask

            reasoning_parts = [html.Strong(f"GNNExplainer Summary for Node {selected_node_idx}:")]
            if node_feat_mask is not None:
                num_features = len(node_feat_mask); top_k_feat = min(5, num_features)
                important_indices = np.argsort(node_feat_mask)[::-1][:top_k_feat]
                feat_summary = ", ".join([f"F_{idx} ({node_feat_mask[idx]:.2f})" for idx in important_indices if node_feat_mask[idx] > 0.01])
                reasoning_parts.append(html.P(f"Top {top_k_feat} Imp. Features: {feat_summary}" if feat_summary else "No features found significantly important."))
            else: reasoning_parts.append(html.P("Feature importance mask N/A."))

            if edge_mask is not None:
                 edge_index_np = current_data.edge_index.cpu().numpy()
                 edge_scores = {}
                 if len(edge_mask) == edge_index_np.shape[1]:
                     for i in range(edge_index_np.shape[1]):
                         u, v = edge_index_np[0, i], edge_index_np[1, i]; score = edge_mask[i]
                         edge_pair = tuple(sorted((u,v))); edge_scores[edge_pair] = max(edge_scores.get(edge_pair, 0.0), score)
                     connected_important_edges = []
                     for (u, v), score in edge_scores.items():
                         if (u == selected_node_idx or v == selected_node_idx) and score > 0.1:
                             neighbor = v if u == selected_node_idx else u
                             connected_important_edges.append((neighbor, score))
                     connected_important_edges.sort(key=lambda item: item[1], reverse=True)
                     top_k_edge = min(5, len(connected_important_edges))
                     if top_k_edge > 0:
                         edge_summary = ", ".join([f"Node {neighbor} ({score:.2f})" for neighbor, score in connected_important_edges[:top_k_edge]])
                         reasoning_parts.append(html.P(f"Top {top_k_edge} Imp. Connections: {edge_summary}"))
                     else: reasoning_parts.append(html.P("No connections found significantly important."))
                 else:
                     reasoning_parts.append(html.P(f"Edge mask length mismatch. Cannot display edge importance."))
                     print(f"Warning: Edge mask length mismatch in update_visualizations.")
                     if 'edge_mask' in explanation_masks: explanation_masks['edge_mask'] = None # Invalidate
            else: reasoning_parts.append(html.P("Edge importance mask N/A."))
            reasoning_content = html.Div(reasoning_parts)
        except Exception as e: print(f"Error processing explanation data: {e}"); reasoning_content = f"Error displaying explanation."; explanation_masks = None
    elif selected_node_idx is not None and trigger == 'explanation-store' and explanation_data is None:
        reasoning_content = f"Could not generate explanation for Node {selected_node_idx}."

    cyto_elements = data_to_cytoscape(current_data, predictions, class_map, selected_node_id_str, selected_edge_data, neighbor_ids=neighbor_ids, explanation_masks=explanation_masks)

    if valid_model_output:
        embeddings_fig = plot_embeddings(embeddings, predictions, current_data.y, class_map, dim_reduction_method, selected_node_id_str)
    else:
        if current_data.num_nodes > 1:
             dummy_embeddings = torch.randn(current_data.num_nodes, 16)
             embeddings_fig = plot_embeddings(dummy_embeddings, None, current_data.y, class_map, dim_reduction_method, selected_node_id_str)
             embeddings_fig.update_layout(title=f'Embeddings ({dim_reduction_method}) - Dummy Data')
        else: embeddings_fig = go.Figure(layout={'title': "Embeddings (N/A)"})

    if selected_node_idx is not None:
        node_features = current_data.x[selected_node_idx]
        current_node_feat_mask = explanation_masks.get('node_feat_mask') if explanation_masks else None
        feature_fig = plot_feature_histogram(node_features, selected_node_id_str, feature_mask=current_node_feat_mask)

    selected_info_content = []
    if selected_node_idx is not None:
        selected_info_content.append(html.Strong(f"Selected Node: {selected_node_idx}"))
        pred_label, true_label = "N/A", "N/A"
        if valid_model_output and selected_node_idx < len(predictions): pred_class = int(predictions[selected_node_idx].item()); pred_label = class_map.get(pred_class, f'Class {pred_class}')
        if hasattr(current_data, 'y') and current_data.y is not None and selected_node_idx < len(current_data.y):
            true_class = int(current_data.y[selected_node_idx].item())
            true_label = class_map.get(true_class, f'Class {true_class}') if true_class != -1 else "N/A (New)"
        selected_info_content.append(html.P(f"Predicted Class: {pred_label}"))
        selected_info_content.append(html.P(f"True Class: {true_label}"))
    elif selected_edge_data: selected_info_content.append(html.Strong(f"Selected Edge: {selected_edge_data.get('source', '?')} -> {selected_edge_data.get('target', '?')}"))
    else: selected_info_content = default_info_text

    if model_type == 'GAT' and selected_node_idx is not None and valid_model_output:
        try:
            gat_model = load_model(model_type, dataset_name, device)
            if gat_model:
                att_data = get_attention_weights(gat_model, current_data, selected_node_idx, device)
                if att_data:
                    neighbor_indices_att, attention_scores = att_data
                    if len(neighbor_indices_att) > 0:
                        att_df = pd.DataFrame({'Neighbor Node': [str(n) for n in neighbor_indices_att], 'Attention Score': attention_scores})
                        att_df = att_df.sort_values('Attention Score', ascending=False).head(20)
                        attn_fig = px.bar(att_df, x='Neighbor Node', y='Attention Score', title=f'Attention Scores to Node {selected_node_idx} (Top 20)', labels={'Attention Score': 'Attention'})
                        attn_fig.update_layout(xaxis_title="Source Neighbor Node", yaxis_title="Attention Score", title_font_size=14)
                    else: attn_fig = go.Figure(layout={'title': f'Node {selected_node_idx} has no incoming attention'})
                else: attn_fig = go.Figure(layout={'title': 'Could not retrieve attention weights.'})
            else: attn_fig = go.Figure(layout={'title': 'GAT Model could not be loaded.'})
        except Exception as e: attn_fig = go.Figure(layout={'title': f'Error getting attention: {e}'}); print(f"Error getting attention: {e}"); import traceback; traceback.print_exc()
    elif model_type != 'GAT': attn_fig = go.Figure(layout={'title': 'Attention view (GAT only).'})

    return cyto_elements, embeddings_fig, feature_fig, selected_info_content, attn_fig, neighbor_info_content, reasoning_content

# --- Run App ---
if __name__ == '__main__':
    print("--- Starting Interactive GNN Explainer ---")
    print(f"Using device: {device}")
    print(f"Models expected in: '{os.path.abspath(MODEL_DIR)}'")
    print(f"Datasets expected in: '{os.path.abspath(DATA_DIR)}'")
    print(f"IMPORTANT: Ensure pre-trained models (.pkl files) exist in '{MODEL_DIR}'.")
    print(f"IMPORTANT: Ensure datasets exist in '{DATA_DIR}'.")
    script_dir = os.path.dirname(__file__) if '__file__' in locals() else '.'
    default_model_path = os.path.join(script_dir, MODEL_DIR, f"{DEFAULT_MODEL}_{DEFAULT_DATASET}.pkl")
    if not os.path.exists(default_model_path): print(f"\nWarning: Default model '{default_model_path}' not found.")
    print("\nDash app starting... Access at http://127.0.0.1:8050/")
    app.run(debug=True, host='0.0.0.0', port=8050)

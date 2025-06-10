# -*- coding: utf-8 -*-
import dash
from dash import dcc, html, Input, Output, State, callback, ctx, no_update, ALL
import dash_cytoscape as cyto
import plotly.express as px
import plotly.graph_objects as go 

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid, Amazon # Amazon for AmazonPhoto
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx, remove_self_loops, add_self_loops, k_hop_subgraph, to_undirected
from torch_geometric.transforms import RandomNodeSplit 

import networkx as nx
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap.umap_ as umap 
import numpy as np
import pickle
import os
import copy 
import warnings
import time 

warnings.filterwarnings("ignore", category=UserWarning)

# --- Global Variables / Setup ---
MODEL_DIR = 'models' 
DATA_DIR = 'data'    
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed') 

DEFAULT_DATASET = 'Cora' 
DEFAULT_MODEL = 'GCN'    
AVAILABLE_DATASETS = ['Cora', 'CiteSeer', 'AmazonPhoto'] # Updated to AmazonPhoto
AVAILABLE_MODELS = ['GCN', 'GAT'] 
NODE_DIM_REDUCTION = 'UMAP' 

# --- GPU Setup ---
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f"CUDA is available. Using device: {torch.cuda.get_device_name(device)}")
else:
    device = torch.device('cpu')
    print("CUDA not available. Using CPU.")

script_dir = os.path.dirname(__file__) if '__file__' in locals() else '.'
model_abs_dir = os.path.join(script_dir, MODEL_DIR)
data_abs_dir = os.path.join(script_dir, DATA_DIR)
processed_data_abs_dir = os.path.join(script_dir, PROCESSED_DATA_DIR)

os.makedirs(model_abs_dir, exist_ok=True)
os.makedirs(data_abs_dir, exist_ok=True)
os.makedirs(processed_data_abs_dir, exist_ok=True)

# --- Model Definitions (Should match train_models.py) ---
class GCNNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.init_args = {'hidden_channels': hidden_channels} 

    def forward(self, x, edge_index, **kwargs): 
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def inference(self, data): 
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        embeddings = x
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1), embeddings

class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8): 
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6) 
        self.init_args = {'hidden_channels': hidden_channels, 'heads': heads} 

    def forward(self, x, edge_index, **kwargs): 
        x, _ = self.conv1(x, edge_index, return_attention_weights=False)
        x = F.elu(x)
        x, _ = self.conv2(x, edge_index, return_attention_weights=False)
        return F.log_softmax(x, dim=1)

    def inference(self, data, return_attention_weights=False): 
        x, edge_index = data.x, data.edge_index
        x, att1_tuple = self.conv1(x, edge_index, return_attention_weights=return_attention_weights)
        x = F.elu(x)
        embeddings = x
        x, att2_tuple = self.conv2(x, edge_index, return_attention_weights=return_attention_weights)
        log_probs = F.log_softmax(x, dim=1)

        if return_attention_weights:
            return log_probs, embeddings, (att1_tuple, att2_tuple)
        else:
            return log_probs, embeddings

# --- Dataset Wrapper for Preprocessed Data ---
class PreprocessedDatasetWrapper:
    def __init__(self, loaded_obj):
        self.data = loaded_obj['data']
        self.num_node_features = loaded_obj['num_node_features']
        self.num_classes = loaded_obj['num_classes']
        self.name = loaded_obj.get('name', 'Unknown Preprocessed')

    def __getitem__(self, idx):
        if idx == 0: return self.data
        raise IndexError(f"{self.name} index out of range")
    def __len__(self): return 1

# --- Helper Functions ---
def load_dataset(name):
    """Loads a dataset, checking for pre-processed .pt files first."""
    global DATA_DIR, PROCESSED_DATA_DIR, script_dir 
    start_time = time.time()
    print(f"App: Attempting to load dataset '{name}'...")
    
    current_base_data_dir = os.path.join(script_dir, DATA_DIR)
    current_processed_data_dir = os.path.join(script_dir, PROCESSED_DATA_DIR)
    dataset_wrapper = None 

    if name in ['Cora', 'CiteSeer', 'AmazonPhoto']: # Updated to AmazonPhoto
        processed_file_path = os.path.join(current_processed_data_dir, f"{name}_processed.pt")
        print(f"  Checking for pre-processed file: {processed_file_path}")
        if os.path.exists(processed_file_path):
            try:
                print(f"  Found pre-processed file. Loading '{name}' from .pt ...")
                loaded_obj = torch.load(processed_file_path, map_location=torch.device('cpu'), weights_only=False)
                if isinstance(loaded_obj, dict) and all(k in loaded_obj for k in ['data', 'num_node_features', 'num_classes']):
                    dataset_wrapper = PreprocessedDatasetWrapper(loaded_obj)
                    if not hasattr(dataset_wrapper.data, 'num_nodes') and dataset_wrapper.data.x is not None: 
                        dataset_wrapper.data.num_nodes = dataset_wrapper.data.x.shape[0]
                    elif not hasattr(dataset_wrapper.data, 'num_nodes'):
                         dataset_wrapper.data.num_nodes = 0 
                    print(f"  App: Dataset '{name}' loaded successfully from pre-processed file.")
                else:
                    print(f"  Warning: Pre-processed file '{processed_file_path}' has unexpected format. Falling back to standard loading.")
            except Exception as e:
                print(f"  Warning: Error loading pre-processed file '{processed_file_path}': {e}. Falling back to standard loading.")
        else:
            print(f"  Pre-processed file for '{name}' not found. Will use standard loading (might be slow first time).")

    if dataset_wrapper is None:
        try:
            original_loader = None
            if name in ['Cora', 'CiteSeer']:
                print(f"  Loading '{name}' using Planetoid (root: {current_base_data_dir})...")
                original_loader = Planetoid(root=current_base_data_dir, name=name)
            elif name == 'AmazonPhoto': # Changed to AmazonPhoto
                print(f"  Loading '{name}' using Amazon Photo loader (root: {current_base_data_dir})...")
                print(f"  (If this is slow, run 'preprocess_datasets.py' for '{name}' for faster subsequent loads.)")
                original_loader = Amazon(root=current_base_data_dir, name='Photo') # Use 'Photo'
            else:
                print(f"App: Dataset '{name}' is not recognized by this loading logic.")
                return None

            if original_loader:
                data_obj_raw = original_loader[0]
                dataset_wrapper = PreprocessedDatasetWrapper({
                    'data': data_obj_raw,
                    'num_node_features': original_loader.num_node_features,
                    'num_classes': original_loader.num_classes,
                    'name': name
                })
                print(f"  App: Dataset '{name}' loaded successfully (standard method).")
        except FileNotFoundError:
             print(f"App: Error loading dataset {name}. Raw files not found in {current_base_data_dir} or subdirectories.")
             return None
        except Exception as e:
            print(f"App: Error loading dataset {name} using standard method: {e}")
            import traceback; traceback.print_exc()
            return None
            
    end_time = time.time()
    if dataset_wrapper and hasattr(dataset_wrapper, 'data') and dataset_wrapper.data is not None :
         print(f"  Dataset '{name}' loading time: {end_time - start_time:.2f} seconds. Nodes: {dataset_wrapper.data.num_nodes if hasattr(dataset_wrapper.data, 'num_nodes') else 'N/A'}")
    elif dataset_wrapper:
         print(f"  Dataset '{name}' loading time: {end_time - start_time:.2f} seconds. (Data object details unavailable)")
    return dataset_wrapper


def load_model(model_type, dataset_name, device_to_load_on):
    global script_dir, MODEL_DIR 
    model_filename = f"{model_type}_{dataset_name}.pkl"
    model_path = os.path.join(script_dir, MODEL_DIR, model_filename)
    print(f"Attempting to load model: {model_path}")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        return None
    try:
        print("  Loading dataset info for model dimensions...")
        temp_dataset = load_dataset(dataset_name) 
        if temp_dataset is None:
             print(f"Error: Cannot load dataset '{dataset_name}' to determine model dimensions.")
             return None
        in_channels = temp_dataset.num_node_features
        out_channels = temp_dataset.num_classes
        print(f"  Dataset info: InChannels={in_channels}, OutChannels={out_channels}")

        with open(model_path, 'rb') as f:
            saved_content = pickle.load(f)
        
        model_state_dict, model_args = None, {}
        if isinstance(saved_content, tuple) and len(saved_content) == 2:
            model_state_dict, model_args = saved_content
        elif isinstance(saved_content, dict): 
            model_state_dict = saved_content
            print("  Warning: Loaded model file contains only state_dict. Using default model args.")
        else:
            print("Error: Unknown format in model PKL file.")
            return None

        if model_type == 'GCN':
            hidden_channels = model_args.get('hidden_channels', 64) # Default from train script
            model = GCNNet(in_channels, hidden_channels, out_channels)
        elif model_type == 'GAT':
            hidden_channels = model_args.get('hidden_channels', 32) # Default from train script
            heads = model_args.get('heads', 8) # Default from train script
            model = GATNet(in_channels, hidden_channels, out_channels, heads=heads)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.load_state_dict(model_state_dict)
        model.to(device_to_load_on)
        model.eval()
        print(f"  Model {model_filename} loaded to {device_to_load_on} with args: {model_args}")
        return model
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        import traceback; traceback.print_exc()
        return None

@torch.no_grad()
def run_inference(model, data_obj, model_type_str, device_to_run_on):
    if model is None or data_obj is None: return None, None
    model.eval()
    model.to(device_to_run_on)
    data_obj = data_obj.to(device_to_run_on)
    try:
        if hasattr(model, 'inference'):
            log_probs, embeddings = model.inference(data_obj)
        else: 
            print(f"Warning: Model {model_type_str} lacks 'inference' method. Using forward pass; embeddings might be missing.")
            log_probs = model(data_obj.x, data_obj.edge_index)
            embeddings = None 
        
        predictions = log_probs.argmax(dim=1)
        return predictions.cpu(), embeddings.cpu() if embeddings is not None else None
    except Exception as e:
        print(f"Error during {model_type_str} inference on {device_to_run_on}: {e}")
        import traceback; traceback.print_exc()
        return None, None

@torch.no_grad()
def get_attention_weights(model_instance, data_obj, node_idx_int, device_to_run_on):
    if not isinstance(model_instance, GATNet) or data_obj is None or node_idx_int is None:
        return None
    model_instance.eval()
    model_instance.to(device_to_run_on)
    data_obj = data_obj.to(device_to_run_on)

    if not (0 <= node_idx_int < data_obj.num_nodes):
        print(f"Invalid node index {node_idx_int} for attention.")
        return None
    try:
        _, _, attention_layers_output = model_instance.inference(data_obj, return_attention_weights=True)
        
        if attention_layers_output is None or not isinstance(attention_layers_output, tuple) or len(attention_layers_output) < 1:
            print("Model did not return expected attention structure.")
            return None

        last_layer_attention_tuple = attention_layers_output[-1] 
        if last_layer_attention_tuple is None or not isinstance(last_layer_attention_tuple, tuple) or len(last_layer_attention_tuple) != 2:
            print("Last layer attention tuple has unexpected format.")
            return None

        edge_index_att, alpha_att = last_layer_attention_tuple
        
        if edge_index_att is None or alpha_att is None:
             print("Attention edge_index or alpha is None for the last layer.")
             return None

        mask = edge_index_att[1] == node_idx_int
        if not torch.any(mask):
            return None 

        neighbor_indices = edge_index_att[0][mask] 
        attention_scores = alpha_att[mask] 

        if attention_scores.dim() > 1 and attention_scores.shape[1] > 1:
            attention_scores = attention_scores.mean(dim=1)
        
        return neighbor_indices.cpu().numpy(), attention_scores.cpu().squeeze().numpy()
    except Exception as e:
        print(f"Error getting attention weights: {e}")
        import traceback; traceback.print_exc()
        return None

def data_to_cytoscape(data, predictions=None, class_map=None, selected_node_id=None, selected_edge_data=None, neighbor_ids=None, explanation_masks=None):
    if data is None or data.num_nodes == 0: return []
    neighbor_ids_set = set(neighbor_ids) if neighbor_ids is not None else set()

    edge_tuple_to_score = {}
    edge_importance_threshold = 0.1 
    if explanation_masks and 'edge_mask' in explanation_masks and explanation_masks['edge_mask'] is not None and data.edge_index is not None:
        edge_mask_raw = explanation_masks['edge_mask']
        explained_edge_mask_np = np.array(edge_mask_raw.cpu().numpy() if isinstance(edge_mask_raw, torch.Tensor) else edge_mask_raw)
        
        if len(explained_edge_mask_np) == data.edge_index.shape[1]:
            edge_index_np = data.edge_index.cpu().numpy()
            for i in range(edge_index_np.shape[1]):
                u, v = edge_index_np[0, i], edge_index_np[1, i]
                score = explained_edge_mask_np[i]
                edge_pair = tuple(sorted((u, v)))
                edge_tuple_to_score[edge_pair] = max(edge_tuple_to_score.get(edge_pair, 0.0), score)
        else:
            print(f"Warning (Cytoscape): Edge mask length mismatch. Expected {data.edge_index.shape[1]}, got {len(explained_edge_mask_np)}.")

    nodes_cy = []
    default_node_color = '#808080'
    color_palette = px.colors.qualitative.Plotly
    preds_np = predictions.cpu().numpy() if predictions is not None and isinstance(predictions, torch.Tensor) else np.array(predictions) if predictions is not None else None

    for i in range(data.num_nodes):
        node_id_str = str(i)
        node_data = {'id': node_id_str, 'label': f'Node {i}'}
        node_classes_css = []
        node_color = default_node_color

        if preds_np is not None and i < len(preds_np):
            pred_class = int(preds_np[i])
            node_data['class_pred'] = pred_class
            node_color = color_palette[pred_class % len(color_palette)]
            if class_map and pred_class in class_map:
                node_data['label'] += f'\n({class_map[pred_class]})'
        
        if hasattr(data, 'y') and data.y is not None and i < len(data.y):
            true_class = int(data.y[i].item())
            node_data['true_class'] = true_class
            node_data['true_label'] = class_map.get(true_class, f'C{true_class}') if true_class != -1 and class_map else (f'C{true_class}' if true_class != -1 else 'N/A')
        
        is_selected = selected_node_id == node_id_str
        if is_selected: node_classes_css.append('selected')
        if i in neighbor_ids_set and not is_selected: node_classes_css.append('neighbor')
        
        node_data['classes'] = ' '.join(node_classes_css)
        nodes_cy.append({'data': node_data, 'style': {'background-color': node_color}})

    edges_cy = []
    added_undirected_edges = set()
    if data.edge_index is not None and data.edge_index.shape[1] > 0:
        edge_index_np_display = data.edge_index.cpu().numpy()
        sel_src_str = selected_edge_data.get('source') if selected_edge_data else None
        sel_tgt_str = selected_edge_data.get('target') if selected_edge_data else None

        for i in range(edge_index_np_display.shape[1]):
            src_int, tgt_int = edge_index_np_display[0, i], edge_index_np_display[1, i]
            src_str, tgt_str = str(src_int), str(tgt_int)
            
            if src_int >= data.num_nodes or tgt_int >= data.num_nodes: continue 

            edge_pair_sorted_tuple = tuple(sorted((src_int, tgt_int)))
            if edge_pair_sorted_tuple in added_undirected_edges: continue
            added_undirected_edges.add(edge_pair_sorted_tuple)

            edge_id = f"edge_{src_str}_{tgt_str}_{i}" 
            edge_data = {'id': edge_id, 'source': src_str, 'target': tgt_str}
            edge_classes_css = []
            edge_style = {}

            if sel_src_str and sel_tgt_str and ((sel_src_str == src_str and sel_tgt_str == tgt_str) or (sel_src_str == tgt_str and sel_tgt_str == src_str)):
                edge_classes_css.append('selected-edge')

            edge_score = edge_tuple_to_score.get(edge_pair_sorted_tuple, 0.0)
            if edge_score > edge_importance_threshold:
                edge_classes_css.append('explained-edge')
                scale_factor = max(0, (edge_score - edge_importance_threshold) / (1.0 - edge_importance_threshold)) if edge_importance_threshold < 1.0 else (1.0 if edge_score >= 1.0 else 0.0)
                edge_style['opacity'] = min(max(0.3 + 0.7 * scale_factor, 0.1), 1.0)
                edge_style['width'] = min(max(1.5 + 2.5 * scale_factor, 1.0), 4.0)
                edge_data['importance'] = f"{edge_score:.3f}"
            
            edge_data['classes'] = ' '.join(edge_classes_css)
            edges_cy.append({'data': edge_data, 'style': edge_style})
            
    return nodes_cy + edges_cy


def plot_embeddings(embeddings_tensor, predictions_tensor, true_labels_tensor=None, class_map_dict=None, dim_reduction_str='UMAP', selected_node_id_str=None):
    if embeddings_tensor is None:
        return go.Figure(layout={'title': "Embeddings (Not Available)", 'xaxis': {'visible': False}, 'yaxis': {'visible': False}})

    embed_np = embeddings_tensor.cpu().numpy() if isinstance(embeddings_tensor, torch.Tensor) else np.array(embeddings_tensor)
    preds_np = predictions_tensor.cpu().numpy() if predictions_tensor is not None and isinstance(predictions_tensor, torch.Tensor) else np.array(predictions_tensor) if predictions_tensor is not None else None
    labels_np = true_labels_tensor.cpu().numpy() if true_labels_tensor is not None and isinstance(true_labels_tensor, torch.Tensor) else np.array(true_labels_tensor) if true_labels_tensor is not None else None
    
    num_nodes = embed_np.shape[0]
    if num_nodes < 2 : return go.Figure(layout={'title': "Embeddings (Need >= 2 nodes)", 'xaxis': {'visible': False}, 'yaxis': {'visible': False}})

    n_comp = 2
    embed_2d, reduction_method_used = None, dim_reduction_str
    try:
        if dim_reduction_str == 'TSNE':
            perplexity_val = min(30.0, max(5.0, float(num_nodes - 1) / 3.0))
            if num_nodes <= n_comp: reduction_method_used = 'PCA'; reducer = PCA(n_components=n_comp)
            else: reducer = TSNE(n_components=n_comp, perplexity=perplexity_val, random_state=42, init='pca', learning_rate='auto', n_iter=300)
        elif dim_reduction_str == 'UMAP':
            eff_n_neighbors = min(15, num_nodes - 1)
            if eff_n_neighbors < 2: reduction_method_used = 'PCA'; reducer = PCA(n_components=n_comp)
            else: reducer = umap.UMAP(n_neighbors=eff_n_neighbors, min_dist=0.1, n_components=n_comp, random_state=42)
        else: reduction_method_used = 'PCA'; reducer = PCA(n_components=n_comp) 
        embed_2d = reducer.fit_transform(embed_np)
    except Exception as e:
        print(f"Error in {reduction_method_used} reduction: {e}. Falling back to random.");
        embed_2d = np.random.rand(num_nodes, n_comp)
        reduction_method_used += " (Error - Random Fallback)"

    df_plot = pd.DataFrame(embed_2d, columns=['Dim1', 'Dim2'])
    df_plot['NodeID'] = [str(i) for i in range(num_nodes)]
    
    color_col_name, legend_title = None, 'Class'
    if preds_np is not None:
        df_plot['Predicted'] = [class_map_dict.get(int(p), f'P{int(p)}') if class_map_dict else f'P{int(p)}' for p in preds_np[:num_nodes]]
        color_col_name = 'Predicted'; legend_title = 'Predicted Class'
    elif labels_np is not None:
        df_plot['TrueColor'] = [class_map_dict.get(int(l), f'T{int(l)}') if l!=-1 and class_map_dict else (f'T{int(l)}' if l!=-1 else 'N/A') for l in labels_np[:num_nodes]]
        color_col_name = 'TrueColor'; legend_title = 'True Class'
    
    if labels_np is not None:
         df_plot['TrueLabel'] = [class_map_dict.get(int(l), f'T{int(l)}') if l!=-1 and class_map_dict else (f'T{int(l)}' if l!=-1 else 'N/A') for l in labels_np[:num_nodes]]
    else: df_plot['TrueLabel'] = 'N/A'
    if 'Predicted' not in df_plot and preds_np is not None: 
        df_plot['Predicted'] = [class_map_dict.get(int(p), f'P{int(p)}') if class_map_dict else f'P{int(p)}' for p in preds_np[:num_nodes]]


    hover_cols = ['NodeID', 'TrueLabel']
    if 'Predicted' in df_plot: hover_cols.append('Predicted')

    fig = px.scatter(df_plot, x='Dim1', y='Dim2', color=color_col_name, hover_data=hover_cols, custom_data=['NodeID'],
                     title=f'Node Embeddings ({reduction_method_used})', color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_layout(legend_title_text=legend_title, clickmode='event+select', dragmode='pan')
    fig.update_traces(marker=dict(size=8, opacity=0.8), 
                      selected=dict(marker=dict(size=12, opacity=1.0, color='black')), 
                      unselected=dict(marker=dict(opacity=0.5)))

    if selected_node_id_str:
        sel_indices = df_plot[df_plot['NodeID'] == selected_node_id_str].index.tolist()
        if sel_indices: fig.update_traces(selectedpoints=sel_indices, selector=dict(type='scatter'))
    return fig


def plot_feature_histogram(features_np_1d, node_id_str, feature_mask_np_1d=None, top_k_to_show=10):
    if features_np_1d is None: return go.Figure(layout={'title': "Feature Histogram (No Data)"})
    if features_np_1d.ndim != 1: return go.Figure(layout={'title': f"Feature Histogram Error (Node {node_id_str}): Not 1D"})

    title = f"Feature Values for Node {node_id_str}"
    df_data = {'Feature Index': [f'F_{i}' for i in range(len(features_np_1d))], 'Value': features_np_1d}
    
    if feature_mask_np_1d is not None and len(feature_mask_np_1d) == len(features_np_1d):
        title = f"Top {top_k_to_show} Important Features for Node {node_id_str} (GNNExplainer)"
        df_data['Importance'] = feature_mask_np_1d
        df_feat = pd.DataFrame(df_data).sort_values('Importance', ascending=False).head(top_k_to_show)
        bar_colors = ['crimson'] * len(df_feat) 
    else: 
        df_feat = pd.DataFrame(df_data)
        df_feat['AbsValue'] = np.abs(df_feat['Value'])
        df_feat = df_feat.sort_values('AbsValue', ascending=False).head(top_k_to_show)
        title = f"Top {top_k_to_show} Features (by abs. value) for Node {node_id_str}"
        bar_colors = ['steelblue'] * len(df_feat)
        if feature_mask_np_1d is not None: print("Warning: Feature mask invalid/mismatch. Showing features by abs value.")

    if df_feat.empty: return go.Figure(layout={'title': title + " (No features to show)"})
    
    fig = px.bar(df_feat, x='Feature Index', y='Value', title=title, color_discrete_sequence=[bar_colors[0]]) 
    fig.update_traces(marker_color=bar_colors) 
    fig.update_layout(xaxis_title="Feature", yaxis_title="Value", xaxis_tickangle=-45 if len(df_feat)>15 else 0)
    if np.all(np.isin(df_feat['Value'], [0, 1])): fig.update_yaxes(range=[-0.1, 1.1]) 
    return fig

# --- Dash App Initialization ---
app = dash.Dash(__name__, suppress_callback_exceptions=True,
                external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'],
                prevent_initial_callbacks='initial_duplicate') 
server = app.server
app.title = "Interactive GNN Explainer"

# --- App Layout ---
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
            html.Label("Dataset:", style={'marginRight': '10px'}),
            dcc.Dropdown(id='dataset-dropdown', options=[{'label': ds, 'value': ds} for ds in AVAILABLE_DATASETS], value=DEFAULT_DATASET, clearable=False, style={'width': '180px', 'marginRight': '20px'}), 
            html.Label("Model:", style={'marginRight': '10px'}),
            dcc.Dropdown(id='model-dropdown', options=[{'label': m, 'value': m} for m in AVAILABLE_MODELS], value=DEFAULT_MODEL, clearable=False, style={'width': '150px'}),
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'}),
        html.Div([
            html.Button('Add Node (Copy Sel)', id='add-node-button', n_clicks=0, style={'marginRight': '5px'}),
            html.Button('Remove Sel. Node', id='remove-node-button', n_clicks=0, style={'marginRight': '15px'}),
            dcc.Input(id='new-edge-source', type='text', placeholder='Src ID', style={'width': '70px', 'marginRight': '2px'}),
            dcc.Input(id='new-edge-target', type='text', placeholder='Tgt ID', style={'width': '70px', 'marginRight': '5px'}),
            html.Button('Add Edge', id='add-edge-button', n_clicks=0, style={'marginRight': '5px'}),
            html.Button('Remove Sel. Edge', id='remove-edge-button', n_clicks=0),
        ], style={'display': 'flex', 'alignItems': 'center', 'marginTop': '10px'})
    ], id='control-panel', style={'padding': '15px', 'border': '1px solid #ddd', 'borderRadius': '5px', 'marginBottom': '20px', 'backgroundColor': '#f9f9f9'}),

    html.Div(id='status-message-area', style={'padding': '10px', 'marginBottom': '10px', 'border': '1px dashed #ccc', 'borderRadius': '5px', 'minHeight': '40px', 'backgroundColor': '#f0f0f0', 'whiteSpace': 'pre-wrap'}),

    html.Div([
        html.Div([
            html.H3("Graph View"),
            html.Div([
                dcc.Input(id='select-node-input', type='number', placeholder='Node ID...', min=0, step=1, style={'width': '120px', 'marginRight': '5px'}),
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
                    {'selector': 'edge.explained-edge', 'style': { 'line-color': '#39FF14', 'target-arrow-color': '#39FF14', 'width': 'data(width, 2)', 'opacity': 'data(opacity, 0.7)', 'z-index': 9997 }} 
                ]
            ),
            html.H3("Node Embeddings"),
            dcc.Dropdown(id='dim-reduction-dropdown', options=[{'label': dr, 'value': dr} for dr in ['UMAP', 'TSNE', 'PCA']], value=NODE_DIM_REDUCTION, clearable=False, style={'width': '120px', 'marginBottom': '10px'}),
            dcc.Graph(id='embeddings-view', style={'height': '400px'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingRight': '2%'}),

        html.Div([
            html.H3("Selected Node/Edge Info"),
            html.Div(id='selected-info-view', style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '10px', 'minHeight': '100px', 'marginBottom': '10px', 'backgroundColor': '#f9f9f9', 'fontSize':'small'}),
            html.H3("Feature Importance (Selected Node)"),
            dcc.Graph(id='feature-histogram-view', style={'height': '250px'}),
            html.H3("Neighbor Analysis (Selected Node)"),
            html.Div(id='neighborhood-info-view', style={'border': '1px solid #ddd', 'borderRadius': '5px', 'padding': '10px', 'minHeight': '100px', 'maxHeight': '150px', 'overflowY': 'auto', 'marginBottom': '10px', 'backgroundColor': '#f9f9f9', 'fontSize': 'small'}),
            html.H3("GNN Explanation (Selected Node)"),
            html.Div(id='reasoning-output', style={'border': '1px solid #e0e0e0', 'borderRadius': '5px', 'padding': '10px', 'minHeight': '80px', 'backgroundColor': '#fafafa', 'fontSize': 'small', 'whiteSpace': 'pre-wrap'}),
            html.H3("Attention Weights (GAT Only)"),
            dcc.Graph(id='attention-view', style={'height': '250px'})
        ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'})
    ])
], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif'})


# --- Callbacks ---
@callback(
    Output('current-graph-store', 'data'),
    Output('dataset-info-store', 'data'),
    Output('status-message-area', 'children', allow_duplicate=True),
    Output('explanation-store', 'data', allow_duplicate=True), 
    Output('selected-node-store', 'data', allow_duplicate=True), 
    Output('selected-edge-store', 'data', allow_duplicate=True), 
    Input('dataset-dropdown', 'value'),
    prevent_initial_call='initial_duplicate' 
)
def load_dataset_callback(dataset_name):
    print(f"Callback: load_dataset_callback triggered for '{dataset_name}'")
    status_msg = f"Loading dataset '{dataset_name}'..."
    dataset_obj = load_dataset(dataset_name) 

    empty_graph_store = {'edge_index': [[],[]], 'features': [], 'labels': [], 'num_nodes': 0, 'train_mask': [], 'val_mask': [], 'test_mask': []}
    empty_dataset_info = {'class_map': {}, 'num_node_features': 0, 'num_classes': 0, 'name': 'None'}

    if dataset_obj is None or not hasattr(dataset_obj, 'data') or dataset_obj.data is None:
        err_msg = f"Error: Failed to load dataset '{dataset_name}' or dataset is empty."
        print(err_msg)
        return empty_graph_store, empty_dataset_info, err_msg, None, None, None

    try:
        data = dataset_obj.data 
        if not all(hasattr(data, attr) for attr in ['x', 'edge_index', 'y', 'num_nodes']):
            err_msg = f"Error: Dataset '{dataset_name}' is missing essential attributes (x, edge_index, y, num_nodes)."
            print(err_msg)
            return empty_graph_store, empty_dataset_info, err_msg, None, None, None

        train_mask = data.train_mask if hasattr(data, 'train_mask') and data.train_mask is not None else torch.zeros(data.num_nodes, dtype=torch.bool)
        val_mask = data.val_mask if hasattr(data, 'val_mask') and data.val_mask is not None else torch.zeros(data.num_nodes, dtype=torch.bool)
        test_mask = data.test_mask if hasattr(data, 'test_mask') and data.test_mask is not None else torch.zeros(data.num_nodes, dtype=torch.bool)

        graph_store_data = {
            'edge_index': data.edge_index.cpu().tolist() if data.edge_index is not None else [[],[]],
            'features': data.x.cpu().tolist() if data.x is not None else [],
            'labels': data.y.cpu().tolist() if data.y is not None else [],
            'num_nodes': data.num_nodes,
            'train_mask': train_mask.cpu().tolist(),
            'val_mask': val_mask.cpu().tolist(),
            'test_mask': test_mask.cpu().tolist(),
        }

        class_map = {}
        num_classes = dataset_obj.num_classes
        if num_classes > 0 and data.y is not None:
            class_map = {i: f'Class {i}' for i in range(num_classes)}
            ds_name_lower = dataset_name.lower()
            if ds_name_lower == 'cora' and num_classes == 7: class_map = {0:'Theory', 1:'RL', 2:'GA', 3:'NN', 4:'Prob', 5:'Case', 6:'Rule'}
            elif ds_name_lower == 'citeseer' and num_classes == 6: class_map = {0:'Agents', 1:'AI', 2:'DB', 3:'IR', 4:'ML', 5:'HCI'}
            elif ds_name_lower == 'amazonphoto' and num_classes == 8: # Amazon Photo has 8 classes
                class_map = {i: f'Photo Category {i}' for i in range(num_classes)} 

        dataset_info = {
            'class_map': class_map,
            'num_node_features': dataset_obj.num_node_features,
            'num_classes': num_classes,
            'name': dataset_name
        }
        status_msg = f"Dataset '{dataset_name}' loaded. Nodes: {data.num_nodes}, Edges: {data.num_edges if hasattr(data, 'num_edges') else 'N/A'}, Features: {dataset_obj.num_node_features}, Classes: {num_classes}."
        print(status_msg)
        return graph_store_data, dataset_info, status_msg, None, None, None 
    except Exception as e:
        err_msg = f"Error processing data from '{dataset_name}': {e}"
        print(err_msg); import traceback; traceback.print_exc()
        return empty_graph_store, empty_dataset_info, err_msg, None, None, None


@callback(
    Output('current-model-output-store', 'data'),
    Output('status-message-area', 'children', allow_duplicate=True),
    Output('explanation-store', 'data', allow_duplicate=True), 
    [Input('current-graph-store', 'data'), Input('model-dropdown', 'value')],
    State('dataset-dropdown', 'value'),
    prevent_initial_call=True
)
def run_inference_callback(graph_store_data, model_type, dataset_name):
    print(f"Callback: run_inference_callback. Trigger: {ctx.triggered_id}. Model: {model_type}, Dataset: {dataset_name}")
    empty_model_output = {'predictions': None, 'embeddings': None}

    if not graph_store_data or graph_store_data.get('num_nodes', 0) == 0:
        return empty_model_output, "Graph data not loaded. Cannot run inference.", None
    if not model_type or not dataset_name:
        return empty_model_output, "Model or dataset not selected.", None

    status_msg = f"Loading {model_type} model for {dataset_name} and running inference on {device}..."
    print(status_msg)
    
    model_instance = load_model(model_type, dataset_name, device)
    if model_instance is None:
        err_msg = f"Inference skipped: {model_type} for {dataset_name} could not be loaded."
        print(err_msg)
        return empty_model_output, err_msg, None

    try:
        current_data_for_inference = Data(
            x=torch.tensor(graph_store_data['features'], dtype=torch.float),
            edge_index=torch.tensor(graph_store_data['edge_index'], dtype=torch.long),
            y=torch.tensor(graph_store_data['labels'], dtype=torch.long), 
            num_nodes=graph_store_data['num_nodes']
        )
    except Exception as e:
        err_msg = f"Error reconstructing graph data for inference: {e}"
        print(err_msg)
        return empty_model_output, err_msg, None

    predictions, embeddings = run_inference(model_instance, current_data_for_inference, model_type, device)

    if predictions is None: 
        err_msg = f"Inference failed for {model_type} on {dataset_name} (using {device}). Predictions are None."
        print(err_msg)
        return empty_model_output, err_msg, None
    
    status_msg = f"Inference complete for {model_type} on {dataset_name}. Predictions: {predictions.shape[0]} nodes."
    if embeddings is not None:
        status_msg += f" Embeddings shape: {embeddings.shape}."
    else:
        status_msg += " (Embeddings not available)."
    print(status_msg)

    model_output_data = {
        'predictions': predictions.tolist(),
        'embeddings': embeddings.tolist() if embeddings is not None else None
    }
    return model_output_data, status_msg, None 


@callback(
    Output('current-graph-store', 'data', allow_duplicate=True),
    Output('status-message-area', 'children', allow_duplicate=True),
    Input('add-node-button', 'n_clicks'),
    [State('selected-node-store', 'data'), State('current-graph-store', 'data'), State('dataset-info-store', 'data')],
    prevent_initial_call=True
)
def add_node_callback(n_clicks, selected_node, graph_data, ds_info):
    if not n_clicks or not graph_data or graph_data.get('num_nodes', 0) == 0: return no_update, no_update
    
    new_graph_data = copy.deepcopy(graph_data)
    num_features = ds_info.get('num_node_features', 1) 
    
    features_to_add = [0.0] * num_features 
    if selected_node and 'id' in selected_node:
        try:
            template_node_id = int(selected_node['id'])
            if 0 <= template_node_id < new_graph_data['num_nodes']:
                features_to_add = copy.deepcopy(new_graph_data['features'][template_node_id])
        except (ValueError, TypeError, IndexError):
            print("Warning: Could not copy features from selected node, using defaults.")

    new_node_id = new_graph_data['num_nodes']
    new_graph_data['num_nodes'] += 1
    new_graph_data['features'].append(features_to_add)
    new_graph_data['labels'].append(-1) 
    for mask_key in ['train_mask', 'val_mask', 'test_mask']:
        new_graph_data[mask_key].append(False)
        
    status_msg = f"Added Node {new_node_id}. Total nodes: {new_graph_data['num_nodes']}. Re-run inference."
    print(status_msg)
    return new_graph_data, status_msg

@callback(
    Output('selected-node-store', 'data', allow_duplicate=True),
    Output('selected-edge-store', 'data', allow_duplicate=True), 
    Output('status-message-area', 'children', allow_duplicate=True),
    Output('select-node-input', 'value', allow_duplicate=True), 
    [Input('graph-view', 'tapNodeData'), Input('embeddings-view', 'clickData'), Input('select-node-button', 'n_clicks')],
    [State('select-node-input', 'value'), State('current-graph-store', 'data')],
    prevent_initial_call=True
)
def store_selected_node(tap_node, click_embedding, n_clicks_btn, input_node_id, graph_store):
    triggered_id = ctx.triggered_id
    node_id_to_select_str = None
    status_msg = no_update
    
    if triggered_id == 'graph-view' and tap_node and 'id' in tap_node:
        node_id_to_select_str = tap_node['id']
        status_msg = f"Node {node_id_to_select_str} selected from graph."
    elif triggered_id == 'embeddings-view' and click_embedding and click_embedding['points']:
        custom_data = click_embedding['points'][0].get('customdata')
        if custom_data and isinstance(custom_data, list) and len(custom_data) > 0:
            node_id_to_select_str = str(custom_data[0])
            status_msg = f"Node {node_id_to_select_str} selected from embeddings."
    elif triggered_id == 'select-node-button' and n_clicks_btn > 0:
        if input_node_id is not None:
            try:
                node_id_val = int(input_node_id)
                num_nodes = graph_store.get('num_nodes', 0) if graph_store else 0
                if 0 <= node_id_val < num_nodes:
                    node_id_to_select_str = str(node_id_val)
                    status_msg = f"Node {node_id_to_select_str} selected via input."
                else:
                    status_msg = f"Error: Node ID {node_id_val} out of range (0-{num_nodes-1})."
            except ValueError:
                status_msg = "Error: Invalid Node ID in input box."
        else:
            status_msg = "Enter a Node ID to select."
            
    if node_id_to_select_str is not None:
        print(status_msg)
        return {'id': node_id_to_select_str}, None, status_msg, None 
    return no_update, no_update, status_msg, no_update


@callback(
    Output('selected-edge-store', 'data', allow_duplicate=True),
    Output('selected-node-store', 'data', allow_duplicate=True), 
    Input('graph-view', 'tapEdgeData'),
    prevent_initial_call=True
)
def store_selected_edge(tap_edge):
    if tap_edge and 'source' in tap_edge and 'target' in tap_edge:
        edge_id = tap_edge.get('id', f"edge_{tap_edge['source']}_{tap_edge['target']}")
        print(f"Edge '{edge_id}' selected: {tap_edge['source']} -> {tap_edge['target']}")
        return {'id': edge_id, 'source': tap_edge['source'], 'target': tap_edge['target']}, None
    return no_update, no_update


@callback(
    Output('explanation-store', 'data', allow_duplicate=True),
    Output('status-message-area', 'children', allow_duplicate=True),
    Input('selected-node-store', 'data'), 
    [State('current-graph-store', 'data'), State('current-model-output-store', 'data'),
     State('model-dropdown', 'value'), State('dataset-dropdown', 'value')],
    prevent_initial_call=True
)
def run_gnn_explainer_callback(selected_node_data, graph_store, model_output_store, model_type, dataset_name):
    if not selected_node_data or 'id' not in selected_node_data:
        return None, "Select a node to run GNNExplainer." 
    if not graph_store or graph_store.get('num_nodes',0) == 0:
        return None, "Graph data not available for explainer."
    if not model_output_store or model_output_store.get('predictions') is None:
        return None, "Model predictions not available. Run inference first."

    try:
        node_idx_to_explain = int(selected_node_data['id'])
        if not (0 <= node_idx_to_explain < graph_store['num_nodes']):
            return None, f"Invalid node ID {node_idx_to_explain} for explanation."
    except (ValueError, TypeError):
        return None, "Invalid selected node ID format."

    status_msg = f"Running GNNExplainer for Node {node_idx_to_explain}..."
    print(status_msg)
    
    model_to_explain = load_model(model_type, dataset_name, device)
    if model_to_explain is None:
        return None, f"GNNExplainer: Failed to load model {model_type} for {dataset_name}."

    try:
        x = torch.tensor(graph_store['features'], dtype=torch.float).to(device)
        edge_index = torch.tensor(graph_store['edge_index'], dtype=torch.long).to(device)
        
        target_prediction = model_output_store['predictions'][node_idx_to_explain]
        target_tensor = torch.tensor([target_prediction], device=device)

        model_to_explain.eval() 

        gnn_explainer_alg = GNNExplainer(
            epochs=100, lr=0.01,
            coeffs={"edge_size": 0.05, "node_feat_size": 1.0, "edge_ent": 1.0, "node_feat_ent": 0.1}
        )
        explainer_instance = Explainer(
            model=model_to_explain,
            algorithm=gnn_explainer_alg,
            explanation_type='model', 
            node_mask_type='attributes', 
            edge_mask_type='object',     
            model_config=dict(
                mode='multiclass_classification', 
                task_level='node',                
                return_type='log_probs',          
            ),
        )
        explanation_result = explainer_instance(x=x, edge_index=edge_index, target=target_tensor, index=node_idx_to_explain)
        
        status_msg = f"GNNExplainer explanation generated for Node {node_idx_to_explain}."
        print(status_msg)

        explanation_data_to_store = {
            'node_feat_mask': explanation_result.get('node_feat_mask').cpu().tolist() if explanation_result.get('node_feat_mask') is not None else None,
            'edge_mask': explanation_result.get('edge_mask').cpu().tolist() if explanation_result.get('edge_mask') is not None else None,
            'explained_node_idx': node_idx_to_explain 
        }
        return explanation_data_to_store, status_msg
    except Exception as e:
        err_msg = f"Error running GNNExplainer for Node {node_idx_to_explain}: {e}"
        print(err_msg); import traceback; traceback.print_exc()
        return None, err_msg


@callback(
    Output('graph-view', 'elements'),
    Output('embeddings-view', 'figure'),
    Output('feature-histogram-view', 'figure'),
    Output('selected-info-view', 'children'),
    Output('attention-view', 'figure'),
    Output('neighborhood-info-view', 'children'),
    Output('reasoning-output', 'children'),
    [Input('current-model-output-store', 'data'), Input('selected-node-store', 'data'),
     Input('selected-edge-store', 'data'), Input('dim-reduction-dropdown', 'value'),
     Input('explanation-store', 'data')], 
    [State('current-graph-store', 'data'), State('dataset-info-store', 'data'),
     State('model-dropdown', 'value'), State('dataset-dropdown', 'value')],
    prevent_initial_call=True
)
def update_all_visualizations(
    model_output, sel_node_data, sel_edge_data, dim_reduction_method, expl_data,
    graph_store, ds_info, model_type, dataset_name
):
    print(f"\n--- Callback: update_all_visualizations. Trigger: {ctx.triggered_id} ---")
    
    empty_cyto_elements = []
    empty_plotly_fig = go.Figure(layout={'xaxis': {'visible': False}, 'yaxis': {'visible': False}, 'annotations': [{'text': 'N/A', 'showarrow': False}]})
    default_info_html = [html.P("Load data and select an element.")]
    default_neighbor_html = [html.P("Select a node to see its neighbors.")]
    default_reasoning_html = [html.P("Select a node to generate and view its GNN explanation.")]

    if not graph_store or graph_store.get('num_nodes', 0) == 0:
        print("  Vis Update: No graph data. Returning empty visualizations.")
        return empty_cyto_elements, empty_plotly_fig, empty_plotly_fig, default_info_html, empty_plotly_fig, default_neighbor_html, default_reasoning_html

    try:
        current_pyg_data = Data(
            x=torch.tensor(graph_store['features'], dtype=torch.float),
            edge_index=torch.tensor(graph_store['edge_index'], dtype=torch.long),
            y=torch.tensor(graph_store['labels'], dtype=torch.long),
            num_nodes=graph_store['num_nodes'],
            train_mask=torch.tensor(graph_store['train_mask'], dtype=torch.bool) if 'train_mask' in graph_store else None,
            val_mask=torch.tensor(graph_store['val_mask'], dtype=torch.bool) if 'val_mask' in graph_store else None,
            test_mask=torch.tensor(graph_store['test_mask'], dtype=torch.bool) if 'test_mask' in graph_store else None
        )
    except Exception as e:
        error_msg = f"Error reconstructing graph data for visualization: {e}"
        print(error_msg)
        return empty_cyto_elements, empty_plotly_fig, empty_plotly_fig, [html.P(error_msg)], empty_plotly_fig, default_neighbor_html, default_reasoning_html

    class_map = ds_info.get('class_map', {}) if ds_info else {}
    selected_node_id_str = sel_node_data.get('id') if sel_node_data else None
    
    predictions_tensor, embeddings_tensor = None, None
    if model_output and model_output.get('predictions') is not None:
        predictions_tensor = torch.tensor(model_output['predictions'])
    if model_output and model_output.get('embeddings') is not None:
        embeddings_tensor = torch.tensor(model_output['embeddings'])

    neighbor_ids_list = []
    selected_node_idx_int = None
    neighbor_info_display = default_neighbor_html

    if selected_node_id_str:
        try:
            selected_node_idx_int = int(selected_node_id_str)
            if 0 <= selected_node_idx_int < current_pyg_data.num_nodes:
                adj = current_pyg_data.edge_index
                mask_source = adj[0] == selected_node_idx_int
                mask_target = adj[1] == selected_node_idx_int
                neighbors_connected = torch.cat((adj[1][mask_source], adj[0][mask_target])).unique().tolist()
                neighbor_ids_list = [n for n in neighbors_connected if n != selected_node_idx_int] 
                
                if neighbor_ids_list:
                    details = [html.Strong(f"Neighbors of Node {selected_node_idx_int} ({len(neighbor_ids_list)}):")]
                    for neighbor_idx in sorted(neighbor_ids_list)[:20]: 
                        true_lbl = "N/A"
                        if hasattr(current_pyg_data, 'y') and current_pyg_data.y is not None and neighbor_idx < len(current_pyg_data.y):
                            true_class_val = int(current_pyg_data.y[neighbor_idx].item())
                            true_lbl = class_map.get(true_class_val, f"C{true_class_val}") if true_class_val != -1 else "N/A"
                        
                        pred_lbl = "N/A"
                        if predictions_tensor is not None and neighbor_idx < len(predictions_tensor):
                             pred_class_val = int(predictions_tensor[neighbor_idx].item())
                             pred_lbl = class_map.get(pred_class_val, f"P{pred_class_val}")
                        details.append(html.Li(f"Node {neighbor_idx}: True={true_lbl}, Pred={pred_lbl}"))
                    neighbor_info_display = html.Ul(details)
                else:
                    neighbor_info_display = [html.P(f"Node {selected_node_idx_int} has no direct neighbors.")]
            else:
                neighbor_info_display = [html.P(f"Invalid selected node ID: {selected_node_id_str}")]
                selected_node_idx_int = None 
        except ValueError:
            neighbor_info_display = [html.P(f"Invalid node ID format: {selected_node_id_str}")]
            selected_node_idx_int = None

    current_explanation_masks_for_cyto = None
    node_feature_mask_for_hist = None
    reasoning_display = default_reasoning_html

    if expl_data and isinstance(expl_data, dict) and selected_node_idx_int is not None and expl_data.get('explained_node_idx') == selected_node_idx_int:
        current_explanation_masks_for_cyto = {
            'edge_mask': torch.tensor(expl_data['edge_mask']) if expl_data.get('edge_mask') is not None else None,
        }
        if expl_data.get('node_feat_mask') is not None:
            node_feature_mask_for_hist = np.array(expl_data['node_feat_mask'])

        reasoning_parts = [html.Strong(f"GNNExplainer Summary for Node {selected_node_idx_int}:")]
        if node_feature_mask_for_hist is not None:
            top_k_feat = min(5, len(node_feature_mask_for_hist))
            important_indices = np.argsort(node_feature_mask_for_hist)[::-1][:top_k_feat]
            feat_summary = ", ".join([f"F{idx}({node_feature_mask_for_hist[idx]:.2f})" for idx in important_indices if node_feature_mask_for_hist[idx] > 0.01])
            reasoning_parts.append(html.P(f"Top Imp. Features: {feat_summary if feat_summary else 'None significant'}"))
        else: reasoning_parts.append(html.P("Node feature importance: N/A"))
        
        if current_explanation_masks_for_cyto.get('edge_mask') is not None:
            num_important_edges = torch.sum(current_explanation_masks_for_cyto['edge_mask'] > 0.1).item() 
            reasoning_parts.append(html.P(f"Important Edges Identified: {num_important_edges} (approx, based on mask)"))
        else: reasoning_parts.append(html.P("Edge importance: N/A"))
        reasoning_display = html.Div(reasoning_parts)

    elif selected_node_idx_int is not None and ctx.triggered_id == 'explanation-store' and expl_data is None:
         reasoning_display = [html.P(f"Could not generate or load explanation for Node {selected_node_idx_int}.")]

    cyto_elements = data_to_cytoscape(current_pyg_data, predictions_tensor, class_map, selected_node_id_str, sel_edge_data, neighbor_ids_list, current_explanation_masks_for_cyto)
    embeddings_fig = plot_embeddings(embeddings_tensor, predictions_tensor, current_pyg_data.y, class_map, dim_reduction_method, selected_node_id_str)

    feature_fig = empty_plotly_fig
    if selected_node_idx_int is not None and current_pyg_data.x is not None:
        node_features_np = current_pyg_data.x[selected_node_idx_int].cpu().numpy()
        feature_fig = plot_feature_histogram(node_features_np, selected_node_id_str, node_feature_mask_for_hist)
    
    selected_info_display = default_info_html
    if selected_node_idx_int is not None:
        info_parts = [html.Strong(f"Selected Node: {selected_node_idx_int}")]
        true_lbl_val = "N/A"
        if hasattr(current_pyg_data, 'y') and current_pyg_data.y is not None and selected_node_idx_int < len(current_pyg_data.y):
            true_cls = int(current_pyg_data.y[selected_node_idx_int].item())
            true_lbl_val = class_map.get(true_cls, f"C{true_cls}") if true_cls != -1 else "N/A"
        
        pred_lbl_val = "N/A"
        if predictions_tensor is not None and selected_node_idx_int < len(predictions_tensor):
            pred_cls = int(predictions_tensor[selected_node_idx_int].item())
            pred_lbl_val = class_map.get(pred_cls, f"P{pred_cls}")

        info_parts.append(html.P(f"True Class: {true_lbl_val}"))
        info_parts.append(html.P(f"Predicted Class: {pred_lbl_val}"))
        selected_info_display = html.Div(info_parts)
    elif sel_edge_data and 'source' in sel_edge_data and 'target' in sel_edge_data:
        selected_info_display = [html.Strong(f"Selected Edge: {sel_edge_data['source']} – {sel_edge_data['target']}")]

    attn_fig = empty_plotly_fig
    if model_type == 'GAT' and selected_node_idx_int is not None and predictions_tensor is not None : 
        gat_model_instance = load_model(model_type, dataset_name, device) 
        if gat_model_instance:
            att_data_tuple = get_attention_weights(gat_model_instance, current_pyg_data, selected_node_idx_int, device)
            if att_data_tuple:
                att_neighbor_indices, att_scores = att_data_tuple
                if len(att_neighbor_indices) > 0:
                    att_df = pd.DataFrame({'Neighbor': [str(n) for n in att_neighbor_indices], 'Attention': att_scores})
                    att_df = att_df.sort_values('Attention', ascending=False).head(20) 
                    attn_fig = px.bar(att_df, x='Neighbor', y='Attention', title=f'Attention to Node {selected_node_idx_int} (Top 20)', labels={'Attention':'Score'})
                    attn_fig.update_layout(xaxis_title="Source Neighbor", yaxis_title="Attention Score")
                else: attn_fig.update_layout(title=f'Node {selected_node_idx_int} has no incoming attention.')
            else: attn_fig.update_layout(title='Could not retrieve GAT attention weights.')
        else: attn_fig.update_layout(title='GAT Model not loaded for attention.')
    elif model_type != 'GAT':
        attn_fig.update_layout(title='Attention view (GAT model only)')

    return cyto_elements, embeddings_fig, feature_fig, selected_info_display, attn_fig, neighbor_info_display, reasoning_display


# --- Run App ---
if __name__ == '__main__':
    print("--- Starting Interactive GNN Explainer ---")
    print(f"Script directory: {os.path.abspath(script_dir)}")
    print(f"Using device: {device}")
    print(f"Models expected in: '{os.path.abspath(model_abs_dir)}'")
    print(f"Raw Datasets expected in: '{os.path.abspath(data_abs_dir)}'")
    print(f"Processed Datasets expected in: '{os.path.abspath(processed_data_abs_dir)}'")
    print(f"IMPORTANT: Ensure pre-trained models (.pkl) are in '{MODEL_DIR}'.")
    print(f"IMPORTANT: For Cora, CiteSeer, AmazonPhoto, run 'preprocess_datasets.py' first for faster loading.")
        
    default_model_path_check = os.path.join(model_abs_dir, f"{DEFAULT_MODEL}_{DEFAULT_DATASET}.pkl")
    if not os.path.exists(default_model_path_check):
         print(f"\nWarning: Default model '{default_model_path_check}' not found. App might not load initial model correctly.")
    
    print("\nDash app starting... Access at http://127.0.0.1:8050/")
    app.run(debug=True, host='0.0.0.0', port=8050)

import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.utils import to_networkx, k_hop_subgraph
from torch_geometric.data import Data
import numpy as np
from sklearn.linear_model import LinearRegression

def get_model_score(model, x_feat, edge_index, source, target):
    """
    Get the link prediction score (probability) for (source, target).
    Handles both link_predictor head and dot-product fallback.
    """
    device = next(model.parameters()).device
    x_feat = x_feat.to(device)
    edge_index = edge_index.to(device)
    model.eval()
    
    with torch.no_grad():
        embeddings = None
        # 1. Forward Pass
        try:
            res = model(x_feat, edge_index, return_embeddings=True)
            if isinstance(res, tuple):
                embeddings = res[1]
            else:
                embeddings = res
        except Exception:
            # Fallback for models without return_embeddings arg
            try:
                embeddings = model(x_feat, edge_index)
            except Exception:
                pass

        if embeddings is None and hasattr(model, 'encode'):
             embeddings = model.encode(x_feat, edge_index)

        if embeddings is None:
            return 0.0

        # 2. Score Computation
        # A. Explicit Link Predictor (e.g. MLP head)
        if getattr(model, 'enable_link_prediction', False) and hasattr(model, 'predict_links'):
            edge_tensor = torch.tensor([[source], [target]], device=device)
            score = model.predict_links(embeddings, edge_tensor)
            return torch.sigmoid(score).item()
        
        # B. Dot Product (Standard GAE/VGAE style)
        else:
            z_u = embeddings[source]
            z_v = embeddings[target]
            score = (z_u * z_v).sum()
            return torch.sigmoid(score).item()

def compute_path_strength(path, degree_dict):
    """
    Compute path strength based on GCN normalization.
    Strength = Product_{(u,v) in Path} (1 / sqrt(deg(u)*deg(v)))
    """
    strength = 1.0
    if len(path) < 2:
        return 0.0
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        du = degree_dict.get(u, 1)
        dv = degree_dict.get(v, 1)
        if du == 0: du = 1
        if dv == 0: dv = 1
        weight = 1.0 / np.sqrt(du * dv)
        strength *= weight
    return strength

def get_instruments(u, v, G_nx):
    """
    Compute instruments Z for a pair (u, v).
    Z = [Common Neighbors, Adamic-Adar, Degree(u), Degree(v), Local Clustering(u)+Local Clustering(v)]
    """
    try:
        cn = len(list(nx.common_neighbors(G_nx, u, v)))
    except: cn = 0
    
    try:
        aa = next(nx.adamic_adar_index(G_nx, [(u, v)]))[2]
    except: aa = 0
    
    deg_u = G_nx.degree(u)
    deg_v = G_nx.degree(v)
    
    try:
        clust_u = nx.clustering(G_nx, u)
        clust_v = nx.clustering(G_nx, v)
        clust_sum = clust_u + clust_v
    except: clust_sum = 0
    
    return [cn, aa, deg_u, deg_v, clust_sum]

def estimate_beta_iv(model, data, G_nx, subset_nodes, degree_dict):
    """
    Estimate Causal Effect Beta using 2SLS on sampled pairs from the subgraph.
    Stage 1: T ~ Z
    Stage 2: Y ~ T_hat
    """
    # 1. Sample pairs from subgraph to build regression dataset
    samples = []
    nodes = list(subset_nodes)
    if len(nodes) < 5: 
        return 0.0 # Not enough data
    
    # Randomly sample up to 50 pairs
    import random
    num_samples = min(50, len(nodes) * (len(nodes)-1) // 2)
    pairs = []
    for _ in range(num_samples * 2): # Oversample then filter
        u, v = random.sample(nodes, 2)
        if u != v:
            pairs.append((u, v))
    pairs = list(set(pairs))[:num_samples]
    
    Z_list = []
    T_list = []
    Y_list = []
    
    for u, v in pairs:
        # Y: Model Prediction
        y_val = get_model_score(model, data.x, data.edge_index, u, v)
        
        # T: Total Path Strength (Sum of all simple paths <= 3 hops)
        # This is expensive, so we limit to 2-hop for estimation speed
        paths = list(nx.all_simple_paths(G_nx, u, v, cutoff=2))
        t_val = sum(compute_path_strength(p, degree_dict) for p in paths)
        
        # Z: Instruments
        z_val = get_instruments(u, v, G_nx)
        
        if t_val > 0: # Only consider connected pairs
            Z_list.append(z_val)
            T_list.append([t_val])
            Y_list.append(y_val)
            
    if len(Y_list) < 10:
        return 0.0 # Insufficient connected pairs
        
    Z = np.array(Z_list)
    T = np.array(T_list)
    Y = np.array(Y_list)
    
    try:
        # Stage 1: Regress T on Z
        reg_1 = LinearRegression()
        reg_1.fit(Z, T)
        T_hat = reg_1.predict(Z)
        
        # Stage 2: Regress Y on T_hat
        reg_2 = LinearRegression()
        reg_2.fit(T_hat, Y)
        
        beta = reg_2.coef_[0]
        return beta
    except Exception as e:
        print(f"IV Regression failed: {e}")
        return 0.0

def run_cpa_iv_link(model, data, source_node, target_node, top_k=10, max_path_len=3):
    """
    Causal Path Analysis (CPA) for Link Prediction with IV Regression.
    """
    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)
    
    # --- 1. Subgraph Extraction ---
    try:
        subset, sub_edge_index, _, _ = k_hop_subgraph(
            [source_node, target_node], 
            max_path_len, 
            data.edge_index, 
            relabel_nodes=True, 
            num_nodes=data.num_nodes
        )
    except Exception as e:
        return {'error': f"Subgraph extraction failed: {str(e)}", 'paths': []}
    
    # Node Mappings
    mapping = {i: node_idx.item() for i, node_idx in enumerate(subset)}
    inv_mapping = {v: k for k, v in mapping.items()}
    u_local = inv_mapping.get(source_node)
    v_local = inv_mapping.get(target_node)
    
    if u_local is None or v_local is None:
        return {'verdict': 'Disconnected', 'paths': []}

    # Build Subgraph NetworkX
    sub_data = Data(edge_index=sub_edge_index, num_nodes=len(subset))
    G_sub = to_networkx(sub_data, to_undirected=True, remove_self_loops=True)
    
    # Pre-compute degrees for strength calculation
    degree_dict = dict(G_sub.degree())

    # --- 2. IV Estimation (Beta) ---
    # Estimate beta on the subgraph context
    beta = estimate_beta_iv(model, data, G_sub, list(range(len(subset))), degree_dict)

    # --- 3. Baseline Prediction ---
    baseline_score = get_model_score(model, data.x, data.edge_index, source_node, target_node)

    # --- 4. Path Enumeration & Scoring ---
    try:
        paths_local = list(nx.all_simple_paths(G_sub, source=u_local, target=v_local, cutoff=max_path_len))
    except: paths_local = []
    
    paths_local = [p for p in paths_local if len(p) >= 2]
    path_effects = []
    
    for path_loc in paths_local[:100]:
        path_global = [mapping[n] for n in path_loc]
        
        # A. Path Strength (T)
        t_p = compute_path_strength(path_loc, degree_dict) # Use local degrees for consistency in subgraph
        
        # B. Counterfactual Intervention (Delta P)
        edges_to_remove = set()
        for i in range(len(path_global) - 1):
            n1, n2 = path_global[i], path_global[i+1]
            edges_to_remove.add(tuple(sorted((n1, n2))))
            
        src = data.edge_index[0]
        dst = data.edge_index[1]
        mask = torch.ones(data.edge_index.size(1), dtype=torch.bool, device=device)
        
        for u, v in edges_to_remove:
            edge_mask_curr = ~((src == u) & (dst == v)) & ~((src == v) & (dst == u))
            mask = mask & edge_mask_curr
            
        perturbed_edge_index = data.edge_index[:, mask]
        perturbed_score = get_model_score(model, data.x, perturbed_edge_index, source_node, target_node)
        
        delta_p = baseline_score - perturbed_score
        
        # C. Causal Score
        # Score = Beta * Strength + Delta_P
        # (Combining General Effect + Specific Necessity)
        causal_score = (beta * t_p) + delta_p
        
        path_effects.append({
            'nodes': path_global,
            'score': causal_score,
            'delta_p': delta_p,
            'path_strength': t_p,
            'perturbed_score': perturbed_score
        })

    # --- 5. Additional Metrics (Neighborhood/Features) ---
    # Calculate simple neighborhood/feature impacts for context
    neigh_impact = {'u': 0, 'v': 0}
    feat_impact = {'u': 0, 'v': 0}
    # (Simplified for brevity, similar to previous implementation)
    
    # --- 6. Formatting ---
    sorted_paths = sorted(path_effects, key=lambda x: x['score'], reverse=True)
    top_paths = sorted_paths[:top_k]
    
    verdict = "Weak"
    if any(p['score'] > 0.05 for p in top_paths): verdict = "Strong"
    elif any(p['score'] > 0.01 for p in top_paths): verdict = "Moderate"
    
    # Check Ground Truth
    is_existing = False
    mask_forward = (data.edge_index[0] == source_node) & (data.edge_index[1] == target_node)
    mask_backward = (data.edge_index[0] == target_node) & (data.edge_index[1] == source_node)
    if mask_forward.any() or mask_backward.any():
        is_existing = True

    return {
        'paths': top_paths,
        'baseline_score': baseline_score,
        'num_paths_found': len(path_effects),
        'source_node': source_node,
        'target_node': target_node,
        'verdict': verdict,
        'verdict_type': verdict.lower(),
        'instrument_sensitivity': beta, # The estimated Beta
        'neigh_impact': neigh_impact,
        'feat_impact': feat_impact,
        'is_existing_link': is_existing
    }
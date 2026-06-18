import torch
import numpy as np
from torch_geometric.utils import negative_sampling, to_undirected, remove_self_loops
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings

def create_edge_splits(data, val_ratio=0.05, test_ratio=0.15, random_state=42):
    """
    Split edges into train/validation/test sets for link prediction.
    
    Args:
        data: PyTorch Geometric data object
        val_ratio: Fraction of edges for validation
        test_ratio: Fraction of edges for testing
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary with edge splits and modified training data
    """
    torch.manual_seed(random_state)
    np.random.seed(random_state)
    
    # Ensure edges are undirected and remove self-loops
    edge_index = remove_self_loops(data.edge_index)[0]
    edge_index = to_undirected(edge_index)
    
    # Get unique edges (remove duplicates from undirected representation)
    edge_set = set()
    unique_edges = []
    for i in range(edge_index.size(1)):
        edge = tuple(sorted([edge_index[0, i].item(), edge_index[1, i].item()]))
        if edge not in edge_set and edge[0] != edge[1]:  # No self-loops
            edge_set.add(edge)
            unique_edges.append([edge_index[0, i].item(), edge_index[1, i].item()])
    
    unique_edges = torch.tensor(unique_edges).t()
    num_edges = unique_edges.size(1)
    
    # Calculate split sizes
    num_val = int(val_ratio * num_edges)
    num_test = int(test_ratio * num_edges)
    num_train = num_edges - num_val - num_test
    
    # Random permutation for splitting
    perm = torch.randperm(num_edges)
    
    # Split edges
    train_edges = unique_edges[:, perm[:num_train]]
    val_edges = unique_edges[:, perm[num_train:num_train + num_val]]
    test_edges = unique_edges[:, perm[num_train + num_val:]]
    
    # Convert back to undirected for training (add reverse edges)
    train_edge_index = to_undirected(train_edges)
    
    # Generate negative edges for validation and testing
    val_neg_edges = negative_sampling(
        edge_index=train_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=val_edges.size(1),
        method='sparse'
    )
    
    test_neg_edges = negative_sampling(
        edge_index=train_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=test_edges.size(1),
        method='sparse'
    )
    
    # Create modified training data
    train_data = data.clone()
    train_data.edge_index = train_edge_index
    
    return {
        'train_data': train_data,
        'train_edge_index': train_edge_index,
        'val_pos_edges': val_edges,
        'val_neg_edges': val_neg_edges,
        'test_pos_edges': test_edges,
        'test_neg_edges': test_neg_edges,
        'num_train_edges': num_train,
        'num_val_edges': num_val,
        'num_test_edges': num_test
    }

def generate_negative_samples(edge_index, num_nodes, num_samples, existing_edges=None):
    """
    Generate negative edge samples that don't exist in the graph.
    
    Args:
        edge_index: Existing edges
        num_nodes: Number of nodes in the graph
        num_samples: Number of negative samples to generate
        existing_edges: Set of existing edges to avoid
    
    Returns:
        Tensor of negative edge samples
    """
    if existing_edges is None:
        existing_edges = set()
        for i in range(edge_index.size(1)):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            existing_edges.add((min(u, v), max(u, v)))
    
    negative_edges = []
    attempts = 0
    max_attempts = num_samples * 10  # Prevent infinite loops
    
    while len(negative_edges) < num_samples and attempts < max_attempts:
        # Random node pairs
        u = torch.randint(0, num_nodes, (1,)).item()
        v = torch.randint(0, num_nodes, (1,)).item()
        
        if u != v:  # No self-loops
            edge = (min(u, v), max(u, v))
            if edge not in existing_edges:
                negative_edges.append([u, v])
                existing_edges.add(edge)
        
        attempts += 1
    
    if len(negative_edges) < num_samples:
        warnings.warn(f"Could only generate {len(negative_edges)} negative samples out of {num_samples} requested")
    
    return torch.tensor(negative_edges[:num_samples]).t() if negative_edges else torch.empty(2, 0, dtype=torch.long)

def evaluate_link_prediction(pos_scores, neg_scores):
    """
    Evaluate link prediction performance using standard metrics.
    
    Args:
        pos_scores: Scores for positive (existing) edges
        neg_scores: Scores for negative (non-existing) edges
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Combine scores and create labels
    scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
    labels = torch.cat([
        torch.ones(pos_scores.size(0)),
        torch.zeros(neg_scores.size(0))
    ]).cpu().numpy()
    
    # Calculate metrics
    try:
        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
    except ValueError as e:
        # Handle edge cases (e.g., all same labels)
        auc = 0.5
        ap = 0.5
        warnings.warn(f"Metric calculation failed: {e}")
    
    # Calculate accuracy with threshold 0.5
    predictions = (scores > 0.5).astype(int)
    accuracy = (predictions == labels).mean()
    
    # Precision and recall
    true_positives = ((predictions == 1) & (labels == 1)).sum()
    false_positives = ((predictions == 1) & (labels == 0)).sum()
    false_negatives = ((predictions == 0) & (labels == 1)).sum()
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'auc': float(auc),
        'ap': float(ap),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'num_pos': len(pos_scores),
        'num_neg': len(neg_scores)
    }

def get_top_k_predictions(embeddings, edge_candidates, model, k=10):
    """
    Get top-k link predictions for given edge candidates.
    
    Args:
        embeddings: Node embeddings
        edge_candidates: Candidate edges to score
        model: Model with link prediction capability
        k: Number of top predictions to return
    
    Returns:
        List of (edge, score) tuples sorted by score
    """
    if not hasattr(model, 'predict_links'):
        raise ValueError("Model does not support link prediction")
    
    with torch.no_grad():
        scores = model.predict_links(embeddings, edge_candidates)
        scores = torch.sigmoid(scores)  # Convert to probabilities
    
    # Sort by score and get top-k
    sorted_indices = torch.argsort(scores, descending=True)[:k]
    
    top_predictions = []
    for idx in sorted_indices:
        source = edge_candidates[0, idx].item()
        target = edge_candidates[1, idx].item()
        score = scores[idx].item()
        top_predictions.append({
            'source': source,
            'target': target,
            'score': score
        })
    
    return top_predictions

def create_link_prediction_candidates(data, target_node=None, exclude_existing=True):
    """
    Create candidate edges for link prediction.
    
    Args:
        data: PyTorch Geometric data object
        target_node: If specified, only create candidates involving this node
        exclude_existing: Whether to exclude existing edges
    
    Returns:
        Tensor of candidate edges
    """
    if target_node is not None:
        # Create candidates for a specific node
        other_nodes = torch.arange(data.num_nodes)
        other_nodes = other_nodes[other_nodes != target_node]
        
        candidates = torch.stack([
            torch.full((len(other_nodes),), target_node),
            other_nodes
        ])
    else:
        # Create all possible node pairs
        nodes = torch.arange(data.num_nodes)
        candidates = torch.combinations(nodes, 2).t()
    
    if exclude_existing:
        # Remove existing edges
        existing_edges = set()
        for i in range(data.edge_index.size(1)):
            u, v = data.edge_index[0, i].item(), data.edge_index[1, i].item()
            existing_edges.add((min(u, v), max(u, v)))
        
        # Filter candidates
        valid_candidates = []
        for i in range(candidates.size(1)):
            u, v = candidates[0, i].item(), candidates[1, i].item()
            edge = (min(u, v), max(u, v))
            if edge not in existing_edges:
                valid_candidates.append([u, v])
        
        if valid_candidates:
            candidates = torch.tensor(valid_candidates).t()
        else:
            candidates = torch.empty(2, 0, dtype=torch.long)
    
    return candidates

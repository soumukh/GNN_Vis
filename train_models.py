# train_models.py (Fixed and Optimized)
import os
import pickle
import torch
import torch.nn.functional as F
from torch.optim import Adam, lr_scheduler
import numpy as np
import copy
from tqdm import tqdm
import logging
from logging.handlers import RotatingFileHandler

# PyTorch Geometric imports
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from node2vec import Node2Vec
import networkx as nx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('/home/sougatam/visualization/optimize/training.log', maxBytes=1e6, backupCount=3),
        logging.StreamHandler()
    ]
)

# Configuration
CONFIG = {
    'MODEL_DIR': 'models',
    'DATA_DIR': 'data',
    'EPOCHS': 300,
    'LEARNING_RATE': 0.005,
    'WEIGHT_DECAY': 1e-4,
    'HIDDEN_CHANNELS_GCN': 64,
    'HIDDEN_CHANNELS_GAT': 32,
    'GAT_HEADS': 8,
    'PATIENCE': 15,
    'BATCH_SIZE': 4096,
    'NODE2VEC_PARAMS': {
        'dimensions': 32,
        'walk_length': 10,
        'num_walks': 50,
        'workers': 8,
        'quiet': True
    },
    'DATASETS': ['Cora', 'CiteSeer', 'PubMed', 'Jazz'],
    'MODELS': ['GCN', 'GAT']
}

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_GPUS = torch.cuda.device_count()
torch.backends.cudnn.benchmark = True

class GCNNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.init_args = {'hidden_channels': hidden_channels}

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        embeddings = x
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1), embeddings

class GATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.3)
        self.conv2 = GATConv(hidden_channels*heads, out_channels, heads=1, concat=False, dropout=0.3)
        self.init_args = {'hidden_channels': hidden_channels, 'heads': heads}

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        embeddings = x
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1), embeddings

class JazzDataset:
    def __init__(self):
        self._load_data()
        
    def _load_data(self):
        data_path = os.path.join(CONFIG['/home/sougatam/jazz'], 'Jazz')
        os.makedirs(data_path, exist_ok=True)
        
        edge_file = os.path.join(data_path, 'jazz.cites')
        feature_file = os.path.join(data_path, 'jazz.content')
        
            
        edges = np.loadtxt(edge_file, dtype=int, delimiter='\t')
        content = np.loadtxt(feature_file, dtype=float)
        
        # Node2Vec embeddings
        G = nx.from_edgelist(edges)
        node2vec = Node2Vec(G, **CONFIG['NODE2VEC_PARAMS'])
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        embeddings = np.array([model.wv[str(n)] for n in sorted(G.nodes())])
        
        features = np.hstack((content[:, 1:-1], embeddings))
        
        self.data = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=torch.tensor(edges.T, dtype=torch.long),
            y=torch.tensor(content[:, -1].astype(int), dtype=torch.long)
        )
        
        num_nodes = self.data.num_nodes
        perm = torch.randperm(num_nodes)
        self.data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        split = [0.8, 0.1, 0.1]
        sections = [int(num_nodes*split[0]), int(num_nodes*(split[0]+split[1]))]
        self.data.train_mask[perm[:sections[0]]] = True
        self.data.val_mask[perm[sections[0]:sections[1]]] = True
        self.data.test_mask[perm[sections[1]:]] = True


class Trainer:
    def __init__(self, model, data):
        if NUM_GPUS > 1:
            self.model = torch.nn.DataParallel(model).to(DEVICE)
        else:
            self.model = model.to(DEVICE)
            
        self.data = data.to(DEVICE)
        self.optimizer = Adam(self.model.parameters(), 
                            lr=CONFIG['LEARNING_RATE'], 
                            weight_decay=CONFIG['WEIGHT_DECAY'])
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', 
                                                      patience=CONFIG['PATIENCE']//2)
        self.best_weights = None

    def train_epoch(self):
        self.model.train()
        self.optimizer.zero_grad()
        out, _ = self.model(self.data)
        loss = F.nll_loss(out[self.data.train_mask], 
                        self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, mask):
        self.model.eval()
        out, _ = self.model(self.data)
        pred = out[mask].argmax(dim=1)
        return (pred == self.data.y[mask]).float().mean().item()

    def run(self):
        best_acc = 0
        no_improve = 0
        progress = tqdm(range(CONFIG['EPOCHS']), desc="Training")
        
        for epoch in progress:
            loss = self.train_epoch()
            train_acc = self.evaluate(self.data.train_mask)
            val_acc = self.evaluate(self.data.val_mask)
            self.scheduler.step(val_acc)
            
            progress.set_postfix_str(
                f"Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}"
            )
            
            if val_acc > best_acc:
                best_acc = val_acc
                no_improve = 0
                self.best_weights = copy.deepcopy(self.model.state_dict())
            else:
                no_improve += 1
                if no_improve >= CONFIG['PATIENCE']:
                    logging.info(f"Early stopping at epoch {epoch}")
                    break

        self.model.load_state_dict(self.best_weights)
        test_acc = self.evaluate(self.data.test_mask)
        return test_acc

def train_model(model_type, dataset_name):
    logging.info(f"\n{'='*40}\nTraining {model_type} on {dataset_name}\n{'='*40}")
    
    try:
        if dataset_name == 'Jazz':
            dataset = JazzDataset().data
        else:
            dataset = Planetoid(root=os.path.join(CONFIG['DATA_DIR'], dataset_name), 
                              name=dataset_name)[0]
            
        in_channels = dataset.num_node_features
        out_channels = len(torch.unique(dataset.y))
        
        if model_type == 'GCN':
            model = GCNNet(in_channels, CONFIG['HIDDEN_CHANNELS_GCN'], out_channels)
        elif model_type == 'GAT':
            model = GATNet(in_channels, CONFIG['HIDDEN_CHANNELS_GAT'], out_channels, 
                         CONFIG['GAT_HEADS'])
        
        trainer = Trainer(model, dataset)
        test_acc = trainer.run()
        
        os.makedirs(CONFIG['MODEL_DIR'], exist_ok=True)
        model_path = f"{CONFIG['MODEL_DIR']}/{model_type}_{dataset_name}.pkl"
        
        model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
        state = {
            'state_dict': model_to_save.state_dict(),
            'init_args': model_to_save.init_args,
            'test_acc': test_acc
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            
        logging.info(f"Model saved to {model_path} | Test Accuracy: {test_acc:.4f}")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}", exc_info=True)

if __name__ == '__main__':
    logging.info(f"Using {NUM_GPUS} Tesla T4 GPUs")
    
    # Create required directories
    os.makedirs(CONFIG['DATA_DIR'], exist_ok=True)
    os.makedirs(CONFIG['MODEL_DIR'], exist_ok=True)
    
    for dataset in CONFIG['DATASETS']:
        for model_type in CONFIG['MODELS']:
            try:
                train_model(model_type, dataset)
            except Exception as e:
                logging.error(f"Failed {model_type} on {dataset}: {str(e)}")
                continue

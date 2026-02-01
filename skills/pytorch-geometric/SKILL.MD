---
name: pytorch-geometric
description: Graph Neural Networks (GNN) for learning on graph-structured data. PyTorch Geometric (PyG) extends PyTorch with the MessagePassing framework — the core abstraction for all GNN layers — and provides standard convolutions (GCNConv, GATConv, GraphSAGEConv, GINConv), graph pooling, batching of variable-size graphs, and datasets. Use when: performing node classification (e.g., predicting labels on a citation network), graph classification (e.g., predicting molecular properties), link prediction (e.g., recommending new connections), learning representations on any graph-structured data (social networks, molecules, knowledge graphs, protein structures), implementing custom GNN architectures via the MessagePassing base class, working with heterogeneous graphs (multiple node/edge types), or any task where data has explicit relational structure that CNNs/RNNs cannot capture. Complements networkx (classical graph algorithms) and rdkit (molecular graphs) — PyG adds the deep learning layer on top.
version: 2.5.0
license: MIT
---

# PyTorch Geometric — Graph Neural Networks

PyTorch Geometric (PyG) is the standard library for deep learning on graphs. Where networkx handles graph algorithms (shortest path, centrality, community detection), PyG handles **learning on graphs**: training neural networks that operate directly on graph structure. The core insight: a GNN layer aggregates information from a node's neighbors, learns which neighbors matter, and produces new node representations — all differentiable, all trainable.

## Core Mental Model

```
A GRAPH has:
  • Nodes (vertices)    — each has a feature vector
  • Edges (connections) — each optionally has attributes
  • Structure           — which nodes connect to which

A GNN LAYER does (per node):
  1. GATHER   messages from neighbors
  2. AGGREGATE messages (sum / mean / max)
  3. UPDATE   own representation using aggregated + self

  Node v:  h_v ← UPDATE( h_v, AGGREGATE( MESSAGE(h_u, e_uv) for u ∈ N(v) ) )

  After k layers: each node "sees" its k-hop neighborhood.
  This is how local structure becomes global representation.

PyG's DATA OBJECT:
  x          → node feature matrix      [num_nodes, num_features]
  edge_index → edge list (COO format)   [2, num_edges]
  edge_attr  → edge feature matrix      [num_edges, num_edge_features]  (optional)
  y          → labels                   [num_nodes] or [num_graphs]     (optional)
```

### edge_index — The Key Format

```
Graph:  0 → 1,  0 → 2,  1 → 2

edge_index = tensor([[0, 0, 1],     ← source nodes
                     [1, 2, 2]])    ← target nodes

  Column i describes edge i: source = edge_index[0, i], target = edge_index[1, i]

  ⚠️  UNDIRECTED graph: store BOTH directions!
      0 — 1  becomes  0→1 AND 1→0  →  edge_index has 2× the edges

  Messages flow: source → target  (default in MessagePassing)
```

## Reference Documentation

**PyG docs**: https://pytorch-geometric.readthedocs.io/en/latest/  
**PyG tutorials**: https://pytorch-geometric.readthedocs.io/en/latest/tutorials.html  
**GitHub**: https://github.com/pyg-team/pytorch_geometric  
**Search patterns**: `Data`, `MessagePassing`, `GCNConv`, `global_mean_pool`, `Batch`

## Quick Reference

### Installation

```bash
# Install PyTorch first (match your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then PyG — use the install finder at https://pytorch-geometric.readthedocs.io/en/latest/install.html
pip install torch-geometric
# Core extensions (required for many features):
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv
```

### Standard Imports

```python
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import GCNConv, GATConv, GraphSAGEConv, GINConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.datasets import Planetoid, TUDataset
```

### Basic Pattern — Build a Graph, Define a GNN, Train

```python
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# ─── 1. Build a graph ───
x = torch.tensor([[1.0, 0.0],    # Node 0 features
                  [0.0, 1.0],    # Node 1 features
                  [1.0, 1.0],    # Node 2 features
                  [0.0, 0.0]],   # Node 3 features
                 dtype=torch.float)

# Edges: 0-1, 1-2, 2-3 (undirected → both directions)
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
                           [1, 0, 2, 1, 3, 2]], dtype=torch.long)

y = torch.tensor([0, 0, 1, 1])   # Node labels (2 classes)

data = Data(x=x, edge_index=edge_index, y=y)
print(data)  # Data(x=[4, 2], edge_index=[2, 6], y=[4])

# ─── 2. Define GNN model ───
class SimpleGCN(torch.nn.Module):
    def __init__(self, in_features, hidden, out_classes):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden)
        self.conv2 = GCNConv(hidden, out_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x   # [num_nodes, out_classes] — logits per node

model = SimpleGCN(in_features=2, hidden=16, out_classes=2)
out   = model(data.x, data.edge_index)   # Shape: [4, 2]
```

## Critical Rules

### ✅ DO

- **Make undirected graphs bidirectional in edge_index** — If edge 0→1 exists, include 1→0 too. Use `torch_geometric.utils.to_undirected()` to do this automatically.
- **Keep edge_index as `torch.long` (int64)** — Always. Node feature tensors are float, edge_index must be long.
- **Use `data.to(device)` to move entire graph** — Moves x, edge_index, edge_attr, y all at once. Don't move tensors individually.
- **Use train_mask/val_mask/test_mask for node classification** — Standard transductive split. Masks are boolean tensors of shape [num_nodes].
- **Use DataLoader for graph classification** — It batches multiple graphs into one Batch object. Don't manually concatenate.
- **Use `global_mean_pool` or `global_add_pool` before the final classifier in graph-level tasks** — Converts variable-size node matrices to fixed-size graph vectors.
- **Add self-loops before GCN layers** — `GCNConv` adds them by default (`add_self_loops=True`). If you disabled them, node features don't propagate to themselves.
- **Use `batch` argument in pooling** — `global_mean_pool(x, batch)` — `batch` tells the pooling which nodes belong to which graph in a Batch.

### ❌ DON'T

- **Don't confuse edge_index shape** — It's `[2, E]`, NOT `[E, 2]`. Row 0 = sources, row 1 = targets. This is the #1 bug in PyG code.
- **Don't use GCN on heterogeneous graphs** — GCNConv assumes homogeneous graphs (one node type, one edge type). Use `HeteroConv` or type-specific layers.
- **Don't forget `model.eval()` and `torch.no_grad()` during inference** — Dropout and batch norm behave differently.
- **Don't assume edge_index is sorted** — PyG doesn't guarantee edge ordering. Don't index into edge_attr assuming a specific edge order.
- **Don't use standard PyTorch DataLoader** — Use `torch_geometric.data.DataLoader` which knows how to batch graphs.
- **Don't stack node features across graphs manually** — `Batch.from_data_list()` handles this with correct edge_index offsetting.

## Anti-Patterns (NEVER)

```python
import torch
from torch_geometric.data import Data

# ❌ BAD: edge_index transposed — [E, 2] instead of [2, E]
edges = [(0,1), (1,2), (2,3)]
edge_index = torch.tensor(edges, dtype=torch.long)   # Shape: [3, 2] ← WRONG
# GCNConv will silently produce garbage or crash.

# ✅ GOOD: Transpose to [2, E]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # Shape: [2, 3] ✓

# ─────────────────────────────────────────────────────────────

# ❌ BAD: Directed edges for an undirected graph — messages flow one way only
edge_index = torch.tensor([[0, 1, 2],
                           [1, 2, 3]], dtype=torch.long)
# Node 3 receives from 2, but node 0 never receives from 1.
# GCN on this graph: nodes at the "end" of chains have rich representations,
# nodes at the "start" stay at initialization.

# ✅ GOOD: Add reverse edges
from torch_geometric.utils import to_undirected
edge_index = to_undirected(edge_index)
# Now: [[0,1,1,2,2,3], [1,0,2,1,3,2]]

# ─────────────────────────────────────────────────────────────

# ❌ BAD: Moving tensors to different devices separately
x          = x.to('cuda')
edge_index = edge_index.to('cuda')
y          = y.to('cuda')   # Easy to forget one → runtime error

# ✅ GOOD: Move the whole Data object
data = data.to('cuda')   # All attributes moved atomically

# ─────────────────────────────────────────────────────────────

# ❌ BAD: Standard DataLoader for graph datasets
from torch.utils.data import DataLoader as TorchDataLoader
loader = TorchDataLoader(dataset, batch_size=32)   # Can't batch graphs!

# ✅ GOOD: PyG DataLoader
from torch_geometric.data import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)
# Returns Batch objects — concatenated graphs with correct edge_index offsets
```

## The Data Object

```python
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

# ─── Construct from scratch ───
data = Data(
    x=torch.randn(5, 16),                          # 5 nodes, 16 features each
    edge_index=torch.tensor([[0,1,2,3], [1,2,3,4]], dtype=torch.long),
    edge_attr=torch.randn(4, 8),                   # 4 edges, 8 features each
    y=torch.tensor([0, 1, 0, 1, 0]),               # Node labels
    pos=torch.randn(5, 2),                         # Node positions (optional)
)

# ─── Inspect ───
print(data)                        # Data(x=[5,16], edge_index=[2,4], ...)
print(data.num_nodes)              # 5
print(data.num_edges)              # 4
print(data.num_node_features)      # 16
print(data.is_undirected())        # True/False

# ─── Make undirected ───
data.edge_index = to_undirected(data.edge_index)
# edge_attr must also be duplicated if present:
# data.edge_attr = torch.cat([data.edge_attr, data.edge_attr], dim=0)

# ─── Build from edge list (e.g., from NetworkX or CSV) ───
import pandas as pd

# Edge list: src, dst, weight
edges_df = pd.DataFrame({
    'src': [0, 1, 2, 3],
    'dst': [1, 2, 3, 0],
    'weight': [0.5, 1.0, 0.3, 0.8]
})

edge_index = torch.tensor([edges_df['src'].values,
                           edges_df['dst'].values], dtype=torch.long)
edge_index = to_undirected(edge_index)

edge_attr  = torch.tensor(edges_df['weight'].values, dtype=torch.float).unsqueeze(1)
edge_attr  = torch.cat([edge_attr, edge_attr], dim=0)   # Mirror for undirected

num_nodes  = max(edges_df['src'].max(), edges_df['dst'].max()) + 1
x          = torch.eye(num_nodes)   # One-hot identity features if no attributes

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ─── Add custom attributes ───
data.graph_label    = torch.tensor([1])            # Graph-level label
data.node_id        = torch.arange(data.num_nodes) # Arbitrary metadata
data.split          = 'train'                      # String attributes fine too
```

## MessagePassing Framework

MessagePassing is the base class for ALL GNN layers in PyG. Understanding it = understanding how GNNs work.

```python
import torch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F

class CustomConv(MessagePassing):
    """
    Custom GNN layer via MessagePassing.

    The propagate() call triggers this sequence:
      1. message()   — compute message for each edge (source → target)
      2. aggregate() — combine messages arriving at each target node
      3. update()    — update each node's representation

    propagate(edge_index, x=x) routes:
      • x_j → source node features (j = source index)
      • x_i → target node features (i = target index)
      Subscript _i = target, _j = source. Always.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')   # Aggregation: 'add', 'mean', 'max'
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Transform features BEFORE propagation (more efficient)
        x = self.lin(x)
        # Propagate: runs message() → aggregate() → update()
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        """
        x_j: source node features for each edge. Shape: [num_edges, out_channels]
        Return: message to send along each edge.
        """
        return x_j   # Simplest: just pass source features through

    def update(self, aggr_out):
        """
        aggr_out: aggregated messages per target node. Shape: [num_nodes, out_channels]
        Return: updated node representation.
        """
        return aggr_out   # Simplest: use aggregation directly


# ─── Attention-weighted custom layer ───
class AttentionConv(MessagePassing):
    """Messages weighted by learned attention scores (simplified GAT)."""
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin   = torch.nn.Linear(in_channels, out_channels)
        self.att   = torch.nn.Parameter(torch.Tensor(1, out_channels))
        torch.nn.init.xavier_uniform_(self.att.unsqueeze(0))

    def forward(self, x, edge_index):
        x = self.lin(x)
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i = target features, x_j = source features
        # Attention score: how much should target i attend to source j?
        alpha = (x_i * self.att).sum(dim=-1) + (x_j * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        # Note: full GAT uses softmax over neighbors — see GATConv for production version
        return x_j * alpha.unsqueeze(-1)


# ─── Usage ───
# conv = CustomConv(16, 32)
# out  = conv(data.x, data.edge_index)   # [num_nodes, 32]
```

## Standard Layers — When to Use Which

```python
from torch_geometric.nn import GCNConv, GATConv, GraphSAGEConv, GINConv
import torch.nn as nn

# ─── GCNConv: Graph Convolutional Network (Kipf & Welling, 2017) ───
# Averages neighbor features (with degree normalization).
# Fast, simple baseline. No attention, no edge features.
# USE: citation networks, social networks, when speed matters.
conv_gcn = GCNConv(in_channels=16, out_channels=32)
# out = conv_gcn(x, edge_index)

# ─── GATConv: Graph Attention Network (Veličković et al., 2018) ───
# Learns attention weights — some neighbors matter more than others.
# Slower than GCN but usually better accuracy.
# USE: when neighbor importance varies, heterogeneous neighborhood structure.
conv_gat = GATConv(in_channels=16, out_channels=32, heads=8, concat=True)
# out shape: [num_nodes, heads * out_channels] if concat=True
# out = conv_gat(x, edge_index)

# ─── GraphSAGEConv: Inductive Representation Learning (Hamilton et al., 2017) ───
# Samples and aggregates neighbor features. Designed for INDUCTIVE setting
# (generalizes to unseen nodes at test time, e.g., new users joining a network).
# USE: large dynamic graphs, inductive node classification.
conv_sage = GraphSAGEConv(in_channels=16, out_channels=32, aggr='mean')
# out = conv_sage(x, edge_index)

# ─── GINConv: Graph Isomorphism Network (Xu et al., 2019) ───
# Theoretically most powerful among message-passing GNNs.
# Uses a learnable MLP to combine self + aggregated neighbors.
# USE: graph classification, when distinguishing graph structures matters.
mlp = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 32))
conv_gin = GINConv(nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 32)))
# out = conv_gin(x, edge_index)


# ─── LAYER SELECTION GUIDE ───
# Task                          | Recommended
# ──────────────────────────────┼─────────────────────
# Citation network (transductive)| GCN or GAT
# Social network (inductive)    | GraphSAGE
# Molecular property prediction | GIN or GAT
# Knowledge graph               | GAT (edge types matter)
# Quick baseline                | GCN (fastest)
# Best accuracy (small graph)   | GAT (multi-head)
# Best graph classification     | GIN (most expressive)
```

## Node Classification Pipeline

```python
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv

# ─── Load dataset: Cora citation network ───
# 2708 papers (nodes), 5429 citations (edges), 7 classes
# Each node has a 1433-dim bag-of-words feature vector
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data    = dataset[0]   # Single graph

print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}, "
      f"Features: {data.num_node_features}, Classes: {dataset.num_classes}")
print(f"Train: {data.train_mask.sum()}, Val: {data.val_mask.sum()}, "
      f"Test: {data.test_mask.sum()}")

# ─── Model ───
class GATNodeClassifier(torch.nn.Module):
    def __init__(self, in_features, hidden, out_classes, heads=8, dropout=0.6):
        super().__init__()
        self.dropout = dropout
        self.conv1 = GATConv(in_features, hidden, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden * heads, out_classes, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x   # [num_nodes, num_classes]

# ─── Training loop ───
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = GATNodeClassifier(dataset.num_node_features, hidden=8, out_classes=dataset.num_classes).to(device)
data   = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

best_val_acc = 0
patience     = 50
patience_ctr = 0

for epoch in range(200):
    # Train
    model.train()
    optimizer.zero_grad()
    out  = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    # Validate
    model.eval()
    with torch.no_grad():
        out      = model(data.x, data.edge_index)
        pred     = out.argmax(dim=1)
        val_acc  = (pred[data.val_mask] == data.y[data.val_mask]).float().mean().item()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state   = {k: v.clone() for k, v in model.state_dict().items()}
        patience_ctr = 0
    else:
        patience_ctr += 1
        if patience_ctr >= patience:
            break

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")

# Test
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    out      = model(data.x, data.edge_index)
    pred     = out.argmax(dim=1)
    test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
print(f"\nTest Accuracy: {test_acc:.4f}")   # Cora baseline: ~0.83
```

## Graph Classification Pipeline

```python
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
import torch.nn as nn

# ─── Load dataset: MUTAG — molecular mutagenicity ───
# 188 molecules (graphs), 2 classes (mutagenic or not)
# Each node = atom (7 features), each edge = bond type
dataset = TUDataset(root='/tmp/MUTAG', name='MUTAG')

# Split
torch.manual_seed(42)
perm    = torch.randperm(len(dataset))
dataset = dataset[perm]
train_dataset = dataset[:150]
test_dataset  = dataset[150:]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# ─── Model: GIN + global pooling ───
class GINGraphClassifier(torch.nn.Module):
    def __init__(self, in_features, hidden, out_classes, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # First layer
        self.convs.append(GINConv(nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )))
        self.bns.append(nn.BatchNorm1d(hidden))

        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
            )))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.classifier = nn.Linear(hidden, out_classes)

    def forward(self, x, edge_index, batch):
        # batch: [num_nodes] — maps each node to its graph index in the Batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        # Global pooling: aggregate all node features → one vector per graph
        x = global_add_pool(x, batch)   # [batch_size, hidden]
        x = self.classifier(x)          # [batch_size, out_classes]
        return x

# ─── Training ───
device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model     = GINGraphClassifier(dataset.num_node_features, hidden=64,
                               out_classes=dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(100):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch      = batch.to(device)
        optimizer.zero_grad()
        out  = model(batch.x, batch.edge_index, batch.batch)
        loss = F.cross_entropy(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Evaluate
    model.eval()
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out   = model(batch.x, batch.edge_index, batch.batch)
            pred  = out.argmax(dim=1)
            correct += (pred == batch.y).sum().item()

    test_acc = correct / len(test_dataset)
    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d} | Loss: {total_loss/len(train_loader):.4f} | Test Acc: {test_acc:.4f}")
```

## Link Prediction

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling

class LinkPredictor(torch.nn.Module):
    """
    Encoder-decoder architecture for link prediction.
    Encoder: GNN produces node embeddings.
    Decoder: dot product of embeddings predicts edge probability.
    """
    def __init__(self, in_features, hidden):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden)
        self.conv2 = GCNConv(hidden, hidden)

    def encode(self, x, edge_index):
        """Produce node embeddings using the message-passing GNN."""
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x   # [num_nodes, hidden]

    def decode(self, z, edge_index):
        """
        Predict scores for given edges via dot product.
        z: node embeddings [num_nodes, hidden]
        edge_index: edges to score [2, num_edges]
        Returns: scores [num_edges] — higher = more likely edge
        """
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def decode_all(self, z):
        """Score ALL possible edges — use only for small graphs."""
        src, dst = z.size(0) * torch.ones(z.size(0), dtype=torch.long).cumsum(0) - z.size(0), \
                   torch.arange(z.size(0)).repeat(z.size(0))
        # Simpler: use torch combinations
        idx = torch.combinations(torch.arange(z.size(0)), r=2).t()
        return self.decode(z, idx), idx


# ─── Training loop ───
def train_link_predictor(data, model, optimizer, device, num_epochs=100):
    """
    Split edges: use SOME edges for message passing (supervision signal),
    and SEPARATE edges as positive supervision + negative samples as negatives.

    This is critical: if you include supervision edges in the GNN input,
    the model trivially learns them → data leakage.
    """
    num_nodes = data.num_nodes
    model     = model.to(device)
    data      = data.to(device)

    # Simple split: use all edges for GNN input, sample positives/negatives for loss
    # (In production, split edges into message-passing / supervision / test sets)
    edge_index = data.edge_index

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Encode: GNN on full graph
        z = model.encode(data.x, edge_index)

        # Positive edges: sample from existing edges
        pos_edge_index = edge_index[:, :edge_index.size(1) // 2]   # Avoid duplicates in undirected

        # Negative edges: sample non-existing edges
        neg_edge_index = negative_sampling(
            edge_index,
            num_nodes=num_nodes,
            num_neg_samples=pos_edge_index.size(1)
        )

        # Scores
        pos_scores = model.decode(z, pos_edge_index)
        neg_scores = model.decode(z, neg_edge_index)

        # Binary cross-entropy loss
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, torch.zeros_like(neg_scores))
        loss     = pos_loss + neg_loss

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    return model
```

## Batching — How PyG Handles Multiple Graphs

```python
import torch
from torch_geometric.data import Data, Batch

# Create 3 small graphs
g1 = Data(x=torch.randn(3, 4), edge_index=torch.tensor([[0,1],[1,2]]).t().contiguous(), y=torch.tensor([0]))
g2 = Data(x=torch.randn(5, 4), edge_index=torch.tensor([[0,1],[1,2],[2,3],[3,4]]).t().contiguous(), y=torch.tensor([1]))
g3 = Data(x=torch.randn(2, 4), edge_index=torch.tensor([[0,1]]).t().contiguous(), y=torch.tensor([0]))

# Batch them
batch = Batch.from_data_list([g1, g2, g3])

print(f"batch.x shape:          {batch.x.shape}")           # [10, 4] — all nodes concatenated
print(f"batch.edge_index shape: {batch.edge_index.shape}")  # [2, 7] — all edges concatenated
print(f"batch.y shape:          {batch.y.shape}")           # [3] — one label per graph
print(f"batch.batch shape:      {batch.batch.shape}")       # [10] — maps each node to graph index

print(f"batch.batch: {batch.batch}")
# tensor([0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
#         └─ g1 ─┘  └──── g2 ─────┘  └g3┘

# KEY: edge_index is OFFSET by cumulative node count:
# g1 edges: [0,1], [1,2]       → stay [0,1], [1,2]
# g2 edges: [0,1], [1,2], ...  → become [3,4], [4,5], ... (offset by 3)
# g3 edges: [0,1]              → becomes [8,9] (offset by 8)

# Unbatch back to individual graphs
graphs = batch.to_data_list()
print(f"Recovered {len(graphs)} graphs")

# ─── Using batch in pooling ───
from torch_geometric.nn import global_mean_pool

node_embeddings = torch.randn(10, 32)   # Pretend GNN output
graph_embeddings = global_mean_pool(node_embeddings, batch.batch)
print(f"Graph embeddings: {graph_embeddings.shape}")   # [3, 32] — one per graph
```

## Heterogeneous Graphs

```python
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv

# ─── Build heterogeneous graph ───
# Knowledge graph: Users watch Movies, Movies belong to Genres
data = HeteroData()

# Node features
data['user'].x   = torch.randn(100, 16)    # 100 users, 16 features
data['movie'].x  = torch.randn(50, 32)     # 50 movies, 32 features
data['genre'].x  = torch.randn(10, 8)      # 10 genres, 8 features

# Edge types: (source_type, relation, target_type)
data['user', 'watches', 'movie'].edge_index   = torch.randint(0, 50, (2, 200))
data['movie', 'belongs_to', 'genre'].edge_index = torch.randint(0, 10, (2, 80))
# Reverse edges for message passing in both directions:
data['movie', 'watched_by', 'user'].edge_index = data['user', 'watches', 'movie'].edge_index.flip(0)

print(data)
# HeteroData with 3 node types and 3 edge types

# ─── Heterogeneous GNN ───
class HeteroGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define a conv for each edge type
        self.conv1 = HeteroConv({
            ('user', 'watches', 'movie'):       SAGEConv(16, 64),
            ('movie', 'watched_by', 'user'):    SAGEConv(32, 64),
            ('movie', 'belongs_to', 'genre'):   SAGEConv(32, 64),
        }, aggr='sum')

        self.conv2 = HeteroConv({
            ('user', 'watches', 'movie'):       SAGEConv(64, 32),
            ('movie', 'watched_by', 'user'):    SAGEConv(64, 32),
            ('movie', 'belongs_to', 'genre'):   SAGEConv(64, 32),
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: torch.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return x_dict   # Updated embeddings per node type

# model = HeteroGNN()
# out   = model(data.x_dict, data.edge_index_dict)
# print(out['movie'].shape)   # [50, 32] — movie embeddings
```

## Practical Workflows

### 1. Molecular Property Prediction

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, BatchNorm
import torch.nn as nn

class MolecularGNN(torch.nn.Module):
    """
    Full molecular property prediction model.
    Input: molecular graph (atoms = nodes, bonds = edges)
    Output: predicted property (e.g., toxicity, solubility)
    """
    def __init__(self, node_features, edge_features, hidden=128, num_layers=5, dropout=0.3):
        super().__init__()
        self.edge_encoder = nn.Linear(edge_features, hidden)
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()

        # Input layer
        self.convs.append(GINConv(nn.Sequential(
            nn.Linear(node_features, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
        )))
        self.bns.append(BatchNorm(hidden))

        for _ in range(num_layers - 1):
            self.convs.append(GINConv(nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
            )))
            self.bns.append(BatchNorm(hidden))

        # Readout: sum + mean pooling concatenated (captures both count and density)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)
        )
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        # Layer-wise readout: concatenate pooled representations from each layer
        h_list = []

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            h_list.append(x)

        # Use last layer's output for readout
        x_sum  = global_add_pool(x, batch)    # [batch, hidden]
        x_mean = global_add_pool(x, batch) / torch.bincount(batch).unsqueeze(1).float()
        x_pool = torch.cat([x_sum, x_mean], dim=1)   # [batch, hidden*2]

        return self.head(x_pool).squeeze(-1)   # [batch] — predicted property

# ─── Usage with RDKit-generated graphs ───
# from rdkit import Chem
# mol = Chem.MolFromSmiles('CCO')
# # Convert mol → Data object (see rdkit skill for molecular featurization)
# # Then: model(data.x, data.edge_index, data.edge_attr, torch.zeros(data.num_nodes, dtype=torch.long))
```

### 2. Knowledge Graph Embedding + Prediction

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

class TransE(nn.Module):
    """
    TransE knowledge graph embedding.
    Scores a triple (h, r, t) as: ||h + r - t||
    Low score = plausible triple.
    """
    def __init__(self, num_entities, num_relations, embedding_dim=100):
        super().__init__()
        self.entity_emb   = nn.Embedding(num_entities, embedding_dim)
        self.relation_emb = nn.Embedding(num_relations, embedding_dim)

        # Initialize
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)

    def forward(self, head, relation, tail):
        """
        head, relation, tail: LongTensors of entity/relation indices
        Returns: distance scores (lower = more plausible)
        """
        h = self.entity_emb(head)      # [batch, dim]
        r = self.relation_emb(relation)
        t = self.entity_emb(tail)
        return torch.norm(h + r - t, dim=-1)   # [batch]

    def predict_tail(self, head, relation, top_k=5):
        """Given (head, relation, ?), predict most likely tail entities."""
        h = self.entity_emb(head).unsqueeze(1)     # [1, 1, dim]
        r = self.relation_emb(relation).unsqueeze(1)
        all_t = self.entity_emb.weight.unsqueeze(0)  # [1, num_entities, dim]
        scores = torch.norm(h + r - all_t, dim=-1)   # [1, num_entities]
        _, top_idx = scores.topk(top_k, dim=1, largest=False)
        return top_idx.squeeze(0)

# ─── Training ───
# model = TransE(num_entities=50000, num_relations=500, embedding_dim=200)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# # Margin loss: score(pos) + margin < score(neg)
# pos_score = model(pos_head, pos_rel, pos_tail)
# neg_score = model(neg_head, neg_rel, neg_tail)   # Corrupted triples
# loss = F.relu(pos_score - neg_score + margin).mean()
```

### 3. Graph Generation Evaluation

```python
import torch
import networkx as nx
import numpy as np
from torch_geometric.utils import to_networkx

def evaluate_generated_graphs(generated_data_list: list) -> dict:
    """
    Standard metrics for evaluating generated molecular/graph structures.
    """
    stats = {
        'num_graphs':       len(generated_data_list),
        'avg_nodes':        [],
        'avg_edges':        [],
        'avg_degree':       [],
        'connected_frac':   [],
        'avg_clustering':   [],
    }

    for data in generated_data_list:
        G = to_networkx(data, to_directed=False)
        stats['avg_nodes'].append(G.number_of_nodes())
        stats['avg_edges'].append(G.number_of_edges())

        degrees = [d for _, d in G.degree()]
        stats['avg_degree'].append(np.mean(degrees) if degrees else 0)
        stats['connected_frac'].append(1.0 if nx.is_connected(G) else 0.0)
        stats['avg_clustering'].append(nx.average_clustering(G))

    # Summarize
    summary = {
        'num_graphs':           stats['num_graphs'],
        'avg_nodes':            np.mean(stats['avg_nodes']),
        'avg_edges':            np.mean(stats['avg_edges']),
        'avg_degree':           np.mean(stats['avg_degree']),
        'connectivity_rate':    np.mean(stats['connected_frac']),
        'avg_clustering_coeff': np.mean(stats['avg_clustering']),
    }
    return summary

# summary = evaluate_generated_graphs(generated_molecules)
# print(summary)
```

## Visualization

```python
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data

def visualize_graph(data: Data, node_labels: list = None, title: str = 'Graph'):
    """Visualize a PyG Data object using NetworkX layout."""
    G = to_networkx(data, to_directed=False)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Layout options: spring (force-directed), kamada_kawai, spectral
    pos = nx.spring_layout(G, seed=42)

    # Node colors: by label if available
    if data.y is not None:
        colors = data.y.numpy()
        nx.draw_networkx_nodes(G, pos, node_color=colors, cmap=plt.cm.Set3,
                               node_size=300, ax=ax)
    else:
        nx.draw_networkx_nodes(G, pos, node_size=300, ax=ax)

    nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)

    if node_labels:
        nx.draw_networkx_labels(G, pos, labels={i: l for i, l in enumerate(node_labels)},
                                font_size=8, ax=ax)

    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# visualize_graph(data, title='Cora Citation Network (sample)')
```

## Common Pitfalls and Solutions

### edge_index Shape Is Wrong

```python
import torch

# ❌ Shape [E, 2] — most common mistake
edges = [(0,1), (1,2), (2,0)]
edge_index = torch.tensor(edges, dtype=torch.long)
print(edge_index.shape)   # [3, 2] ← WRONG

# ✅ Transpose to [2, E]
edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
print(edge_index.shape)   # [2, 3] ✓

# ✅ Or build directly in correct shape
edge_index = torch.tensor([[0, 1, 2],    # sources
                           [1, 2, 0]],   # targets
                          dtype=torch.long)
```

### Forgetting to Make Graph Undirected

```python
import torch
from torch_geometric.utils import to_undirected

# ❌ Only one direction — half the nodes don't receive messages
edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
# Node 0 never receives a message from anyone!

# ✅ Make undirected
edge_index = to_undirected(edge_index)
# [[0,1,1,2,2,3], [1,0,2,1,3,2]] — both directions present
```

### Device Mismatch in Batched Training

```python
import torch
from torch_geometric.data import DataLoader

# ❌ Model on GPU, batch not moved → RuntimeError
model = model.cuda()
for batch in loader:
    out = model(batch.x, batch.edge_index)   # batch.x is on CPU!

# ✅ Always move batch to device
device = torch.device('cuda')
for batch in loader:
    batch = batch.to(device)    # Moves x, edge_index, edge_attr, y, batch — ALL of them
    out   = model(batch.x, batch.edge_index)
```

### GAT Output Shape with Multi-Head

```python
from torch_geometric.nn import GATConv

# GATConv with heads=8, out_channels=32, concat=True (default):
conv = GATConv(16, 32, heads=8, concat=True)
# Output shape: [num_nodes, 8 * 32] = [num_nodes, 256] ← NOT [num_nodes, 32]!

# Next layer must accept 256, not 32:
conv2 = GATConv(8 * 32, 64, heads=1, concat=False)   # Final layer: heads=1, concat=False
# Output: [num_nodes, 64]

# ─── Or use concat=False throughout ───
conv = GATConv(16, 32, heads=8, concat=False)   # Averages across heads
# Output: [num_nodes, 32]
```

### Global Pooling Without batch Tensor

```python
from torch_geometric.nn import global_mean_pool
import torch

# ❌ Pooling without batch → treats entire batch as one graph
x     = torch.randn(100, 32)   # 100 nodes total (from 4 graphs)
pooled = global_mean_pool(x, batch=None)   # Returns single [1, 32] vector — WRONG

# ✅ Pass batch tensor from DataLoader
# batch = tensor([0,0,0,...,1,1,1,...,2,2,...,3,3,...])
pooled = global_mean_pool(x, batch)   # Returns [4, 32] — one vector per graph ✓
```

---

PyG's power is the **MessagePassing** abstraction: every GNN layer — no matter how complex — reduces to "gather neighbor messages, aggregate, update." Master that loop and you can implement any architecture in the literature. The standard pipeline is always: build Data → define model (stack of conv layers + pooling) → train with appropriate loss (cross-entropy for classification, BCE for link prediction) → evaluate. The only fundamental difference from standard PyTorch is how graphs batch — `Batch` handles the edge_index offsetting automatically, but you must pass `batch.batch` to pooling layers.
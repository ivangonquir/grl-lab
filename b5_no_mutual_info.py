import os
import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from b2_exploiting_the_graph_structure import train_epoch, evaluate, GNN

OUTPUTS_DIR = 'grl-lab-outputs'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = Planetoid(root='data/PubMed', name='PubMed')
data = dataset[0].to(device)

mappings = torch.load(os.path.join(OUTPUTS_DIR, 'a1_mappings.pt'), weights_only=False)
entity_to_id = mappings['entity_to_id']


# Stash original features
x_original = data.x.clone()

# Random features, same shape
torch.manual_seed(42)
data.x = torch.randn(data.num_nodes, 500, device=device)

model = GNN(in_channels=500, hidden_channels=256, out_channels=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

best_val, best_test = 0, 0
for epoch in range(200):
    loss = train_epoch(model, data, optimizer, criterion)
    accs = evaluate(model, data)
    if accs['val'] > best_val:
        best_val, best_test = accs['val'], accs['test']
    if epoch % 20 == 0:
        print(f"epoch {epoch:3d}  loss={loss:.4f}  "
              f"train={accs['train']:.3f}  val={accs['val']:.3f}  test={accs['test']:.3f}")

print(f"\nBest val: {best_val:.4f}   Test at best val: {best_test:.4f}")

# Restore for the next experiment
data.x = x_original


# --- B.5.2: KGE initialisation ---

# Load TransE embeddings from disk (file produced by section 9 of A.3)
transe_emb = torch.load(os.path.join(OUTPUTS_DIR, 'best_transe_entity_emb.pt'), weights_only=False)
emb_dim = transe_emb.shape[1]

# Align KGE rows to PyG node order
x_kge = torch.zeros(data.num_nodes, emb_dim)
for label, kge_id in entity_to_id.items():
    pyg_idx = int(label.replace('paper_', '')) if label.startswith('paper_') else int(label)
    if 0 <= pyg_idx < data.num_nodes:
        x_kge[pyg_idx] = transe_emb[kge_id]

# Swap in the KGE features
x_original = data.x.clone()
data.x = x_kge.to(device)

# Fresh model with the new input dimension
model = GNN(in_channels=emb_dim, hidden_channels=256, out_channels=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()

# Train
best_val, best_test = 0, 0
for epoch in range(200):
    loss = train_epoch(model, data, optimizer, criterion)
    accs = evaluate(model, data)
    if accs['val'] > best_val:
        best_val, best_test = accs['val'], accs['test']
    if epoch % 20 == 0:
        print(f"epoch {epoch:3d}  loss={loss:.4f}  "
              f"train={accs['train']:.3f}  val={accs['val']:.3f}  test={accs['test']:.3f}")

print(f"\nBest val: {best_val:.4f}   Test at best val: {best_test:.4f}")

# Restore TF-IDF features
data.x = x_original

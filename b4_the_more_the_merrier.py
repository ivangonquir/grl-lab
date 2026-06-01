import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from b2_exploiting_the_graph_structure import train_epoch, evaluate


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = Planetoid(root='data/PubMed', name='PubMed')
data = dataset[0].to(device)


class DeepGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


criterion = nn.CrossEntropyLoss()

depth_results = {}
for num_layers in [2, 4, 8, 16]:
    torch.manual_seed(2026)
    torch.cuda.manual_seed(2026)
    model = DeepGNN(500, 256, 3, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, best_test = 0, 0
    for epoch in range(200):
        loss = train_epoch(model, data, optimizer, criterion)
        accs = evaluate(model, data)
        if accs['val'] > best_val:
            best_val, best_test = accs['val'], accs['test']
    depth_results[num_layers] = best_test
    print(f"{num_layers} layers: test = {best_test:.4f}")
    
    
    
import os
import matplotlib.pyplot as plt

os.makedirs('images', exist_ok=True)

depths = list(depth_results.keys())
accs   = list(depth_results.values())

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(depths, accs, marker='o', linewidth=2, markersize=8, color='crimson')
ax.axhline(0.40, color='gray', linestyle='--', alpha=0.6,
           label='Majority-class baseline ($\\approx 0.40$)')
ax.set_xticks(depths)
ax.set_xlabel('Number of GCN layers')
ax.set_ylabel('Test accuracy')
ax.set_title('Effect of GNN depth on classification accuracy')
ax.set_ylim(0.3, 0.85)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('images/b4_depth_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()

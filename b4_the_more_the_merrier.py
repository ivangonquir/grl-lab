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

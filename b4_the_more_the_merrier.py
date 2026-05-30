import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

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
    
    
    
    
    
depth_results = {}
for num_layers in [2, 4, 8, 16]:
    model = DeepGNN(500, 256, 3, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_val, best_test = 0, 0
    for epoch in range(200):
        loss = train_epoch(model, data)
        accs = evaluate(model, data)
        if accs['val'] > best_val:
            best_val, best_test = accs['val'], accs['test']
    depth_results[num_layers] = best_test
    print(f"{num_layers} layers: test = {best_test:.4f}")
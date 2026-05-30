import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

OUTPUTS_DIR = 'grl-lab-outputs'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = Planetoid(root='data/PubMed', name='PubMed')
data = dataset[0]


class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, return_embs=False):
        x = x.to(device)
        edge_index = edge_index.to(device)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        if return_embs:
            return x                       # 256-dim node embeddings

        return self.classifier(x)          # 3-dim class logits

gnn_model = GNN(dataset.num_node_features, 256, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01, weight_decay=5e-4)


def train_gnn(model):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask].to(device), data.y[data.train_mask].to(device))
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(500):
    loss = train_gnn(gnn_model)


gnn_model.eval()
embs = gnn_model(data.x, data.edge_index, return_embs=True)
embs = embs.detach().cpu().numpy()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

labels = data.y.cpu().numpy()

pca = PCA(n_components=2).fit(embs)
embs_pca = pca.transform(embs)

tsne = TSNE(n_components=2, perplexity=30, random_state=42, init='pca')
embs_tsne = tsne.fit_transform(embs)

class_names = ['Diabetes Exp.', 'Diabetes T1', 'Diabetes T2']

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for emb_2d, ax, name in [(embs_pca, axes[0], 'PCA'),
                         (embs_tsne, axes[1], 't-SNE')]:
    for c in range(3):
        m = labels == c
        ax.scatter(emb_2d[m, 0], emb_2d[m, 1], s=8, alpha=0.6, label=class_names[c])
    ax.set_title(f'GNN node embeddings — {name}')
    ax.legend(fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
plt.tight_layout()
plt.savefig('images/embeddings_pca.png')

# Save embeddings for b8
np.save(os.path.join(OUTPUTS_DIR, 'b7_embs.npy'), embs)

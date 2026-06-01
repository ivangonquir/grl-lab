import os
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.datasets import Planetoid

OUTPUTS_DIR = 'grl-lab-outputs'

dataset = Planetoid(root='data/PubMed', name='PubMed')
data = dataset[0]

mappings = torch.load(os.path.join(OUTPUTS_DIR, 'a1_mappings.pt'), weights_only=False)
entity_to_id = mappings['entity_to_id']

embs = np.load(os.path.join(OUTPUTS_DIR, 'b7_embs.npy'))

# Align KGE embeddings to PyG node indexing
kge_raw = torch.load(os.path.join(OUTPUTS_DIR, 'best_kge_entity_emb.pt'), weights_only=False).numpy()
kge_aligned = np.zeros((data.num_nodes, kge_raw.shape[1]))
for label, kge_id in entity_to_id.items():
    pyg_idx = int(label.replace('paper_', '')) if label.startswith('paper_') else int(label)
    if 0 <= pyg_idx < data.num_nodes:
        kge_aligned[pyg_idx] = kge_raw[kge_id]

gnn_emb = embs   # already [N, 256] numpy from B.7

def top_k_cos(emb, idx, k=10):
    sims = cosine_similarity(emb[idx:idx+1], emb)[0]
    sims[idx] = -np.inf
    return set(np.argsort(-sims)[:k].tolist())

# Sample one paper for the report
P = 10
kge_nbrs = top_k_cos(kge_aligned, P)
gnn_nbrs = top_k_cos(gnn_emb,     P)
print(f"KGE neighbours: {sorted(kge_nbrs)}")
print(f"GNN neighbours: {sorted(gnn_nbrs)}")
print(f"Overlap: {kge_nbrs & gnn_nbrs}  ({len(kge_nbrs & gnn_nbrs)}/10)")

# Or, more robustly: average overlap across a random sample of papers
import random
random.seed(42)
sample = random.sample(range(data.num_nodes), 200)
overlaps = [len(top_k_cos(kge_aligned, i) & top_k_cos(gnn_emb, i)) for i in sample]
print(f"Mean overlap across 200 papers: {np.mean(overlaps):.2f}/10")

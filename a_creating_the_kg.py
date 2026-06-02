import os
import numpy as np
import torch
from pykeen.triples import TriplesFactory
from torch_geometric.datasets import Planetoid

OUTPUTS_DIR = 'grl-lab-outputs'
os.makedirs(OUTPUTS_DIR, exist_ok=True)

dataset = Planetoid(root='data/PubMed', name='PubMed')
data = dataset[0]

edge_index = data.edge_index

heads = edge_index[0].cpu().numpy()
tails = edge_index[1].cpu().numpy()

triples = np.array([
    [f"paper_{h}", "cites", f"paper_{t}"]
    for h, t in zip(heads, tails)
])

tf = TriplesFactory.from_labeled_triples(triples)
training, testing, validation = tf.split([0.8, 0.1, 0.1], random_state=2026)

torch.save(tf,         os.path.join(OUTPUTS_DIR, 'tf.pt'))
torch.save(training,   os.path.join(OUTPUTS_DIR, 'training.pt'))
torch.save(testing,    os.path.join(OUTPUTS_DIR, 'testing.pt'))
torch.save(validation, os.path.join(OUTPUTS_DIR, 'validation.pt'))

print(f"Saved KG splits to {OUTPUTS_DIR}/")

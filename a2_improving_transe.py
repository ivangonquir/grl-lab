import os
import numpy as np
import torch
from pykeen.pipeline import pipeline

OUTPUTS_DIR = 'grl-lab-outputs'

tf         = torch.load(os.path.join(OUTPUTS_DIR, 'tf.pt'),         weights_only=False)
training   = torch.load(os.path.join(OUTPUTS_DIR, 'training.pt'),   weights_only=False)
testing    = torch.load(os.path.join(OUTPUTS_DIR, 'testing.pt'),    weights_only=False)
validation = torch.load(os.path.join(OUTPUTS_DIR, 'validation.pt'), weights_only=False)

mappings      = torch.load(os.path.join(OUTPUTS_DIR, 'a1_mappings.pt'), weights_only=False)
id_to_entity  = mappings['id_to_entity']
entity_to_id  = mappings['entity_to_id']

# 1. Find a one-to-many hub in the data
mapped_triples = training.mapped_triples.cpu().numpy()
cites_rel_id = tf.relation_to_id["cites"]
citation_edges = mapped_triples[mapped_triples[:, 1] == cites_rel_id]

# Find the paper node with the most outgoing citations
heads, counts = np.unique(citation_edges[:, 0], return_counts=True)
hub_head_id = heads[np.argmax(counts)]
hub_count = np.max(counts)

# Extract the IDs of all papers cited by this specific hub
target_paper_ids = citation_edges[citation_edges[:, 0] == hub_head_id, 2]
target_tensor_ids = torch.tensor(target_paper_ids, dtype=torch.long)

print(f"Analyzing Hub: {id_to_entity[hub_head_id]} (Cites {hub_count} papers)\n")


# 2. Run TransE pipeline
print("Training TransE baseline...")
result = pipeline(
    training=training,
    testing=testing,
    validation=validation,
    model='TransE',
    training_loop='sLCWA',
    epochs=20,
    random_seed=2026,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

model = result.model
all_embeddings = model.entity_representations[0](indices=None).detach()


# 3. Compute hetero-neighborhood vs global distances
hub_embeddings = all_embeddings[target_tensor_ids]
hub_dist_matrix = torch.cdist(hub_embeddings, hub_embeddings, p=2)
hub_triu = torch.triu_indices(len(target_paper_ids), len(target_paper_ids), offset=1)
hub_distances = hub_dist_matrix[hub_triu[0], hub_triu[1]]

num_entities = all_embeddings.shape[0]
random_indices_1 = torch.randint(0, num_entities, (5000,))
random_indices_2 = torch.randint(0, num_entities, (5000,))
global_distances = torch.norm(all_embeddings[random_indices_1] - all_embeddings[random_indices_2], dim=1)

print("\n=== MATHEMATICAL PROOF OF TRANSE COLLAPSE ===")
print(f"Average Global Distance (Any 2 random papers):   {global_distances.mean().item():.4f}")
print(f"Average Neighborhood Distance (Papers cited by hub): {hub_distances.mean().item():.4f}")
print(f"Compression Ratio: {hub_distances.mean().item() / global_distances.mean().item():.2%}")

hub_label = id_to_entity[hub_head_id]
for t_id in target_paper_ids[:5]:
    print(f"({hub_label}, cites, {id_to_entity[t_id]})")
    
    
    
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(0, max(global_distances.max().item(),
                          hub_distances.max().item()), 50)
ax.hist(global_distances.cpu().numpy(), bins=bins, alpha=0.55, color='steelblue',
        density=True, label=f'Random pairs (mean = {global_distances.mean():.3f})')
ax.hist(hub_distances.cpu().numpy(), bins=bins, alpha=0.55, color='crimson',
        density=True, label=f'Hub citations (mean = {hub_distances.mean():.3f})')
ax.axvline(global_distances.mean().item(), color='steelblue', linestyle='--')
ax.axvline(hub_distances.mean().item(), color='crimson', linestyle='--')
ax.set_xlabel('Pairwise L2 distance')
ax.set_ylabel('Density')
ax.set_title('Pairwise distances in TransE embedding space')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/a2_distance_distributions.png', dpi=150, bbox_inches='tight')
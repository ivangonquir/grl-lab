from creating_the_kg import *
import torch
from pykeen.pipeline import pipeline
import os


def train_basic_model():
    result = pipeline(
        training=training,
        validation=validation,
        testing=testing,
        model='TransE',
        epochs=15,
        random_seed=2026,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )
    return result.model

model = train_basic_model()


# === A.1 — Retrieval ===
# Extract learned embeddings from the PyKEEN model
entity_embeddings   = model.entity_representations[0](indices=None).detach().cpu()
relation_embeddings = model.relation_representations[0](indices=None).detach().cpu()

entity_to_id    = training.entity_to_id
id_to_entity    = {v: k for k, v in entity_to_id.items()}
relation_to_id  = training.relation_to_id

chosen_paper_str  = "paper_10"
chosen_paper_id   = entity_to_id[chosen_paper_str]
cites_relation_id = relation_to_id["cites"]

# All citations in the full graph (used for exclusion + reporting)
full_triples   = tf.mapped_triples
mask_full      = (full_triples[:, 0] == chosen_paper_id) & (full_triples[:, 1] == cites_relation_id)
all_cited_ids  = full_triples[mask_full, 2].tolist()

# Training citations only (used for the averaging)
train_triples   = training.mapped_triples
mask_train      = (train_triples[:, 0] == chosen_paper_id) & (train_triples[:, 1] == cites_relation_id)
train_cited_ids = train_triples[mask_train, 2].tolist()

print(f"{chosen_paper_str} — full graph ({len(all_cited_ids)} citations):")
for c_id in all_cited_ids:
    print(f"  {id_to_entity[c_id]}")
print(f"\n{chosen_paper_str} — training citations only ({len(train_cited_ids)}):")
for c_id in train_cited_ids:
    print(f"  {id_to_entity[c_id]}")

# h* = mean(t_i) - r  (over training citations)
vec_r       = relation_embeddings[cites_relation_id]
cited_embs  = entity_embeddings[train_cited_ids]
h_star      = cited_embs.mean(dim=0) - vec_r

# Nearest-neighbour search excluding P_10 and every paper it already cites
excluded   = set([chosen_paper_id, *all_cited_ids])
distances  = torch.norm(entity_embeddings - h_star, dim=1)
sorted_idx = torch.argsort(distances).tolist()

for idx in sorted_idx:
    if idx not in excluded:
        retrieved_id = idx
        break

retrieved_str = id_to_entity[retrieved_id]
print(f"\nRetrieved paper: {retrieved_str}   "
      f"(L2 distance to h*: {distances[retrieved_id].item():.4f})")

# Sanity check — h* should land near e_{P_10}
e_P = entity_embeddings[chosen_paper_id]
print(f"||h* - e_{chosen_paper_str}|| = {torch.norm(h_star - e_P).item():.4f}")



# === A.1 — 2D PCA visualisation ===
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Fit PCA on the full embedding space, then project the points of interest
all_emb = entity_embeddings.cpu().numpy()
pca     = PCA(n_components=2).fit(all_emb)

chosen_2d    = pca.transform(entity_embeddings[chosen_paper_id].cpu().numpy().reshape(1, -1))[0]
cited_2d     = pca.transform(entity_embeddings[train_cited_ids].cpu().numpy())
h_star_2d    = pca.transform(h_star.cpu().numpy().reshape(1, -1))[0]
retrieved_2d = pca.transform(entity_embeddings[retrieved_id].cpu().numpy().reshape(1, -1))[0]


IMAGES_DIR = 'images'
os.makedirs(IMAGES_DIR, exist_ok = True)

plt.figure(figsize=(8, 6))
plt.scatter(*chosen_2d,     color='royalblue', s=140, marker='o', label=f'Chosen paper ({chosen_paper_str})')
plt.scatter(cited_2d[:, 0], cited_2d[:, 1],
            color='orange',    s=90,  marker='^', label='Training citations')
plt.scatter(*h_star_2d,     color='crimson',   s=180, marker='*', label=r'$h^*$ (target)')
plt.scatter(*retrieved_2d,  color='seagreen',  s=140, marker='s', label=f'Retrieved ({retrieved_str})')

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('TransE embedding space — 2D PCA projection')
plt.legend(loc='best', fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/a1_2D_sketch.png', dpi=150, bbox_inches='tight')


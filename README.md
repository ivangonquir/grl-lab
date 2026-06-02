# grl-lab

Experiments on Knowledge Graph Embeddings (KGE) and Graph Neural Networks (GNN) using the PubMed citation graph.

## Installation

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt \
    --index-url https://download.pytorch.org/whl/cu126
```

> The `--index-url` flag is required for `torch`, `torchvision`, and `torchaudio` to install the CUDA 12.6 builds. Your NVIDIA driver must support CUDA 12.6 or later.

---

## Running the scripts

All scripts must be run from the `grl-lab/` directory using the local virtual environment:

```bash
cd grl-lab/
.venv/bin/python <script.py>
```

Outputs (embeddings, models, CSVs) are written to `grl-lab-outputs/`. Images are written to `images/`.

---

## Run order

### Step 1 — Inspect the dataset (only needed if skipping the A scripts)

```bash
.venv/bin/python load_pubmed.py
```

Downloads and caches the PubMed dataset locally and prints basic stats. Only run this if you want to execute the B scripts without running the A scripts first — otherwise `creating_the_kg.py` (Step 2) already downloads the dataset as a side effect.

---

### Step 2 — Build the knowledge graph (required by all A scripts)

```bash
.venv/bin/python creating_the_kg.py
```

Converts the PubMed citation graph to RDF triples and saves the KGE train/test/validation splits to `grl-lab-outputs/`.

---

### Step 3 — A scripts (Knowledge Graph Embeddings)

**Run `a1` and `a3` before `a2`, `b5`, and `b8`.**

| Script | Depends on | Description |
|--------|-----------|-------------|
| `a1_basic_model.py` | Step 2 | Trains a basic TransE model, retrieves papers via embedding arithmetic, plots 2D PCA. Saves `a1_mappings.pt`, `a1_entity_emb.pt`, `a1_relation_emb.pt` |
| `a3_training_kges.py` | Step 2 | Trains TransE, TransH, DistMult, ComplEx + hyperparameter sweep on DistMult. Saves `best_kge_entity_emb.pt`, `best_transe_entity_emb.pt` |
| `a2_improving_transe.py` | `a1` | Analyses hub collapse in TransE, plots pairwise distance distributions |
| `a4_negative_sampling.py` | Step 2 | Computes Bernoulli vs uniform corruption probabilities (tph / hpt) |

```bash
.venv/bin/python a1_basic_model.py
.venv/bin/python a3_training_kges.py
.venv/bin/python a2_improving_transe.py
.venv/bin/python a4_negative_sampling.py
```

---

### Step 4 — B scripts (Graph Neural Networks)

**Run `load_pubmed.py` first (Step 1) to ensure the dataset is cached.**

| Script | Depends on | Description |
|--------|-----------|-------------|
| `b1_forget_about_the_graph.py` | Step 1 | Logistic Regression, SVM, Random Forest on raw TF-IDF node features. Saves `b1-out.csv` |
| `b2_exploiting_the_graph_structure.py` | Step 1 | Trains a 2-layer GCN for node classification, plots training curves |
| `b4_the_more_the_merrier.py` | `b2` (import) | Sweeps GCN depth [2, 4, 8, 16 layers], plots accuracy vs depth |
| `b5_no_mutual_info.py` | `a1`, `a3`, `b2` (import) | GCN with (1) random features and (2) KGE-initialised features |
| `b7_where_are_the_embeddings.py` | Step 1 | Trains GCN, extracts node embeddings, plots PCA and t-SNE. Saves `b7_embs.npy` |
| `b8_embedding_spaces.py` | `a1`, `a3`, `b7` | Compares KGE and GNN nearest-neighbour structure via cosine similarity |

```bash
.venv/bin/python b1_forget_about_the_graph.py
.venv/bin/python b2_exploiting_the_graph_structure.py
.venv/bin/python b4_the_more_the_merrier.py
.venv/bin/python b5_no_mutual_info.py
.venv/bin/python b7_where_are_the_embeddings.py
.venv/bin/python b8_embedding_spaces.py
```

---

## Dependency graph

```
load_pubmed  (downloads dataset)
     │
creating_the_kg
├── a1_basic_model ──────────────────┐
│   └── a2_improving_transe          │
├── a3_training_kges ────────────────┤
├── a4_negative_sampling             │
│                                    │
b1  (needs dataset)                  │
b2  (needs dataset) ──┬─ b4         │
                      └─ b5 ────────┤
b7  (needs dataset) ───── b8 ───────┘
```

---

## Output files

| File | Produced by |
|------|-------------|
| `grl-lab-outputs/tf.pt` | `creating_the_kg.py` |
| `grl-lab-outputs/training.pt` | `creating_the_kg.py` |
| `grl-lab-outputs/a1_mappings.pt` | `a1_basic_model.py` |
| `grl-lab-outputs/best_kge_entity_emb.pt` | `a3_training_kges.py` |
| `grl-lab-outputs/best_transe_entity_emb.pt` | `a3_training_kges.py` |
| `grl-lab-outputs/b1-out.csv` | `b1_forget_about_the_graph.py` |
| `grl-lab-outputs/b7_embs.npy` | `b7_where_are_the_embeddings.py` |
| `images/a1_2D_sketch.png` | `a1_basic_model.py` |
| `images/a2_distance_distributions.png` | `a2_improving_transe.py` |
| `images/gcn_training_curves.png` | `b2_exploiting_the_graph_structure.py` |
| `images/b4_depth_accuracy.png` | `b4_the_more_the_merrier.py` |
| `images/embeddings.png` | `b7_where_are_the_embeddings.py` |

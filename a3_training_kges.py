# ============================================================
# A.3 - Train KGE models, sweep best model (DistMult), save embeddings
#
# Saves all outputs locally to ./grl-lab-outputs/.
# Adjust `output_dir` below if you want a different location.
#
# Prerequisites: `training`, `validation`, `testing`, and `tf`
# (TriplesFactory) must already be defined in the running
# environment (typically from an earlier cell that loads and
# splits the PubMed citation graph).
# ============================================================

import os
import torch
import pandas as pd
from pykeen.pipeline import pipeline

# ------------------------------------------------------------
# 1. Local output directory
# ------------------------------------------------------------

output_dir = './grl-lab-outputs'
os.makedirs(output_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
print(f"Output directory: {os.path.abspath(output_dir)}")


# ------------------------------------------------------------
# 2. Training helper
# ------------------------------------------------------------

def train_and_eval(model_name, run_name=None, **kwargs):
    """
    Train a PyKEEN KGE model and return:
    - the full PyKEEN result object
    - a dictionary with Hits@1, Hits@3 and MRR
    """
    if run_name is None:
        run_name = model_name

    print(f"\n==============================")
    print(f"Running {run_name}")
    print(f"==============================")

    result = pipeline(
        training=training,
        validation=validation,
        testing=testing,
        model=model_name,
        epochs=50,
        random_seed=2026,
        device=device,
        evaluation_kwargs=dict(batch_size=512),
        stopper='early',
        stopper_kwargs=dict(
            frequency=10,
            patience=3,
            relative_delta=0.002,
        ),
        **kwargs,
    )

    metrics = {
        'Hits@1': result.metric_results.get_metric('hits_at_1'),
        'Hits@3': result.metric_results.get_metric('hits_at_3'),
        'MRR':    result.metric_results.get_metric('mean_reciprocal_rank'),
    }

    print(f"{run_name} metrics:")
    print(metrics)

    return result, metrics


# ------------------------------------------------------------
# 3. Train different KGE models
#    Translational: TransE, TransH
#    Semantic / bilinear: DistMult, ComplEx
# ------------------------------------------------------------

base_models = ['TransE', 'TransH', 'DistMult', 'ComplEx']

runs = {}

for model_name in base_models:
    result, metrics = train_and_eval(model_name)
    runs[model_name] = {
        'result': result,
        'metrics': metrics,
        'model_name': model_name,
        'kwargs': {},
    }


# ------------------------------------------------------------
# 4. Hyperparameter sweep on the best base model: DistMult
# ------------------------------------------------------------

sweep_results = {}

# 4.1 Embedding dimension
for dim in [16, 64, 256]:
    run_name = f'DistMult_dim={dim}'

    result, metrics = train_and_eval(
        'DistMult',
        run_name=run_name,
        model_kwargs=dict(
            embedding_dim=dim,
        ),
    )

    sweep_results[run_name] = {
        'result': result,
        'metrics': metrics,
        'model_name': 'DistMult',
        'kwargs': {
            'model_kwargs': dict(embedding_dim=dim),
        },
    }


# 4.2 Number of negative samples per positive sample
for n in [1, 10, 100]:
    run_name = f'DistMult_neg={n}'

    result, metrics = train_and_eval(
        'DistMult',
        run_name=run_name,
        negative_sampler_kwargs=dict(
            num_negs_per_pos=n,
        ),
    )

    sweep_results[run_name] = {
        'result': result,
        'metrics': metrics,
        'model_name': 'DistMult',
        'kwargs': {
            'negative_sampler_kwargs': dict(num_negs_per_pos=n),
        },
    }


# 4.3 Loss margin
# We explicitly use MarginRankingLoss so that the margin parameter is meaningful.
for margin in [0.5, 1.0, 2.0]:
    run_name = f'DistMult_margin={margin}'

    result, metrics = train_and_eval(
        'DistMult',
        run_name=run_name,
        loss='MarginRankingLoss',
        loss_kwargs=dict(
            margin=margin,
        ),
    )

    sweep_results[run_name] = {
        'result': result,
        'metrics': metrics,
        'model_name': 'DistMult',
        'kwargs': {
            'loss': 'MarginRankingLoss',
            'loss_kwargs': dict(margin=margin),
        },
    }


# ------------------------------------------------------------
# 5. Combine all results into one table
# ------------------------------------------------------------

all_runs = {}
all_runs.update(runs)
all_runs.update(sweep_results)

all_metrics = {
    run_name: run_info['metrics']
    for run_name, run_info in all_runs.items()
}

df = pd.DataFrame(all_metrics).T
df = df.sort_values('MRR', ascending=False)

print("\n\n==============================")
print("Final A.3 Results")
print("==============================")
print(df)

# Save results table
metrics_path = os.path.join(output_dir, 'a3_kge_results.csv')
df.to_csv(metrics_path)

print(f"\nSaved metrics table to:")
print(metrics_path)


# ------------------------------------------------------------
# 6. Select the best model by MRR
# ------------------------------------------------------------

best_name = df.index[0]
best_mrr = df.loc[best_name, 'MRR']

best_run = all_runs[best_name]
best_result = best_run['result']
best_model_name = best_run['model_name']

print("\n==============================")
print("Best KGE model")
print("==============================")
print(f"Best run:   {best_name}")
print(f"Model type: {best_model_name}")
print(f"MRR:        {best_mrr:.4f}")


# ------------------------------------------------------------
# 7. Extract and save best entity and relation embeddings
# ------------------------------------------------------------

best_model = best_result.model
best_model.eval()

with torch.no_grad():
    best_entity_emb = best_model.entity_representations[0]().detach().cpu()
    best_relation_emb = best_model.relation_representations[0]().detach().cpu()

entity_emb_path = os.path.join(output_dir, 'best_kge_entity_emb.pt')
relation_emb_path = os.path.join(output_dir, 'best_kge_relation_emb.pt')

torch.save(best_entity_emb, entity_emb_path)
torch.save(best_relation_emb, relation_emb_path)

print("\nSaved best KGE embeddings:")
print(entity_emb_path)
print(relation_emb_path)

print(f"\nEntity embedding shape:   {best_entity_emb.shape}")
print(f"Relation embedding shape: {best_relation_emb.shape}")


# ------------------------------------------------------------
# 8. Save entity/relation ID mappings for later alignment
# ------------------------------------------------------------

entity_to_id = training.entity_to_id
relation_to_id = training.relation_to_id

entity_to_id_path = os.path.join(output_dir, 'entity_to_id.csv')
relation_to_id_path = os.path.join(output_dir, 'relation_to_id.csv')

pd.DataFrame(
    list(entity_to_id.items()),
    columns=['entity', 'id'],
).to_csv(entity_to_id_path, index=False)

pd.DataFrame(
    list(relation_to_id.items()),
    columns=['relation', 'id'],
).to_csv(relation_to_id_path, index=False)

print("\nSaved ID mappings:")
print(entity_to_id_path)
print(relation_to_id_path)


# ------------------------------------------------------------
# 9. Optional: also save the best TransE embeddings for comparison
#    This is useful because the assignment mentions TransE often,
#    but the main embeddings to keep are still the best overall ones.
# ------------------------------------------------------------

transe_candidates = [name for name in df.index if name.startswith('TransE')]

if len(transe_candidates) > 0:
    best_transe_name = transe_candidates[0]
    best_transe_result = all_runs[best_transe_name]['result']

    best_transe_model = best_transe_result.model
    best_transe_model.eval()

    with torch.no_grad():
        best_transe_entity_emb = best_transe_model.entity_representations[0]().detach().cpu()

    best_transe_emb_path = os.path.join(output_dir, 'best_transe_entity_emb.pt')
    torch.save(best_transe_entity_emb, best_transe_emb_path)

    print("\nAlso saved best TransE entity embeddings:")
    print(f"Best TransE run: {best_transe_name}")
    print(best_transe_emb_path)


# ------------------------------------------------------------
# 10. Short interpretation helper
# ------------------------------------------------------------

print("\n==============================")
print("Short interpretation")
print("==============================")

print(f"""
The best-performing run according to MRR is: {best_name}.

Since the best base model was DistMult, the hyperparameter sweep was performed on DistMult.
This is more coherent than sweeping TransE, because the goal of A.3 is not only to compare
models but also to keep the best-performing embeddings for later tasks.

The saved file 'best_kge_entity_emb.pt' should therefore be used later for the KGE
initialization in the GNN part.
""")




import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

def get_metrics(v):
    """
    Handles:
    - old format: (result, metrics)
    - metrics dict directly
    - new unified format: {'result': ..., 'metrics': ..., ...}
    """
    if isinstance(v, tuple):
        return v[1]
    if isinstance(v, dict) and 'metrics' in v:
        return v['metrics']
    return v

def extract_sweep(prefix):
    keys = sorted(
        [k for k in sweep_results if k.startswith(prefix)],
        key=lambda k: float(k.split('=')[1])
    )

    xs = [float(k.split('=')[1]) for k in keys]
    hits1 = [get_metrics(sweep_results[k])['Hits@1'] for k in keys]
    hits3 = [get_metrics(sweep_results[k])['Hits@3'] for k in keys]
    mrr   = [get_metrics(sweep_results[k])['MRR']    for k in keys]

    return xs, hits1, hits3, mrr

plots = [
    ('DistMult_dim=',    'Embedding dimension',          'dim',    True),
    ('DistMult_neg=',    'Negative samples per positive', 'negs',   True),
    ('DistMult_margin=', 'Loss margin',                   'margin', False),
]

for prefix, xlabel, name, logx in plots:
    xs, h1, h3, mrr = extract_sweep(prefix)

    fig, ax = plt.subplots(figsize=(5, 4))

    ax.plot(xs, h1,  marker='o', label='Hits@1')
    ax.plot(xs, h3,  marker='s', label='Hits@3')
    ax.plot(xs, mrr, marker='^', label='MRR')

    if logx:
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())

    ax.set_xticks(xs)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Metric value')
    ax.set_title(f'Effect of {xlabel.lower()} on DistMult')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'images/distmult_sweep_{name}.png', dpi=150, bbox_inches='tight')
    plt.show()
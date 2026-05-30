from creating_the_kg import *
import torch

# Obtain the corruption probabilities here
triples = tf.mapped_triples
cites_id = tf.relation_to_id['cites']
cite_triples = triples[triples[:, 1] == cites_id]

# Out-degree per head (only over heads that appear), in-degree per tail
_, out_deg = torch.unique(cite_triples[:, 0], return_counts=True) # Number of times it appears on the left == number of times the paper cites
_, in_deg  = torch.unique(cite_triples[:, 2], return_counts=True) # Number of times it appears on the right == number of times the paper is cited

tph = out_deg.float().mean().item()
hpt = in_deg.float().mean().item()

p_head = tph / (tph + hpt)
p_tail = hpt / (tph + hpt)

print(f"tph = {tph:.3f}, hpt = {hpt:.3f}")
print(f"Bernoulli: P(corrupt head) = {p_head:.3f}, P(corrupt tail) = {p_tail:.3f}")
print(f"Uniform:   P(corrupt head) = 0.500, P(corrupt tail) = 0.500")
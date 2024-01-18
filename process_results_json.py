import json
import numpy as np
import torch

expval_budget = 3
file = "/home/azzolin/sedignn/LECI_fork/storage/metric_results/id_repr_LECIGIN_3l_meanpool_0.5dp_mitig_backboneNone_mitig_samplingfeatavgedgeattnmean_110_nec_idval_budgetsamples1000_expbudget3.json"
with open(f"{file}", "r") as f:
    results = json.load(f)

print(results.keys())

for w in results.keys():
    sa = torch.tensor(results[w])

    if "nec" in file:
        vals = 1 - torch.exp(-sa)
    else:
        vals = torch.exp(-sa)
    c = 0
    means = []
    while c < len(vals):
        samples = []
        for e in range(expval_budget):
            samples.append(vals[c+e])
        means.append(np.mean(samples))
        c+= expval_budget
    if "nec" in file:
        print(f"NEC for w>={w}: {np.mean(means)}")
    else:
        print(f"SUFF for w>={w}: {np.mean(means)}")
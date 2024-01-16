import json
import numpy as np
import torch

file = "/home/azzolin/sedignn/LECI_fork/storage/metric_results/id_repr_LECIGIN_3l_meanpool_0.5dp_mitig_backboneNone_mitig_samplingfeatavgedgeattnmean_110_suff_val.json"
with open(f"{file}", "r") as f:
    results = json.load(f)

print(results.keys())

for w in results.keys():
    sa = torch.tensor(results[w])

    suff = torch.exp(-sa)
    print(f"SUFF for w>={w}: {suff.mean()}")
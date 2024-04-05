import json
import numpy as np
import torch
import random

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from scipy.stats import pearsonr, spearmanr, hmean
from sklearn.preprocessing import RobustScaler

# "CIGA":{
#     "id_val": {
#         "acc": [],
#         "plaus_f1": [],
#         "wiou": [],
#         "suff": [],
#         "nec": [],
#         "faith_aritm": [],
        # "faith_armon": [],
        # "faith_gmean": []
#     },
#     "val": {
#         "acc": [],
#         "plaus_f1": [],
#         "wiou": [],
#         "suff": [],
#         "nec": [],
        # "faith_aritm": [],
        # "faith_armon": [],
        # "faith_gmean": []
#     },
#     "test": {
#         "acc": [],
#         "plaus_f1": [],
#         "wiou": [],
#         "suff": [],
#         "nec": [],
        # "faith_aritm": [],
        # "faith_armon": [],
        # "faith_gmean": []
#     }
# },

def armonic(a,b):
    return 2 * (a*b) / (a+b)

def pick_best_faith(data, where, faith_type):
    if len(data[where][faith_type]) == 1:
        return 0
    else:
        return np.argmax(data[where][faith_type][:-1])

plaus_type = "wiou"
file_name = "suff++_old" #suff_simple_old
pick_acc = "entire_model" #"best_r"/"entire_model"
with open(f"storage/metric_results/aggregated_id_results_{file_name}.json", "r") as jsonFile:
    data = json.load(jsonFile)
with open(f"storage/metric_results/acc_plaus.json", "r") as jsonFile:
    acc_plaus = json.load(jsonFile)

markers = {
    "GOODMotif basis": "o",
    "GOODMotif2 basis": "*",
    "GOODMotif size": "^",
    "GOODSST2 length": "3",
    "GOODTwitter length": "4",
    "GOODHIV scaffold": "D",
    "LBAPcore assay": "d",
    "GOODCMNIST color": "p"
}
colors = {
    "LECIGIN": "blue",
    "CIGAGIN": "orange", 
    "GSATGIN": "green",
    "LECIvGIN": "deepskyblue",
    "CIGAvGIN": "moccasin", 
    "GSATvGIN": "lime",
}

print(data.keys())




def scatter_trio():
    num_cols = 4
    num_rows = 3
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, 15))

    for j, faith_type in enumerate(["faith_aritm_L1", "faith_armon_L1", "faith_gmean_L1"]):
        for i, (split_metric, split_acc) in enumerate([("id_val", "id_val"), ("test", "test"), ("id_val", "val"), ("val", "test")]):
            for dataset in ["GOODMotif basis", "GOODMotif2 basis", "GOODMotif size"]: #data.keys()
                for model in ["LECIGIN", "CIGAGIN", "GSATGIN", "LECIvGIN", "CIGAvGIN", "GSATvGIN"]:
                    if not model in data[dataset].keys():
                        continue
                    if not faith_type in data[dataset][model][split_metric].keys():
                        continue

                    best_r = pick_best_faith(data[dataset][model], split_metric, faith_type)
                    faith = data[dataset][model][split_metric][faith_type][best_r]
                    plaus = acc_plaus[dataset][model][split_metric][plaus_type][best_r]
                    acc   = acc_plaus[dataset][model][split_acc]["acc"][-1 if pick_acc == "entire_model" else best_r]

                    axs[j%num_rows, i%num_cols].scatter(faith, plaus, marker=markers[dataset], label=model, c=colors[model]) #c=acc, vmin=0., vmax=1.
                    axs[j%num_rows, i%num_cols].annotate(f"{acc:.2f}", (faith, plaus + (-1)**(random.randint(0,1))*random.randint(1,4)*0.005), fontsize=7)
                    axs[j%num_rows, i%num_cols].grid(visible=True, alpha=0.5)
                    axs[j%num_rows, i%num_cols].set_xlim(0., 1.)
                    axs[j%num_rows, i%num_cols].set_ylim(0., 1.1)
                    axs[j%num_rows, i%num_cols].set_ylabel(f"{plaus_type}")
                    axs[j%num_rows, i%num_cols].set_xlabel(f"{faith_type}")
                    axs[j%num_rows, i%num_cols].set_title(f"metric: {split_metric} - acc: {split_acc}")

    legend_elements = []
    for key, value in markers.items():
        legend_elements.append(
            Line2D([0], [0], marker=value, color='w', label=key, markerfacecolor='grey', markersize=15)
        )
    for key, value in colors.items():
        legend_elements.append(
            Patch(facecolor=value, label=key)
        )

    plt.suptitle(f"{file_name} - pick accuracy: {pick_acc}")
    axs[1, -1].legend(handles=legend_elements, loc='upper left') #, loc='center'
    # plt.colorbar()
    plt.savefig("GOOD/kernel/pipelines/plots/illustrations/automatic/scatter_trio.png")
    plt.close()


def low_discrepancy():
    ##
    # Faith as a necessary condition for low discrepancy from ID and OOD test acc
    ##

    num_cols = 2
    num_rows = 3
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(9, 15))

    for j, faith_type in enumerate(["faith_aritm_L1", "faith_armon_L1", "faith_gmean_L1"]):
        for i, (split_metric_id, split_metric_ood) in enumerate([("id_val", "test"), ("val", "test")]):
            split_acc = "test"
            acc_coll, combined_coll = [], []
            for dataset in ["GOODMotif basis", "GOODMotif2 basis", "GOODMotif size"]: #data.keys()

                # if dataset == "Motif size":
                #     continue

                for model in ["LECIGIN", "CIGAGIN", "GSATGIN", "LECIvGIN", "CIGAvGIN", "GSATvGIN"]:
                    if not model in data[dataset].keys():
                        continue
                    if not faith_type in data[dataset][model][split_metric_id].keys():
                        continue

                    best_r = pick_best_faith(data[dataset][model], split_metric_id, faith_type)
                    faith_id   = np.array(data[dataset][model][split_metric_id][faith_type])[best_r]
                    faith_ood  = np.array(data[dataset][model][split_metric_ood][faith_type])[best_r]
                    combined = abs(faith_id - faith_ood)
                    
                    # best_r = pick_best_faith(data[dataset][model], split_metric, "wiou")
                    # plaus_id      = np.array(acc_plaus[dataset][model][split_metric_id]["wiou"])[-1]
                    # plaus_ood      = np.array(acc_plaus[dataset][model][split_metric_ood]["wiou"])[-1]
                    # combined = armonic(plaus_id, plaus_ood)
                    
                    # combined = hmean([faith_id, faith_ood, plaus_id, plaus_ood])
                    if isinstance(combined, float):
                        combined_coll.append(combined)
                    else:
                        combined_coll.extend(combined)

                    acc_id    = acc_plaus[dataset][model][split_metric_id]["acc"][-1 if pick_acc == "entire_model" else best_r]
                    acc_ood   = acc_plaus[dataset][model][split_metric_ood]["acc"][-1 if pick_acc == "entire_model" else best_r]
                    
                    acc = abs(acc_id - acc_ood)
                    if isinstance(acc, float):
                        acc_coll.append(acc)
                    else:
                        acc_coll.extend(acc)
                    
                    axs[j%num_rows, i%num_cols].scatter(combined, acc, marker=markers[dataset], label=model, c=colors[model])
                    # axs[j%num_rows, i%num_cols].annotate(f"{acc:.2f}", (faith_id, faith_ood + (-1)**(random.randint(0,1))*random.randint(1,4)*0.005), fontsize=7)
                    axs[j%num_rows, i%num_cols].grid(visible=True, alpha=0.5)
                    axs[j%num_rows, i%num_cols].set_xlim(0.0, 1.)
                    axs[j%num_rows, i%num_cols].set_ylim(-0.2, 1.)
                    axs[j%num_rows, i%num_cols].set_ylabel(f"Acc abs difference ({split_metric_id} - {split_metric_ood})")
                    axs[j%num_rows, i%num_cols].set_xlabel(f"Faith abs difference ({split_metric_id}, {split_metric_ood}) ({faith_type})")
                    axs[j%num_rows, i%num_cols].set_title(f"")
            if len(acc_coll) > 0 and len(combined_coll) > 0:
                combined_coll, acc_coll = np.array(acc_coll), np.array(combined_coll)
                pcc = pearsonr(acc_coll, combined_coll)
                axs[j%num_rows, i%num_cols].annotate(f"PCC: {pcc.statistic:.2f} ({pcc.pvalue:.2f})", (0.8, 0.8), fontsize=7)

    plt.suptitle(f"{file_name} - pick accuracy: {pick_acc}")
    # axs[2, -1].legend(handles=legend_elements, loc='upper left') #, loc='center'
    # plt.colorbar()
    plt.savefig("GOOD/kernel/pipelines/plots/illustrations/automatic/low_discrepancy.png")
    plt.close()










if __name__ == "__main__":
    low_discrepancy()
    # scatter_trio()    





#####################################################################################################################################################
# PAST PLOTS made for 'manual' #
#####################################################################################################################################################







##
# Generate plot combining faith and plaus into a unique metric, via hmean for example
# Then compute also PCC
##


# num_cols = 6
# num_rows = 3
# fig, axs = plt.subplots(num_rows, num_cols, figsize=(18, 15))

# for j, faith_type in enumerate(["faith_aritm", "faith_armon", "faith_gmean"]):
#     for i, (split_metric, split_acc) in enumerate([("id_val", "id_val"), ("test", "test"), ("id_val", "val"), ("val", "test"), ("id_val", "test"), ("val", "val")]):
#         combined_coll, acc_coll = [], []
#         for dataset in data.keys():
            
#             # if dataset == "Motif size":
#             #     continue

#             for model in ["LECI", "CIGA", "GSAT"]:
#                 if not faith_type in data[dataset][model][split_metric].keys():
#                     continue

#                 best_r = pick_best_faith(data[dataset][model], split_metric, faith_type)
#                 faith = np.array(data[dataset][model][split_metric][faith_type])[:]
#                 plaus = np.array(data[dataset][model][split_metric][plaus_type])[:]
#                 combined = armonic(faith, plaus)
#                 if isinstance(combined, float):
#                     combined_coll.append(combined)
#                 else:
#                     combined_coll.extend(combined)

#                 if pick_acc == "entire_model":
#                     acc   = data[dataset][model][split_acc]["acc"][:] #TODO: pick all but last including CIGA
#                 else:
#                     acc   = data[dataset][model][split_acc]["acc"][best_r]
#                 if isinstance(combined, float):
#                     acc_coll.append(acc)
#                 else:
#                     acc_coll.extend(acc)                

#                 axs[j%num_rows, i%num_cols].scatter(acc, combined, marker=markers[dataset], label=model, c=colors[model])
#                 # axs[j%num_rows, i%num_cols].annotate(f"{acc:.2f}", (faith, plaus + (-1)**(random.randint(0,1))*random.randint(1,4)*0.005), fontsize=7)
#                 axs[j%num_rows, i%num_cols].grid(visible=True, alpha=0.5)
#                 axs[j%num_rows, i%num_cols].set_xlim(0., 1.)
#                 axs[j%num_rows, i%num_cols].set_ylim(0., 1.1)
#                 axs[j%num_rows, i%num_cols].set_ylabel(f"hmean({plaus_type}, {faith_type})")
#                 axs[j%num_rows, i%num_cols].set_xlabel(f"Acc")
#                 axs[j%num_rows, i%num_cols].set_title(f"metric: {split_metric} - acc: {split_acc}")

#         if len(acc_coll) > 0 and len(combined_coll) > 0:
#             combined_coll, acc_coll = np.array(acc_coll), np.array(combined_coll)
#             combined_coll, acc_coll = (combined_coll - np.min(combined_coll)) / (np.max(combined_coll) - np.min(combined_coll)), (acc_coll - np.min(acc_coll)) / (np.max(acc_coll) - np.min(acc_coll))
#             pcc = spearmanr(acc_coll, combined_coll)
#             axs[j%num_rows, i%num_cols].annotate(f"PCC: {pcc.statistic:.2f} ({pcc.pvalue:.2f})", (0.5, 1.), fontsize=7)

# # legend_elements = []
# # for key, value in markers.items():
# #     legend_elements.append(
# #         Line2D([0], [0], marker=value, color='w', label=key, markerfacecolor='grey', markersize=15)
# #     )
# # for key, value in colors.items():
# #     legend_elements.append(
# #         Patch(facecolor=value, label=key)
# #     )

# plt.suptitle(f"{file_name} - pick accuracy: {pick_acc}")
# # axs[2, -1].legend(handles=legend_elements, loc='upper left') #, loc='center'
# # plt.colorbar()
# plt.savefig("GOOD/kernel/pipelines/plots/illustrations/automatic/hmean_faith_plaus.png")
# plt.close()
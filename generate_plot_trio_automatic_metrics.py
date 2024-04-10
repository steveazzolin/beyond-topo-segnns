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
    "GOODSST2 length": "v",
    "GOODTwitter length": "|",
    "GOODHIV scaffold": "D",
    "LBAPcore assay": "d",
    "GOODCMNIST color": "p" # did not converge
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
    num_rows = 1
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 5))
    datasets = ["GOODMotif basis", "GOODMotif2 basis", "GOODMotif size", "GOODSST2 length", "GOODTwitter length", "GOODHIV scaffold"]

    for j, faith_type in enumerate(["faith_armon_L1"]):
        for i, (split_metric_id, split_metric_ood) in enumerate([("id_val", "test"), ("val", "test")]):
            split_acc = "test"
            acc_coll, combined_coll = [], []
            for dataset in datasets: #data.keys()

                # if dataset == "Motif size":
                #     continue

                for model in ["LECIGIN", "CIGAGIN", "GSATGIN", "LECIvGIN", "CIGAvGIN", "GSATvGIN"]:
                    if not model in data[dataset].keys():
                        continue
                    if not faith_type in data[dataset][model][split_metric_id].keys():
                        continue

                    best_r = pick_best_faith(data[dataset][model], split_metric_id, "faith_armon_L1")
                    faith_id   = np.array(data[dataset][model][split_metric_id][faith_type])[best_r]
                    best_r = pick_best_faith(data[dataset][model], split_metric_ood, "faith_armon_L1")
                    faith_ood  = np.array(data[dataset][model][split_metric_ood][faith_type])[best_r]
                    combined = faith_id - faith_ood
                    # combined = faith_id + faith_ood
                    
                    # best_r = pick_best_faith(data[dataset][model], split_metric, "wiou")
                    # plaus_id       = np.nan_to_num(np.array(acc_plaus[dataset][model][split_metric_id]["wiou"]))[-1]
                    # plaus_ood      = np.nan_to_num(np.array(acc_plaus[dataset][model][split_metric_ood]["wiou"]))[-1]
                    # combined = armonic(plaus_id, plaus_ood)
                    
                    # combined = hmean([faith_id, faith_ood, plaus_id, plaus_ood])
                    if isinstance(combined, float):
                        combined_coll.append(combined)
                    else:
                        combined_coll.extend(combined)

                    acc_id    = acc_plaus[dataset][model][split_metric_id]["acc"][-1 if pick_acc == "entire_model" else best_r]
                    acc_ood   = acc_plaus[dataset][model][split_metric_ood]["acc"][-1 if pick_acc == "entire_model" else best_r]
                    
                    acc = acc_id - acc_ood
                    if isinstance(acc, float):
                        acc_coll.append(acc)
                    else:
                        acc_coll.extend(acc)
                    
                    axs[i%num_cols].scatter(combined, acc, marker=markers[dataset], label=model, c=colors[model])
                    axs[i%num_cols].grid(visible=True, alpha=0.5)
                    # axs[i%num_cols].set_xlim(0.0, 1.)
                    # axs[i%num_cols].set_ylim(-0.2, 1.)
                    axs[i%num_cols].set_xlim(-0.2, 0.75)
                    axs[i%num_cols].set_ylim(-0.2, 1.)
                    axs[i%num_cols].set_ylabel(f"Performance difference {split_metric_id} - {split_metric_ood}")
                    axs[i%num_cols].set_xlabel(f"Faithfulness difference {split_metric_id} - {split_metric_ood}")
                    axs[i%num_cols].set_title(f"")
            if len(acc_coll) > 0 and len(combined_coll) > 0:
                combined_coll, acc_coll = np.array(combined_coll), np.array(acc_coll)
                # pcc = pearsonr(acc_coll, combined_coll)
                # axs[i%num_cols].annotate(f"PCC: {pcc.statistic:.2f} ({pcc.pvalue:.2f})", (0.0, 0.8), fontsize=7)
                m, b = np.polyfit(combined_coll, acc_coll, 1)
                x = combined_coll.tolist() + [-0.2, 4]
                axs[i%num_cols].plot(x, np.poly1d((m, b))(x), "r--", alpha=0.5)

    legend_elements = []
    for dataset in datasets:
        legend_elements.append(
            Line2D([0], [0], marker=markers[dataset], color='w', label=dataset, markerfacecolor='grey', markersize=15)
        )
    for model in ["LECIGIN", "CIGAGIN", "GSATGIN"]:
        legend_elements.append(
            Patch(facecolor=colors[model], label=model.replace("GIN", ""))
        )
    axs[1].legend(handles=legend_elements, loc='upper right') #, loc='center'
    plt.suptitle(f"{file_name} - pick accuracy: {pick_acc}")
    plt.savefig("GOOD/kernel/pipelines/plots/illustrations/automatic/low_discrepancy.png")
    plt.savefig("GOOD/kernel/pipelines/plots/illustrations/automatic/pdfs/low_discrepancy.pdf")
    plt.close()

def lower_bound_plaus():
    ##
    # Faith as a necessary condition for low discrepancy from ID and OOD test acc
    ##

    num_cols = 2
    num_rows = 1
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 5))

    for j, faith_type in enumerate(["faith_armon_L1"]):
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

                    best_r = pick_best_faith(data[dataset][model], split_metric_ood, faith_type)
                    faith_ood  = np.array(data[dataset][model][split_metric_ood][faith_type])[best_r]
                    combined = faith_id + faith_ood
                    
                    # best_r = pick_best_faith(data[dataset][model], split_metric, "wiou")
                    plaus_id       = np.nan_to_num(np.array(acc_plaus[dataset][model][split_metric_id]["wiou"]))[-1]
                    plaus_ood      = np.nan_to_num(np.array(acc_plaus[dataset][model][split_metric_ood]["wiou"]))[-1]
                    combined = combined + plaus_id + plaus_ood # show empirically the lower bound
                    
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
                    
                    axs[i%num_cols].scatter(combined, acc, marker=markers[dataset], label=model, c=colors[model])
                    # axs[i%num_cols].annotate(f"{acc:.2f}", (faith_id, faith_ood + (-1)**(random.randint(0,1))*random.randint(1,4)*0.005), fontsize=7)
                    axs[i%num_cols].grid(visible=True, alpha=0.5)
                    axs[i%num_cols].set_xlim(0.0, 4.)
                    axs[i%num_cols].set_ylim(-0.2, 1.)
                    axs[i%num_cols].set_ylabel(f"Accuracy difference |{split_metric_id} - {split_metric_ood}|")
                    axs[i%num_cols].set_xlabel(f"Faithfulness + Plausibility ({split_metric_id}, {split_metric_ood})")
                    axs[i%num_cols].set_title(f"")
            if len(acc_coll) > 0 and len(combined_coll) > 0:
                combined_coll, acc_coll = np.array(combined_coll), np.array(acc_coll)
                # pcc = pearsonr(combined_coll, acc_coll)
                # axs[j%num_rows, i%num_cols].annotate(f"PCC: {pcc.statistic:.2f} ({pcc.pvalue:.2f})", (0.8, 0.8), fontsize=7)
                m, b = np.polyfit(combined_coll, acc_coll, 1)
                x = combined_coll.tolist() + [0, 4]
                axs[i%num_cols].plot(x, np.poly1d((m, b))(x), "r--", alpha=0.5)

    legend_elements = []
    for dataset in ["GOODMotif basis", "GOODMotif2 basis", "GOODMotif size"]:
        legend_elements.append(
            Line2D([0], [0], marker=markers[dataset], color='w', label=dataset, markerfacecolor='grey', markersize=15)
        )
    for model in ["LECIGIN", "CIGAGIN", "GSATGIN"]:
        legend_elements.append(
            Patch(facecolor=colors[model], label=model.replace("GIN", ""))
        )
    axs[1].legend(handles=legend_elements, loc='upper right') #, loc='center'

    plt.suptitle(f"{file_name} - pick accuracy: {pick_acc}")
    plt.savefig("GOOD/kernel/pipelines/plots/illustrations/automatic/lower_bound.png")
    plt.savefig("GOOD/kernel/pipelines/plots/illustrations/automatic/pdfs/lower_bound.pdf")
    plt.close()

def lower_bound_unsup():
    ##
    # Faith as a necessary condition for low discrepancy from ID and OOD test acc
    ##

    num_cols = 2
    num_rows = 3
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(9, 15))

    for j, faith_type in enumerate(["suff++_L1", "faith_armon_L1", "faith_gmean_L1"]):
        for i, (split_metric_id, split_metric_ood) in enumerate([("id_val", "test"), ("val", "test")]):
            split_acc = "test"
            acc_coll, combined_coll = [], []
            for dataset in ["GOODMotif basis", "GOODMotif2 basis", "GOODMotif size", "GOODSST2 length", "GOODTwitter length", "GOODHIV scaffold", "GOODCMNIST color"]: #data.keys()
                
                # if dataset == "Motif size":
                #     continue

                for model in ["LECIGIN", "CIGAGIN", "GSATGIN", "LECIvGIN", "CIGAvGIN", "GSATvGIN"]:
                    if not model in data[dataset].keys():
                        continue
                    if not faith_type in data[dataset][model][split_metric_id].keys():
                        continue

                    best_r = pick_best_faith(data[dataset][model], split_metric_id, faith_type)
                    faith_id   = np.array(data[dataset][model][split_metric_id][faith_type])[best_r]

                    best_r = pick_best_faith(data[dataset][model], split_metric_ood, faith_type)
                    faith_ood  = np.array(data[dataset][model][split_metric_ood][faith_type])[best_r]
                    combined = (faith_id + faith_ood)
                    
                    # best_r = pick_best_faith(data[dataset][model], split_metric, "wiou")
                    # plaus_id       = np.nan_to_num(np.array(acc_plaus[dataset][model][split_metric_id]["wiou"]))[-1]
                    # plaus_ood      = np.nan_to_num(np.array(acc_plaus[dataset][model][split_metric_ood]["wiou"]))[-1]
                    # combined = combined # + plaus_id + plaus_ood # show empirically the lower bound
                    
                    # combined = hmean([faith_id, faith_ood, plaus_id, plaus_ood])
                    if isinstance(combined, float):
                        combined_coll.append(combined)
                    else:
                        combined_coll.extend(combined)

                    acc_id    = acc_plaus[dataset][model][split_metric_id]["acc_ori"] #[-1 if pick_acc == "entire_model" else best_r]
                    acc_ood   = acc_plaus[dataset][model][split_metric_ood]["acc_ori"] #[-1 if pick_acc == "entire_model" else best_r]
                    acc_baseline_id  = acc_plaus[dataset]["GIN"][split_metric_id]["acc"][-1]
                    acc_baseline_ood = acc_plaus[dataset]["GIN"][split_metric_ood]["acc"][-1]
                    
                    acc = abs(acc_id - acc_ood)
                    # acc = acc / abs(acc_baseline_id - acc_baseline_ood)
                    if isinstance(acc, float):
                        acc_coll.append(acc)
                    else:
                        acc_coll.extend(acc)
                    
                    axs[j%num_rows, i%num_cols].scatter(combined, acc, marker=markers[dataset], label=model, c=colors[model])
                    # axs[j%num_rows, i%num_cols].annotate(f"{acc:.2f}", (faith_id, faith_ood + (-1)**(random.randint(0,1))*random.randint(1,4)*0.005), fontsize=7)
                    axs[j%num_rows, i%num_cols].grid(visible=True, alpha=0.5)
                    axs[j%num_rows, i%num_cols].set_xlim(0.0, 1.5)
                    axs[j%num_rows, i%num_cols].set_ylim(0.0, 1.)
                    axs[j%num_rows, i%num_cols].set_ylabel(f"Acc abs difference ({split_metric_id} - {split_metric_ood})")
                    axs[j%num_rows, i%num_cols].set_xlabel("$Faith_{id}$ + $Faith_{ood}$" + f" ({split_metric_id}, {split_metric_ood}) ({faith_type})")
                    axs[j%num_rows, i%num_cols].set_title(f"")
            if len(acc_coll) > 0 and len(combined_coll) > 0:
                combined_coll, acc_coll = np.array(combined_coll), np.array(acc_coll)
                # pcc = pearsonr(combined_coll, acc_coll)
                # axs[j%num_rows, i%num_cols].annotate(f"PCC: {pcc.statistic:.2f} ({pcc.pvalue:.2f})", (0.8, 0.8), fontsize=7)
                # m, b = np.polyfit(combined_coll, acc_coll, 1)
                # x = combined_coll.tolist() + [0, 2]
                # axs[j%num_rows, i%num_cols].plot(x, np.poly1d((m, b))(x), "r--", alpha=0.5)

    plt.suptitle(f"{file_name} - pick accuracy: {pick_acc}")
    # axs[2, -1].legend(handles=legend_elements, loc='upper left') #, loc='center'
    plt.savefig("GOOD/kernel/pipelines/plots/illustrations/automatic/lower_bound_unsup.png")
    plt.close()

def faith_acc_gain():
    ##
    # Faith as a necessary condition for low discrepancy from ID and OOD test acc
    ##

    num_cols = 3
    num_rows = 1
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 5))

    for j, faith_type in enumerate(["faith_armon_L1"]):
        for i, what_correlate in enumerate(["Faithfulness", "Plausibility", "Armonic Mean"]):
            split_acc = "test"
            split_ood = "val"
            acc_coll, combined_coll = [], []
            for dataset in ["GOODMotif basis", "GOODMotif2 basis", "GOODMotif size"]: #data.keys()
                #, "GOODSST2 length", "GOODTwitter length", "GOODHIV scaffold", "GOODCMNIST color"
                # if dataset == "Motif size":
                #     continue

                for model in ["LECIGIN", "CIGAGIN", "GSATGIN", "LECIvGIN", "CIGAvGIN", "GSATvGIN"]:
                    if not model in data[dataset].keys():
                        continue
                    if not faith_type in data[dataset][model][split_ood].keys():
                        continue

                    best_r = pick_best_faith(data[dataset][model], "id_val", faith_type)
                    faith_id  = np.array(data[dataset][model]["id_val"][faith_type])[best_r]                    
                    best_r = pick_best_faith(data[dataset][model], split_ood, faith_type)
                    faith_ood  = np.array(data[dataset][model][split_ood][faith_type])[best_r]                    

                    if what_correlate == "Faithfulness":
                        combined = faith_ood
                    elif what_correlate == "Plausibility":
                        plaus_id      = np.nan_to_num(np.array(acc_plaus[dataset][model]["id_val"]["wiou"]))[-1]
                        plaus_ood      = np.nan_to_num(np.array(acc_plaus[dataset][model][split_ood]["wiou"]))[-1]
                        combined = plaus_ood
                    elif what_correlate == "Armonic Mean":
                        plaus_id      = np.nan_to_num(np.array(acc_plaus[dataset][model]["id_val"]["wiou"]))[-1]
                        plaus_ood      = np.nan_to_num(np.array(acc_plaus[dataset][model][split_ood]["wiou"]))[-1]
                        combined = hmean([faith_ood, plaus_ood])
                    
                    if isinstance(combined, float):
                        combined_coll.append(combined)
                    else:
                        combined_coll.extend(combined)

                    acc_ood_gin    = acc_plaus[dataset]["GIN"][split_ood]["acc"][-1 if pick_acc == "entire_model" else best_r]
                    acc_ood_model   = acc_plaus[dataset][model][split_ood]["acc_ori"]
                    
                    acc = acc_ood_model - acc_ood_gin
                    if isinstance(acc, float):
                        acc_coll.append(acc)
                    else:
                        acc_coll.extend(acc)
                    
                    axs[i%num_cols].scatter(combined, acc, marker=markers[dataset], label=model, c=colors[model])
                    # axs[j%num_rows, i%num_cols].annotate(f"{acc:.2f}", (faith_id, faith_ood + (-1)**(random.randint(0,1))*random.randint(1,4)*0.005), fontsize=7)
                    axs[i%num_cols].grid(visible=True, alpha=0.5)
                    if what_correlate == "Faithfulness":
                        axs[i%num_cols].set_xlim(0.4, 0.8)
                        axs[i%num_cols].set_ylim(-0.2, 0.9)
                    elif what_correlate == "Plausibility":
                        axs[i%num_cols].set_xlim(0., 1.1)
                        axs[i%num_cols].set_ylim(-0.2, 0.9)
                    else:
                        axs[i%num_cols].set_xlim(0., 1)
                        axs[i%num_cols].set_ylim(-0.2, 0.9)
                    axs[i%num_cols].set_ylabel(f"Acc gain wrt ERM ({split_ood})")
                    axs[i%num_cols].set_xlabel(f"{what_correlate} ({split_ood})")
                    axs[i%num_cols].set_title(f"")
            if len(acc_coll) > 0 and len(combined_coll) > 0:
                combined_coll, acc_coll = np.array(combined_coll), np.array(acc_coll)
                pcc = pearsonr(combined_coll, acc_coll)
                axs[i%num_cols].annotate(f"PCC: {pcc.statistic:.2f} ({pcc.pvalue:.2f})", (0.6, 0.6), fontsize=7)
                # m, b = np.polyfit(combined_coll, acc_coll, 1)
                # x = combined_coll.tolist() + [0, 2]
                # axs[i%num_cols].plot(x, np.poly1d((m, b))(x), "r--", alpha=0.3)

    legend_elements = []
    for dataset in ["GOODMotif basis", "GOODMotif2 basis", "GOODMotif size"]:
        legend_elements.append(
            Line2D([0], [0], marker=markers[dataset], color='w', label=dataset, markerfacecolor='grey', markersize=15)
        )
    for model in ["LECIGIN", "CIGAGIN", "GSATGIN"]:
        legend_elements.append(
            Patch(facecolor=colors[model], label=model.replace("GIN", ""))
        )
    axs[1].legend(handles=legend_elements, loc='upper right')
    plt.suptitle(f"{file_name} - pick accuracy: {pick_acc}")
    plt.savefig("GOOD/kernel/pipelines/plots/illustrations/automatic/faith_acc_gain.png")
    plt.savefig("GOOD/kernel/pipelines/plots/illustrations/automatic/pdfs/faith_acc_gain.pdf")
    plt.close()

def compare_faith_mitigations():
    ##
    # Show how mitigation strategies impacted FAITH
    ##

    

        # with open(f"storage/metric_results/acc_plaus.json", "r") as jsonFile:
        #     acc_plaus = json.load(jsonFile)
        # num_cols = 2
        # num_rows = 3
        # fig, axs = plt.subplots(num_rows, num_cols, figsize=(9, 15))

    for j, faith_type in enumerate(["faith_armon_L1"]):
        for i, split_metric in enumerate(["test"]):
            for dataset in ["GOODMotif basis", "GOODMotif2 basis", "GOODMotif size", "GOODSST2 length", "GOODTwitter length", "GOODHIV scaffold", "LBAPcore assay", "GOODCMNIST color"]:
                for file_name in ["suff++_old_mitigreadout_weighted_mitigvirtual_weighted"]: #"suff++_old", "suff++_old_mitigreadout_weighted", "suff++_old_mitigreadout_weighted_mitigvirtual_weighted"
                    with open(f"storage/metric_results/aggregated_id_results_{file_name}.json", "r") as jsonFile:
                        data = json.load(jsonFile)
                    for model in ["LECIGIN", "LECIvGIN"]:
                        if not dataset in data.keys() or not model in data[dataset].keys():
                            continue
                        if not faith_type in data[dataset][model][split_metric].keys():
                            continue

                        best_r = pick_best_faith(data[dataset][model], split_metric, faith_type)
                        faith   = np.array(data[dataset][model][split_metric][faith_type])[best_r]
                        
                        print(f"{file_name:<32}\t{dataset:<20}\t{model:<10}\t{faith_type:<15}\t{split_metric:<6} = {faith:.3f}")
                        
                        # axs[j%num_rows, i%num_cols].scatter(combined, acc, marker=markers[dataset], label=model, c=colors[model])
                        # axs[j%num_rows, i%num_cols].grid(visible=True, alpha=0.5)
                        # axs[j%num_rows, i%num_cols].set_xlim(0.0, 1.5)
                        # axs[j%num_rows, i%num_cols].set_ylim(0.0, 1.)
                        # axs[j%num_rows, i%num_cols].set_ylabel(f"Acc abs difference ({split_metric_id} - {split_metric_ood})")
                        # axs[j%num_rows, i%num_cols].set_xlabel("$Faith_{id}$ + $Faith_{ood}$" + f" ({split_metric_id}, {split_metric_ood}) ({faith_type})")
                        # axs[j%num_rows, i%num_cols].set_title(f"")

        # plt.suptitle(f"{file_name} - pick accuracy: {pick_acc}")
        # plt.savefig("GOOD/kernel/pipelines/plots/illustrations/automatic/lower_bound_unsup.png")
        # plt.close()
        return

if __name__ == "__main__":
    # low_discrepancy()
    # lower_bound_plaus()  # as per slide
    # lower_bound_unsup() # as per slide
    # faith_acc_gain() # as per slide
    compare_faith_mitigations()

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
r"""Kernel pipeline: main pipeline, initialization, task loading, etc.
"""
import sys
import os
import time
from typing import Tuple, Union
import json
from collections import defaultdict
from datetime import datetime

import torch.nn
from torch.utils.data import DataLoader
from torch_geometric import __version__ as pyg_v

from GOOD import config_summoner
from GOOD.data import load_dataset, create_dataloader
from GOOD.kernel.pipeline_manager import load_pipeline
from GOOD.networks.model_manager import load_model
from GOOD.ood_algorithms.ood_manager import load_ood_alg
from GOOD.utils.args import args_parser
from GOOD.utils.config_reader import CommonArgs, Munch
from GOOD.utils.initial import reset_random_seed
from GOOD.utils.logger import load_logger
from GOOD.utils.metric import assign_dict
from GOOD.definitions import OOM_CODE

import numpy as np
import matplotlib.pyplot as plt
import wandb

if pyg_v == "2.4.0":
    torch.set_num_threads(6)

def initialize_model_dataset(config: Union[CommonArgs, Munch]) -> Tuple[torch.nn.Module, Union[dict, DataLoader]]:
    r"""
    Fix random seeds and initialize a GNN and a dataset. (For project use only)

    Returns:
        A GNN and a data loader.
    """
    # Initial
    reset_random_seed(config)

    print(f'#IN#\n-----------------------------------\n    Task: {config.task}\n'
          f'{time.asctime(time.localtime(time.time()))}')
    # Load dataset
    print(f'#IN#Load Dataset {config.dataset.dataset_name}')
    dataset = load_dataset(config.dataset.dataset_name, config)
    print(f"#D#Dataset: {dataset}")
    print('#D#', dataset['train'][0] if type(dataset) is dict else dataset[0])
    print(dataset["id_val"].get(0))

    # for split in ["train", "id_val", "val", "test"]:
    #     print(f"({split}) {dataset[split].y.unique(return_counts=True)}")
    #     print(f"({split}) {np.mean([d.edge_index.shape[1] for d in dataset[split]]):.3f} +- {np.min([d.edge_index.shape[1] for d in dataset[split]]):.3f}")
    # print(dataset["test"].data)
    # print(dataset["test"][0].edge_index)
    # print(dataset["test"][0].node_perm)
    # dict_perm = {j: p.item() for j, p in enumerate(dataset["test"][0].node_perm)}
    # tmp = torch.tensor([ [dict_perm[x.item()], dict_perm[y.item()]] for x,y in dataset["test"][0].edge_index.T ]).T
    # print(tmp)
    # print(dataset["test"][0].ori_edge_index)

    # for split in ["id_val", "test"]:
    #     if hasattr(dataset[split], "edge_attr"):
    #         edge_attrs = {(u.item(), v.item()): dataset[split].edge_attr[i] for i, (u,v) in enumerate(dataset[split].edge_index.T)}
    #         for (u,v) in dataset[split].edge_index.T:
    #             assert torch.equal(edge_attrs[u.item(), v.item()], edge_attrs[v.item(), u.item()]), "Edge attributes not equal"

    loader = create_dataloader(dataset, config)

    # Load model
    print('#IN#Loading model...')
    model = load_model(config.model.model_name, config)

    return model, loader

def stability_detector_rebuttal(args):
    assert len(args.metrics.split("/")) == 1, args.metrics.split("/")
    
    load_splits = ["id"]
    if args.splits != "":
        splits = args.splits.split("/")
    else:
        splits = ["id_val"]
    if args.ratios != "":
        ratios = [float(r) for r in args.ratios.split("/")]
    else:
        ratios = [.3, .6, .9, 1.]
    # ratios = [0.3]
    print("Using ratios = ", ratios)

    results = {l: {k: defaultdict(list) for k in splits} for l in load_splits}
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)

        for i, seed in enumerate(args.seeds.split("/")):
            print(f"COMPUTING STABILITY DETECTOR FOR LOAD SPLIT = {load_split} AND SEED {seed}\n\n")
            seed = int(seed)        
            args.random_seed = seed
            args.exp_round = seed
            
            config = config_summoner(args)
            config["mitigation_backbone"] = args.mitigation_backbone
            config["mitigation_sampling"] = args.mitigation_sampling
            config["task"] = "test"
            config["load_split"] = load_split
            if l == 0 and i == 0:
                load_logger(config)
            
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
            pipeline.load_task(load_param=True, load_split=load_split) 

            # if "CIGA" in config.model.model_name:
            #     ratios = [pipeline.model.att_net.ratio]
            
            (edge_scores, graphs, graphs_nx, labels, avg_graph_size, \
            causal_subgraphs_r, spu_subgraphs_r, expl_accs_r, causal_masks_r)  = pipeline.compute_scores_and_graphs(
                ratios,
                splits,
                convert_to_nx=("suff" in args.metrics) and (not "suff_simple" in args.metrics)
            )
            intervention_bank = None

            for split in splits:
                score, acc_int, _ = pipeline.compute_stability_detector_rebuttal(
                    seed,
                    ratios,
                    split,
                    metric=args.metrics,
                    intervention_distrib=config.intervention_distrib,
                    intervention_bank=intervention_bank,
                    edge_scores=edge_scores[split],
                    graphs=graphs[split],
                    graphs_nx=graphs_nx[split],
                    labels=labels[split],
                    avg_graph_size=avg_graph_size[split],
                    causal_subgraphs_r=causal_subgraphs_r[split],
                    spu_subgraphs_r=spu_subgraphs_r[split],
                    expl_accs_r=expl_accs_r[split],
                    causal_masks_r=causal_masks_r[split]
                )
                results[load_split][split][args.metrics].append(score["all_L1"])
                for m in ["wiou_original", "wiou_perturbed"]:
                    for s in [""]:
                        results[load_split][split][m + s].append(score[m + s])
    
    print(f"\n\nDONE {config.dataset.dataset_name} - {config.model.model_name}")
    for load_split in results.keys():
        for split in results[load_split]:
            for metric in ["wiou_original", "wiou_perturbed"]:
                matrix = np.array(results[load_split][split][metric])
                mean_over_seeds = np.mean(matrix, axis=0)
                mean_over_ratios = np.mean(mean_over_seeds)
                std = np.std(matrix, axis=0)
                print(f"({load_split}) {split} {metric.upper()}: \t{mean_over_seeds} +- {std} (mean across ratios: {mean_over_ratios})")

def stability_detector_extended(args):
    startOverallTime = datetime.now()
    assert len(args.metrics.split("/")) == 1, args.metrics.split("/")
    
    load_splits = ["id"]
    if args.splits != "":
        splits = args.splits.split("/")
    else:
        splits = ["id_val"]

    if args.ratios != "":
        ratios = [float(r) for r in args.ratios.split("/")]
    else:
        ratios = [.3, .6, .9, 1.]
    print("Using ratios = ", ratios)

    results = {l: {k: defaultdict(list) for k in splits} for l in load_splits}
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)

        for i, seed in enumerate(args.seeds.split("/")):
            print(f"COMPUTING STABILITY DETECTOR FOR LOAD SPLIT = {load_split} AND SEED {seed}\n\n")
            seed = int(seed)        
            args.random_seed = seed
            args.exp_round = seed
            
            config = config_summoner(args)
            config["mitigation_backbone"] = args.mitigation_backbone
            config["mitigation_sampling"] = args.mitigation_sampling
            config["task"] = "test"
            config["load_split"] = load_split
            if l == 0 and i == 0:
                load_logger(config)
            
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
            pipeline.load_task(load_param=True, load_split=load_split) 
            
            (edge_scores, graphs, graphs_nx, labels, avg_graph_size, \
            causal_subgraphs_r, spu_subgraphs_r, expl_accs_r, causal_masks_r)  = pipeline.compute_scores_and_graphs(
                ratios,
                splits,
                convert_to_nx=("suff" in args.metrics) and (not "suff_simple" in args.metrics)
            )
            intervention_bank = None

            for split in splits:
                score, acc_int, _ = pipeline.compute_stability_detector_extended(
                    seed,
                    ratios,
                    split,
                    metric=args.metrics,
                    intervention_distrib=config.intervention_distrib,
                    intervention_bank=intervention_bank,
                    edge_scores=edge_scores[split],
                    graphs=graphs[split],
                    graphs_nx=graphs_nx[split],
                    labels=labels[split],
                    avg_graph_size=avg_graph_size[split],
                    causal_subgraphs_r=causal_subgraphs_r[split],
                    spu_subgraphs_r=spu_subgraphs_r[split],
                    expl_accs_r=expl_accs_r[split],
                    causal_masks_r=causal_masks_r[split]
                )
                results[load_split][split][args.metrics].append(score["all_L1"])
                for m in ["plausibility_wiou", "stability_wiou", "stability_f1", "stability_mcc"]:
                    for s in ["_original", "_perturbed"]:
                        results[load_split][split][m + s].append(score[m + s])
    
    print(f"\n\nDONE {config.dataset.dataset_name} - {config.model.model_name}")
    for load_split in results.keys():
        for split in results[load_split]:
            for m in ["plausibility_wiou", "stability_wiou", "stability_f1", "stability_mcc"]:
                for s in ["_original", "_perturbed"]:
                    metric = m + s
                    matrix = np.array(results[load_split][split][metric])
                    mean_over_seeds = np.mean(matrix, axis=0)
                    mean_over_ratios = np.mean(mean_over_seeds)
                    std = np.std(matrix, axis=0)
                    print(f"({load_split}) ({split}) {metric.upper()}: \t{mean_over_seeds} +- {std} (mean across ratios: {mean_over_ratios:.3f})")
                print()
    print("Overall time of execution: ", datetime.now() - startOverallTime)

def plot_explanation_examples(args):
    assert len(args.metrics.split("/")) == 1, args.metrics.split("/")
    
    load_splits = ["id"]
    if args.splits != "":
        splits = args.splits.split("/")
    else:
        splits = ["id_val"]
    if args.ratios != "":
        ratios = [float(r) for r in args.ratios.split("/")]
    else:
        ratios = [.3, .6, .9, 1.]
    # ratios = [0.3]
    print("Using ratios = ", ratios)

    results = {l: {k: defaultdict(list) for k in splits} for l in load_splits}
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)

        for i, seed in enumerate(args.seeds.split("/")):
            print(f"PLOTTING EXAMPLES OF EXPLANATIONS FOR LOAD SPLIT = {load_split} AND SEED {seed}\n\n")
            seed = int(seed)        
            args.random_seed = seed
            args.exp_round = seed
            
            config = config_summoner(args)
            config["mitigation_backbone"] = args.mitigation_backbone
            config["mitigation_sampling"] = args.mitigation_sampling
            config["task"] = "test"
            config["load_split"] = load_split
            if l == 0 and i == 0:
                load_logger(config)
            
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
            pipeline.load_task(load_param=True, load_split=load_split) 
            
            (edge_scores, graphs, graphs_nx, labels, avg_graph_size, \
            causal_subgraphs_r, spu_subgraphs_r, expl_accs_r, causal_masks_r)  = pipeline.compute_scores_and_graphs(
                ratios,
                splits,
                convert_to_nx=("suff" in args.metrics) and (not "suff_simple" in args.metrics)
            )
            intervention_bank = None

            for split in splits:
                pipeline.generate_explanation_examples(
                    seed,
                    ratios,
                    split,
                    metric=args.metrics,
                    intervention_distrib=config.intervention_distrib,
                    intervention_bank=intervention_bank,
                    edge_scores=edge_scores[split],
                    graphs=graphs[split],
                    graphs_nx=graphs_nx[split],
                    labels=labels[split],
                    avg_graph_size=avg_graph_size[split],
                    causal_subgraphs_r=causal_subgraphs_r[split],
                    spu_subgraphs_r=spu_subgraphs_r[split],
                    expl_accs_r=expl_accs_r[split],
                    causal_masks_r=causal_masks_r[split]
                )

def permute_attention_scores(args):
    load_splits = ["ood"]
    splits = ["id_val", "val", "test"]
    results = {l: {k: defaultdict(list) for k in splits} for l in load_splits}
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)

        for i, seed in enumerate(args.seeds.split("/")):
            print(f"GENERATING ATTN. PERMUTATION FOR LOAD SPLIT = {load_split} AND SEED {seed}\n\n")
            seed = int(seed)        
            args.random_seed = seed
            args.exp_round = seed
            
            config = config_summoner(args)
            config["mitigation_backbone"] = args.mitigation_backbone
            config["mitigation_sampling"] = args.mitigation_sampling
            config["task"] = "test"
            config["load_split"] = load_split
            if l == 0 and i == 0:
                load_logger(config)
            
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
            pipeline.load_task(load_param=True, load_split=load_split) 

            for s in splits:
                acc_ori, acc = pipeline.permute_attention_scores(s)
                results[load_split][s]["ori"].append(acc_ori)
                results[load_split][s]["perm"].append(acc)
    
    print(f"{config.dataset.dataset_name} - {config.model.model_name}")
    for load_split in results.keys():
        for split in results[load_split]:
            print(f"({load_split}) {split} {loader[split].dataset.metric} orig.: \t{np.mean(results[load_split][split]['ori']):.3f} +- {np.std(results[load_split][split]['ori']):.3f}")
            print(f"({load_split}) {split} {loader[split].dataset.metric} perm.: \t{np.mean(results[load_split][split]['perm']):.3f} +- {np.std(results[load_split][split]['perm']):.3f}")

def test_motif(args):
    load_splits = ["id"]
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)

        for i, seed in enumerate(args.seeds.split("/")):
            print(f"GENERATING PLOT FOR LOAD SPLIT = {load_split} AND SEED {seed}\n\n")
            seed = int(seed)        
            args.random_seed = seed
            args.exp_round = seed
            
            config = config_summoner(args)
            config["mitigation_backbone"] = args.mitigation_backbone
            config["mitigation_sampling"] = args.mitigation_sampling
            config["task"] = "test"
            config["load_split"] = load_split
            if l == 0 and i == 0:
                load_logger(config)
            
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
            pipeline.load_task(load_param=True, load_split=load_split) 

            pipeline.test_motif()


def generate_panel(args):
    load_splits = ["id"]
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)

        for i, seed in enumerate(args.seeds.split("/")):
            print(f"GENERATING PLOT FOR LOAD SPLIT = {load_split} AND SEED {seed}\n\n")
            seed = int(seed)        
            args.random_seed = seed
            args.exp_round = seed
            
            config = config_summoner(args)
            config["mitigation_backbone"] = args.mitigation_backbone
            config["mitigation_sampling"] = args.mitigation_sampling
            config["task"] = "test"
            config["load_split"] = load_split
            if l == 0 and i == 0:
                load_logger(config)
            
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
            pipeline.load_task(load_param=True, load_split=load_split) 

            pipeline.generate_panel()

def generate_global_explanation(args):
    load_splits = ["id"]
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)

        for i, seed in enumerate(args.seeds.split("/")):
            print(f"GENERATING GLOBAL EXPLANATION FOR LOAD SPLIT = {load_split} AND SEED {seed}\n\n")
            seed = int(seed)        
            args.random_seed = seed
            args.exp_round = seed
            
            config = config_summoner(args)
            config["mitigation_backbone"] = args.mitigation_backbone
            config["mitigation_sampling"] = args.mitigation_sampling
            config["task"] = "test"
            config["load_split"] = load_split
            if l == 0 and i == 0:
                load_logger(config)
            
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
            pipeline.load_task(load_param=True, load_split=load_split) 

            pipeline.generate_global_explanation()


def generate_plot_sampling(args):
    load_splits = ["id"]
    splits = ["test"]
    seeds = args.seeds.split("/")
    ratios = [0.3, 0.6, 0.8, 0.9, 1.0] #[0.3, 0.45, 0.6, 0.75, 0.9]   0.3, 0.6, 0.75, 0.8, 0.85, 0.9, 0.95
    sampling_alphas = [0.03, 0.05]
    all_metrics, all_accs = {}, {}
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)

        for i, seed in enumerate(seeds):
            seed = int(seed)        
            args.random_seed = seed
            args.exp_round = seed
            
            config = config_summoner(args)
            config["mitigation_backbone"] = args.mitigation_backbone
            config["mitigation_sampling"] = args.mitigation_sampling
            config["task"] = "test"
            config["load_split"] = load_split
            if l == 0 and i == 0:
                load_logger(config)
            
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
            pipeline.load_task(load_param=True, load_split=load_split)

            (edge_scores, graphs, graphs_nx, labels, avg_graph_size, \
            causal_subgraphs_r, spu_subgraphs_r, expl_accs_r, causal_masks_r)  = pipeline.compute_scores_and_graphs(
                ratios,
                splits,
                convert_to_nx="suff" in args.metrics
            )
            
            metrics, accs = pipeline.generate_plot_sampling_type(splits, ratios, sampling_alphas, graphs, graphs_nx, causal_subgraphs_r, causal_masks_r, avg_graph_size)
            all_metrics[str(seed)] = metrics
            all_accs[str(seed)] = accs

            # print(all_metrics.keys())
            # print(all_metrics["1"].keys())
            # print(all_metrics["1"]["test"].keys())
            # print(all_metrics["1"]["test"][0.3].keys())
            # print(all_metrics["1"]["test"][0.3]["RFID_0.03"])
        
        for SPLIT in splits:
            num_cols = len(sampling_alphas)
            fig, axs = plt.subplots(1, num_cols, figsize=(2.9*num_cols, 3.9), sharey=True)
            colors = {
                "NEC KL": "blue", "NEC L1":"lightblue", "FID L1 div": "green", "Model FID": "orange", "Phen. FID": "red", "Change pred": "violet"
            }
            sampling_name = {"RFID_": "RFID+ ($)", "FIXED_": "Fixed Deconfounded ($)", "DECONF_": "NEC ($)", "DECONF_R_": "NEC ($)"}
            for j, sampling_type_ori in enumerate(["RFID_", "DECONF_", "DECONF_R_"]): #"FIXED_", 
                for alpha_i, alpha in enumerate(sampling_alphas):
                    param = str(alpha_i+1 if sampling_type_ori == "FIXED_" else alpha)
                    sampling_type = sampling_type_ori + param
                    anneal, anneal_std = [], []
                    for r in ratios:
                        for i, metric_name in enumerate(["NEC L1"]):
                            anneal.append(np.mean([all_metrics[s][SPLIT][r][sampling_type][metric_name] for s in seeds]))
                            anneal_std.append(np.std([all_metrics[s][SPLIT][r][sampling_type][metric_name] for s in seeds]))
                            # axs[j%num_cols,alpha_i%num_cols].scatter(r, np.mean([all_metrics[s][SPLIT][r][sampling_type][metric_name] for s in seeds]), label=f"{metric_name}" if r == 0.3 else None, c=colors[metric_name])
                            # axs[j%num_cols,alpha_i%num_cols].scatter(r, np.mean([all_metrics[s][SPLIT][r][sampling_type][metric_name] for s in seeds]) - np.std([all_metrics[s][SPLIT][r][sampling_type][metric_name] for s in seeds]), c=colors[metric_name], alpha=.5, marker="^")
                            # axs[j%num_cols,alpha_i%num_cols].scatter(r, np.mean([all_metrics[s][SPLIT][r][sampling_type][metric_name] for s in seeds]) + np.std([all_metrics[s][SPLIT][r][sampling_type][metric_name] for s in seeds]), c=colors[metric_name], alpha=.5, marker="v")
                        # axs[j%num_cols,alpha_i%num_cols].scatter(r, np.mean([all_accs[s][SPLIT][r] for s in seeds]), label=f"Model acc" if r == 0.3 else None, c="orange", alpha=0.5)
                        # axs[j%num_cols,alpha_i%num_cols].scatter(r, np.mean([all_accs[s][SPLIT][r] for s in seeds]) - np.std([all_accs[s][SPLIT][r] for s in seeds]), c="orange", alpha=.5, marker="^")
                        # axs[j%num_cols,alpha_i%num_cols].scatter(r, np.mean([all_accs[s][SPLIT][r] for s in seeds]) + np.std([all_accs[s][SPLIT][r] for s in seeds]), c="orange", alpha=.5, marker="v")
                    
                    if "RFID" in sampling_type:
                        l = f"$\kappa=${param}"
                    elif "DECONF_R_" in sampling_type:
                        l = f"$b=${param}||R||"
                    elif "DECONF_" in sampling_type:
                        l = f"$b=${param}" + "$\\bar{m}$"

                    axs[alpha_i%num_cols].errorbar(
                        ratios,
                        anneal,
                        yerr=anneal_std,
                        fmt='-o',
                        capsize=5,
                        label=sampling_name[sampling_type_ori].replace('$', l))
                    # axs[j%num_cols,alpha_i%num_cols].plot(ratios, anneal)
                    # axs[alpha_i%num_cols].plot(ratios, [np.mean([all_accs[s][SPLIT][r] for s in seeds]) for r in ratios], c="orange", alpha=0.5)
                    axs[alpha_i%num_cols].grid(visible=True, alpha=0.5)
                    # axs[alpha_i%num_cols].set_title(f"{sampling_name[sampling_type_ori].replace('$', str(param))}")
                    # axs[alpha_i%num_cols].set_xlabel("ratio")
                    # axs[alpha_i%num_cols].set_ylabel("metric value")
                    axs[alpha_i%num_cols].set_ylim((0., 1.1))
                    axs[alpha_i%num_cols].legend(loc='best', fontsize=11)
            # plt.suptitle(f"{config.dataset.dataset_name}/{config.dataset.domain}")
            fig.supxlabel('size ratio', fontsize=13)
            fig.supylabel('value', fontsize=13)
            
            # plt.xticks(fontsize=12)
            plt.legend()
            # fig.subplots_adjust(bottom=0.3, top=0.95, left=0.1, right=0.9)
            plt.tight_layout()
            plt.savefig(f"./GOOD/kernel/pipelines/plots/metrics/R_dev_nec_sampling_{config.ood.ood_alg}_{config.dataset.dataset_name}_{config.dataset.domain}_({SPLIT}).png")
            # plt.savefig(f"./GOOD/kernel/pipelines/plots/metrics/pdfs/small_v2_dev_nec_sampling_{config.ood.ood_alg}_{config.dataset.dataset_name}_{config.dataset.domain}_({SPLIT}).pdf")
            plt.show(); 


def evaluate_metric(args):
    load_splits = ["id"]

    if args.splits != "":
        splits = args.splits.split("/")
    else:
        splits = ["id_val", "val", "test"] #"id_val", "val", "test"
    print("Using splits = ", splits)
        
    if args.ratios != "":
        ratios = [float(r) for r in args.ratios.split("/")]
    else:
        ratios = [.3, .6, .9, 1.]
    print("Using ratios = ", ratios)
    startTime = datetime.now()

    metrics_score = {}
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)
        print(f"USING LOAD SPLIT = {load_split}\n\n")
        # with open(f"storage/metric_results/{load_split}_results.json", "r") as jsonFile:
        #     results_big = json.load(jsonFile)

        metrics_score[load_split] = {s: defaultdict(list) for s in splits + ["test", "test_R"]}
        for i, seed in enumerate(args.seeds.split("/")):
            seed = int(seed)        
            args.random_seed = seed
            args.exp_round = seed
            
            config = config_summoner(args)
            config["mitigation_backbone"] = args.mitigation_backbone
            config["mitigation_sampling"] = args.mitigation_sampling
            config["task"] = "test"
            config["load_split"] = load_split
            expname = f"{load_split}_{config.util_model_dirname}_{config.dataset.dataset_name}{config.dataset.domain}" \
                f"_budgetsamples{config.numsamples_budget}_expbudget{config.expval_budget}" \
                f"_samplingtype{config.samplingtype}_necnumbersamples{config.nec_number_samples}"\
                f"_nec_alpha_1{config.nec_alpha_1}_fidelity_alpha_2{config.fidelity_alpha_2}"
            if l == 0 and i == 0:
                load_logger(config)
            
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
            pipeline.load_task(load_param=True, load_split=load_split)
            if "CIGA" in config.model.model_name:
                ratios = [pipeline.model.att_net.ratio, 1.0]

            if not (len(args.metrics.split("/")) == 1 and args.metrics.split("/")[0] == "acc"):
                (edge_scores, graphs, graphs_nx, labels, avg_graph_size, \
                causal_subgraphs_r, spu_subgraphs_r, expl_accs_r, causal_masks_r)  = pipeline.compute_scores_and_graphs(
                    ratios,
                    splits,
                    convert_to_nx=("suff" in args.metrics) and (not "suff_simple" in args.metrics)
                )
                intervention_bank = None
                # if "suff" in args.metrics:
                #     intervention_bank = pipeline.compute_intervention_bank(ratios, splits=["id_val", "val", "test"], graphs_nx=graphs_nx, causal_subgraphs_r=causal_subgraphs_r)

            for metric in args.metrics.split("/"):
                print(f"\n\nEvaluating {metric.upper()} for seed {seed} with load_split {load_split}\n")

                if metric == "acc":
                    assert not (config.acc_givenR and config.mask)
                    if not config.acc_givenR:
                        for split in splits + (["test"] if not "test" in splits else []):
                            pipeline.compute_accuracy_binarizing(split, givenR=False, metric_collector=metrics_score[load_split][split])
                    print("\n\nComputing now with givenR...\n")
                    pipeline.compute_accuracy_binarizing("test", givenR=True, metric_collector=metrics_score[load_split]["test_R"])
                    continue
                elif metric == "plaus":
                    for split in splits:
                        metrics_score[load_split][split]["wiou"].append([np.mean([e[0] for e in expl_accs_r[split][r]]) for r in ratios])
                        metrics_score[load_split][split]["wiou_std"].append([np.std([e[0] for e in expl_accs_r[split][r]]) for r in ratios])
                        metrics_score[load_split][split]["F1"].append([np.mean([e[1] for e in expl_accs_r[split][r]]) for r in ratios])
                        metrics_score[load_split][split]["F1_std"].append([np.std([e[1] for e in expl_accs_r[split][r]]) for r in ratios])
                    continue

                for split in splits:
                    score, acc_int, results = pipeline.compute_metric_ratio(
                        ratios,
                        split,
                        metric=metric,
                        intervention_distrib=config.intervention_distrib,
                        intervention_bank=intervention_bank,
                        edge_scores=edge_scores[split],
                        graphs=graphs[split],
                        graphs_nx=graphs_nx[split],
                        labels=labels[split],
                        avg_graph_size=avg_graph_size[split],
                        causal_subgraphs_r=causal_subgraphs_r[split],
                        spu_subgraphs_r=spu_subgraphs_r[split],
                        expl_accs_r=expl_accs_r[split],
                        causal_masks_r=causal_masks_r[split]
                    )
                    # assign_dict(
                    #     results_big,
                    #     [expname, split, metric, f"seed_{seed}"],
                    #     score
                    # )
                    metrics_score[load_split][split][metric].append(score)
                    metrics_score[load_split][split][metric + "_acc_int"].append(acc_int)
        
        # if not "suff" in args.metrics and not "acc" in args.metrics:
        #     print("\n\n")
        #     for split in splits:
        #         avg_score = {}
        #         for metric_key in results_big[expname][split][metric][f"seed_{args.seeds[0]}"].keys():
        #             sa = []
        #             for seed in results_big[expname][split][metric].keys():
        #                 sa.append(results_big[expname][split][metric][seed][metric_key])
        #             avg_score[metric_key] = np.mean(sa, axis=0)
        #         print(f"Manually averaged results ({split}): ", avg_score)
        #         print("\n\n")
        #         assign_dict(
        #             results_big,
        #             [expname, split, metric, "seed_avg"],
        #             avg_score
        #         )
        #         print(results_big[expname][split]["nec"])
        #         print("\n\n")

        # if metric.lower() in ("suff", "suff++" "nec", "nec++", "fidp", "fidm") and config.save_metrics:
        #     with open(f"storage/metric_results/{load_split}_results.json", "w") as f:
        #         json.dump(results_big, f)        

    if config.save_metrics:
        if not os.path.exists(f"storage/metric_results/aggregated_{load_split}_results_{config.log_id}.json"):
            with open(f"storage/metric_results/aggregated_{load_split}_results_{config.log_id}.json", 'w') as file:
                file.write("{}")
        with open(f"storage/metric_results/aggregated_{load_split}_results_{config.log_id}.json", "r") as jsonFile:
            results_aggregated = json.load(jsonFile)
    else:
        results_aggregated = None

    for load_split in load_splits:
        print("\n\n", "-"*50, f"\nPrinting evaluation results for load_split {load_split}\n\n")
        for split in splits:
            print(f"\nEval split {split}")
            for metric in args.metrics.split("/"):
                print(f"{metric} = {metrics_score[load_split][split][metric]}")
        if "acc" in args.metrics.split("/"):
            for split in splits + ["test", "test_R"]:
                print(f"\nEval split {split}")
                for metric in ["acc"]: #, "plaus", "wiou"
                    print(f"{metric} = {metrics_score[load_split][split][metric]}")

        if "plaus" in args.metrics:
            print("\n\n", "-"*50, "\nComputing Plausibility")
            for split in splits:
                print(f"\nEval split {split}")
                for div in ["wiou", "F1"]:
                    s = metrics_score[load_split][split][div]
                    print_metric(div, s, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, div])
            continue

        print("\n\n", "-"*50, "\nPrinting evaluation averaged per seed")
        for split in splits:
            print(f"\nEval split {split}")
            for metric in args.metrics.split("/"):
                if "acc" == metric:
                    continue
                # for c in metrics_score[load_split][split][metric][0].keys():
                #     s = [
                #         metrics_score[load_split][split][metric][i][c] for i in range(len(metrics_score[load_split][split][metric]))
                #     ]
                #     print_metric(metric + f" class {c}", s, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, metric])
                for div in ["L1", "KL"]:
                    s = [
                        metrics_score[load_split][split][metric][i][f"all_{div}"] for i in range(len(metrics_score[load_split][split][metric]))
                    ]
                    print_metric(metric + f" class all_{div}", s, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, metric+f"_{div}"])
                print_metric(metric + "_acc_int", metrics_score[load_split][split][metric + "_acc_int"], results_aggregated, key=[config.dataset.dataset_name+" "+config.dataset.domain, config.model.model_name, split, metric+"_acc_int"])
                
        if "acc" in args.metrics.split("/"):
            for split in splits + ["test", "test_R"]:
                print(f"\nEval split {split}")
                print_metric("acc", metrics_score[load_split][split]["acc"])
                for a in ["plaus", "wiou"]:
                    for c in metrics_score[load_split][split][a][0].keys():
                        s = [
                            metrics_score[load_split][split][a][i][c] for i in range(len(metrics_score[load_split][split][a]))
                        ]
                        print_metric(a + f" class {c}", s)

        print("\n\n", "-"*50, "\nComputing faithfulness")
        for split in splits:
            print(f"\nEval split {split}")            
            for div in ["L1", "KL"]:
                if "suff" in args.metrics.split("/") and "nec" in args.metrics.split("/"):                
                    suff = get_tensorized_metric(metrics_score[load_split][split]["suff"], f"all_{div}")
                    nec = get_tensorized_metric(metrics_score[load_split][split]["nec"], f"all_{div}")[:, :suff.shape[1]]             
                    faith_aritm = aritm(suff, nec)
                    faith_armonic = armonic(suff, nec)
                    faith_gmean = gmean(suff, nec)
                    print_metric(f"Faith. Aritm ({div})= \t\t", faith_aritm, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, f"faith_aritm_{div}"])
                    print_metric(f"Faith. Armon ({div})= \t\t", faith_armonic, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, f"faith_armon_{div}"])
                    print_metric(f"Faith. GMean ({div})= \t", faith_gmean, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, f"faith_gmean_{div}"])

                if "suff++" in args.metrics.split("/") and "nec" in args.metrics.split("/"):
                    suff = get_tensorized_metric(metrics_score[load_split][split]["suff++"], f"all_{div}")
                    nec = get_tensorized_metric(metrics_score[load_split][split]["nec"], f"all_{div}")[:, :suff.shape[1]]             
                    faith_aritm = aritm(suff, nec)
                    faith_armonic = armonic(suff, nec)
                    faith_gmean = gmean(suff, nec)
                    print_metric(f"Faith. Aritm ({div})= \t\t", faith_aritm, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, f"faith_aritm_{div}"])
                    print_metric(f"Faith. Armon ({div})= \t\t", faith_armonic, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, f"faith_armon_{div}"])
                    print_metric(f"Faith. GMean ({div})= \t", faith_gmean, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, f"faith_gmean_{div}"])
                
                if "suff_simple" in args.metrics.split("/") and "nec" in args.metrics.split("/"):
                    suff = get_tensorized_metric(metrics_score[load_split][split]["suff_simple"], f"all_{div}")
                    nec = get_tensorized_metric(metrics_score[load_split][split]["nec"], f"all_{div}")[:, :suff.shape[1]]             
                    faith_aritm = aritm(suff, nec)
                    faith_armonic = armonic(suff, nec)
                    faith_gmean = gmean(suff, nec)
                    print_metric(f"Faith. Aritm ({div})= \t\t", faith_aritm, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, f"faith_aritm_{div}"])
                    print_metric(f"Faith. Armon ({div})= \t\t", faith_armonic, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, f"faith_armon_{div}"])
                    print_metric(f"Faith. GMean ({div})= \t", faith_gmean, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, f"faith_gmean_{div}"])

                if "suff" in args.metrics.split("/") and "nec++" in args.metrics.split("/"):
                    suff = get_tensorized_metric(metrics_score[load_split][split]["suff"], f"all_{div}")
                    necpp = get_tensorized_metric(metrics_score[load_split][split]["nec++"], f"all_{div}")[:, :suff.shape[1]]
                    faith_aritm = aritm(suff, necpp)
                    faith_armonic = armonic(suff, necpp)
                    faith_gmean = gmean(suff, necpp)
                    print_metric(f"Faith.++ Aritm ({div})= \t\t", faith_aritm, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, f"faith++_aritm_{div}"])
                    print_metric(f"Faith.++ Armon ({div})= \t\t", faith_armonic, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, f"faith++_gmean_{div}"])
                    print_metric(f"Faith.++ GMean ({div})= \t", faith_gmean, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, split, f"faith++_gmean_{div}"])

                if "fidp" in args.metrics.split("/") and "fidm" in args.metrics.split("/"):
                    assert False
                    fidm = torch.tensor(metrics_score[load_split][split]["fidm"])
                    fidp  = torch.tensor(metrics_score[load_split][split]["fidp"])
                    faith_armonic = armonic(fidm, fidp)
                    faith_gmean = gmean(fidm, fidp)
                    print_metric("Char. Score = \t\t", faith_armonic)
                    print_metric("Char. Score GMean = \t", faith_gmean)
        print(f"Computed for split load_split = {load_split}\n\n\n")
    
    if config.save_metrics:
        exit("NO!") #TODO: remove me
        with open(f"storage/metric_results/aggregated_{load_split}_results_{config.log_id}.json", "w") as f:
            json.dump(results_aggregated, f)     
    
    print("Completed in ", datetime.now() - startTime, f" for {config.model.model_name} {config.dataset.dataset_name}/{config.dataset.domain}")
    print("\n\n")
    sys.stdout.flush()
                    
def gmean(a,b):
    return (a*b).sqrt()
def aritm(a,b):
    return (a+b) / 2
def armonic(a,b):
    return 2 * (a*b) / (a+b)
def gstd(a):
    return a.log().std().exp()
def print_metric(name, data, results_aggregated=None, key=None):
    avg = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    print(name, " = ", ", ".join([f"{avg[i]:.3f} +- {std[i]:.3f}" for i in range(len(avg))]))
    if not results_aggregated is None:
        assign_dict(
            results_aggregated,
            key,
            avg.tolist()
        )
        key[-1] += "_std" # add _std to the metric name
        assign_dict(
            results_aggregated,
            key,
            std.tolist()
        )
def get_tensorized_metric(scores, c):
    return torch.tensor([
        scores[i][c] for i in range(len(scores))
    ])


def main():
    args = args_parser()

    assert not args.seeds is None, args.seeds
    # assert args.metrics != ""

    if args.task == 'eval_metric':
        evaluate_metric(args)
        exit(0)
    if args.task == 'plot_panel':
        generate_panel(args)
        exit(0)
    if args.task == 'plot_global':
        generate_global_explanation(args)
        exit(0)
    if args.task == 'test_motif':
        test_motif(args)
        exit(0)
    if args.task == 'permute_attention':
        permute_attention_scores(args)
        exit(0)
    if args.task == 'plot_sampling':
        generate_plot_sampling(args)
        exit(0)
    if args.task == 'stability_detector':
        # stability_detector_rebuttal(args)
        stability_detector_extended(args)
        exit(0)
    if args.task == 'plot_explanations':
        plot_explanation_examples(args)
        exit(0)
        

    run = None
    test_scores, test_losses = defaultdict(list), defaultdict(list)
    test_likelihoods_avg, test_likelihoods_prod, test_likelihoods_logprod, test_wious = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for i, seed in enumerate(args.seeds.split("/")):
        seed = int(seed)
        print(f"\n\n#D#Running with seed = {seed}")
        
        args.random_seed = seed
        args.exp_round = seed
        
        config = config_summoner(args)
        config["mitigation_backbone"] = args.mitigation_backbone
        config["mitigation_sampling"] = args.mitigation_sampling
        print(config.random_seed, config.exp_round)
        print(args)
        if i == 0:
            load_logger(config)
        
        model, loader = initialize_model_dataset(config)
        ood_algorithm = load_ood_alg(config.ood.ood_alg, config)

        pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)

        if config.task == 'train':
            startTrainTime = datetime.now()
            if config.wandb:
                run = wandb.init(
                    project="global-local-modular-gnn",
                    config=config,
                    entity="mcstewe",
                    name=f'{config.dataset.dataset_name}_{config.dataset.domain}{config.ood_dirname}_{config.util_model_dirname}_{config.random_seed}'
                )
                wandb.watch(pipeline.model, log="all", log_freq=10)

            # Train model
            pipeline.load_task()            
            print(f'\nTraining end ({datetime.now() - startTrainTime}).\n')

            # Eval model
            pipeline.task = 'test'
            test_score, test_loss = pipeline.load_task(load_param=True, load_split="id")
            test_scores["saved_score"].append(test_score)
            for s in ["id_val", "id_test", "val", "test"]:
                sa = pipeline.evaluate(s, compute_suff=False)
                test_scores[s].append(sa['score'])
        elif config.task == 'test':
            test_score, test_loss = pipeline.load_task(load_param=True, load_split="id")
            
            # Set manual weights for DEBUG
            # model.global_side_channel.classifier.classifier[0].weight = torch.nn.Parameter(
            #     torch.tensor([[1.0, 0., 0.]], device=config.device)
            # )
            # model.global_side_channel.classifier.classifier[0].bias = torch.nn.Parameter(
            #     torch.tensor([[-4.9]], device=config.device)
            # )
            # print(model.global_side_channel.classifier.classifier[0].weight)
            # model.global_side_channel.classifier.classifier[0].reset_parameters()


            for s in ["train", "id_val", "id_test", "val", "test"]:
                sa = pipeline.evaluate(
                    s,
                    compute_suff=False, 
                    compute_wiou=config.global_side_channel == "simple_concept" and config.model.model_name != "GIN"
                )
                test_scores[s].append(sa['score'])
                test_losses[s].append(sa['loss'].item())
                test_wious[s].append(sa['wiou'].item())
                test_likelihoods_avg[s].append(sa['likelihood_avg'].item())
                test_likelihoods_prod[s].append(sa['likelihood_prod'].item())
                test_likelihoods_logprod[s].append(sa['likelihood_logprod'].item())
            # test_score, test_loss = pipeline.load_task(load_param=True, load_split="ood")
            # for s in ["train", "id_val", "val", "test"]:
            #     sa = pipeline.evaluate(s, compute_suff=False)
            #     test_scores["ood_" + s].append(sa['score'])
            #     test_losses["ood_" + s].append(sa['loss'].item())
            # print(f"Printing obtained and stored scores: {sa['score']} !=? {test_score}")
    
    if config.save_metrics:
        with open(f"storage/metric_results/acc_plaus.json", "r") as jsonFile:
            results_aggregated = json.load(jsonFile)
    
    print("\n\nFinal accuracies: ")
    for s in test_scores.keys():
        print(f"{s.upper():<10} = {np.mean(test_scores[s]):.3f} +- {np.std(test_scores[s]):.3f}")

    if config.global_side_channel == "simple_concept" and config.model.model_name != "GIN":
        print("\n\nFinal WIoUs: ")
        for s in test_wious.keys():
            print(f"{s.upper():<10} = {np.mean(test_wious[s]):.3f} +- {np.std(test_wious[s]):.3f}")

    print("\nFinal losses: ")
    for s in test_losses.keys():
        print(f"{s.upper():<10} = {np.mean(test_losses[s]):.4f} +- {np.std(test_losses[s]):.4f}")
            
    for s in [""]: #"ood_"
        print(f"Diff id_val-test {s} = {abs(np.mean(test_losses[s + 'id_val']) - np.mean(test_losses[s + 'test'])):.4f} ")

    if config.save_metrics:
        print("Saving metrics to json...")
        for s in test_losses.keys():
            for name, d in zip(
                ["loss_entiresplit", "likelihood_avg_entiresplit", "likelihood_prod_entiresplit", "likelihood_logprod_entiresplit"], 
                [test_losses, test_likelihoods_avg, test_likelihoods_prod, test_likelihoods_logprod]
            ):
                key = [config.dataset.dataset_name + " " + config.dataset.domain, config.model.model_name, s, name]
                if s in results_aggregated[key[0]][key[1]].keys():            
                    assign_dict(
                        results_aggregated,
                        key,
                        np.mean(d[s])
                    )
                    key[-1] += "_std"
                    assign_dict(
                        results_aggregated,
                        key,
                        np.std(d[s])
                    )
        with open(f"storage/metric_results/acc_plaus.json", "w") as f:
            json.dump(results_aggregated, f)  

    if config.global_side_channel in ("simple", "simple_filternode", "simple_concept"):
        with torch.no_grad():
            # Print weights of global channel
            if config.global_side_channel in ("simple", "simple_filternode"):
                w = model.global_side_channel.classifier.classifier[0].weight.detach().cpu().numpy()
                b = model.global_side_channel.classifier.classifier[0].bias.detach().cpu().numpy()
                print(f"\nWeight vector of global side channel:\nW: {w}\nb:{b}")
                print(f"\nBeta combination parameter of global side channel:{model.beta.sigmoid().item():.4f}\n")   
            elif config.global_side_channel == "simple_concept":
                print("\nConcept relevance scores:\n", model.combinator.classifier[0].alpha_norm.cpu().numpy(), "\n")

            if config.global_side_channel == "simple_filternode":
                # Print attention filter score for each unique node feature
                feats = loader["test"].dataset.x.unique(dim=0).to(config.device)
                node_feat_attn = model.global_side_channel.node_filter(feats)
                print("Node filtering scores for unique test node features:\n", torch.cat((feats, node_feat_attn), dim=1))

    if config.wandb and run:
        run.finish()

def goodtg():
    try:
        main()
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f'#E#{e}')
            exit(OOM_CODE)
        else:
            raise e


if __name__ == '__main__':
    main()

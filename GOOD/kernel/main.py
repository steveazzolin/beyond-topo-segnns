r"""Kernel pipeline: main pipeline, initialization, task loading, etc.
"""
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
from GOOD.definitions import OOM_CODE

import numpy as np

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

    for split in ["train", "id_val", "val", "test"]:
        print(f"({split}) {dataset[split].y.unique(return_counts=True)}")
        print(f"({split}) {np.mean([d.edge_index.shape[1] for d in dataset[split]]):.3f} +- {np.min([d.edge_index.shape[1] for d in dataset[split]]):.3f}")
    # print(dataset["test"].data)
    # print(dataset["test"][0].edge_index)
    # print(dataset["test"][0].node_perm)
    # dict_perm = {j: p.item() for j, p in enumerate(dataset["test"][0].node_perm)}
    # tmp = torch.tensor([ [dict_perm[x.item()], dict_perm[y.item()]] for x,y in dataset["test"][0].edge_index.T ]).T
    # print(tmp)
    # print(dataset["test"][0].ori_edge_index)

    loader = create_dataloader(dataset, config)

    # Load model
    print('#IN#Loading model...')
    model = load_model(config.model.model_name, config)

    return model, loader


def permute_attention_scores(args):
    load_splits = ["id"]
    splits = ["id_val", "val", "test"] #, "val", "test"
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
            print(f"({load_split}) {split} Acc original: \t{np.mean(results[load_split][split]['ori']):.3f} +- {np.std(results[load_split][split]['ori']):.3f}")
            print(f"({load_split}) {split} Acc permutation: \t{np.mean(results[load_split][split]['perm']):.3f} +- {np.std(results[load_split][split]['perm']):.3f}")

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

def evaluate_metric(args):
    load_splits = ["id"]
    splits = ["id_val", "val", "test"]
    startTime = datetime.now()

    metrics_score = {}
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)
        print(f"USING LOAD SPLIT = {load_split}\n\n")

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
            config["device"] = "cuda:0"
            if l == 0 and i == 0:
                load_logger(config)
            
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
            pipeline.load_task(load_param=True, load_split=load_split)

            edge_scores, graphs = {s: None for s in splits}, {s: None for s in splits}
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

                for split in splits:
                    score, acc_int, results, edge_scores_split, graphs_split = pipeline.compute_metric_ratio(
                        split,
                        metric=metric,
                        intervention_distrib=config.intervention_distrib,
                        edge_scores=edge_scores[split],
                        graphs=graphs[split]
                    )
                    edge_scores[split] = edge_scores_split
                    graphs[split] = graphs_split
                    metrics_score[load_split][split][metric].append(score)
                    metrics_score[load_split][split][metric + "_acc_int"].append(acc_int)

                    if metric.lower() in ("suff", "nec", "nec++", "fidp", "fidm") and config.save_metrics:
                        expname = f"{config.load_split}_{config.util_model_dirname}_{config.dataset.dataset_name}_{config.random_seed}_{metric}_{split}_budgetsamples{config.numsamples_budget}_expbudget{config.expval_budget}"
                        with open(f"storage/metric_results/{expname}.json", "w") as f:
                            json.dump(results, f)

    for load_split in load_splits:
        print("\n\n", "-"*50, f"\nPrinting evaluation results for load_split {load_split}\n\n")
        for split in splits:
            print(f"\nEval split {split}")
            for metric in args.metrics.split("/"):
                print(f"{metric} = {metrics_score[load_split][split][metric]}")
        if "acc" in args.metrics.split("/"):
            for split in splits + ["test", "test_R"]:
                print(f"\nEval split {split}")
                for metric in ["acc", "plaus", "wiou"]:
                    print(f"{metric} = {metrics_score[load_split][split][metric]}")

        print("\n\n", "-"*50, "\nPrinting evaluation averaged per seed")
        for split in splits:
            print(f"\nEval split {split}")
            for metric in args.metrics.split("/"):
                if "acc" == metric:
                    continue
                for c in metrics_score[load_split][split][metric][0].keys():
                    s = [
                        metrics_score[load_split][split][metric][i][c] for i in range(len(metrics_score[load_split][split][metric]))
                    ]
                    print_metric(metric + f" class {c}", s)
                # print_metric(metric, metrics_score[load_split][split][metric])
                print_metric(metric + "_acc_int", metrics_score[load_split][split][metric + "_acc_int"])
                
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
            if "suff" in args.metrics.split("/") and "nec" in args.metrics.split("/"):
                suff = torch.tensor(metrics_score[load_split][split]["suff"]) #.mean(0) #1xratio
                nec  = torch.tensor(metrics_score[load_split][split]["nec"])[:, :suff.shape[1]]                
                faith_armonic = armonic(suff, nec)
                faith_gmean = gmean(suff, nec)
                print_metric("Faith. = \t", faith_armonic)
                print_metric("Faith. GMean = \t", faith_gmean)

            if "suff" in args.metrics.split("/") and "nec++" in args.metrics.split("/"):
                suff = get_tensorized_metric(metrics_score[load_split][split]["suff"], "all")
                necpp = get_tensorized_metric(metrics_score[load_split][split]["nec++"], "all")[:, :suff.shape[1]]
                faith_armonic = armonic(suff, necpp)
                faith_gmean = gmean(suff, necpp)
                print_metric("Faith.++ = \t", faith_armonic)
                print_metric("Faith.++ GMean = \t", faith_gmean)

            if "fidp" in args.metrics.split("/") and "fidm" in args.metrics.split("/"):
                fidm = torch.tensor(metrics_score[load_split][split]["fidm"])
                fidp  = torch.tensor(metrics_score[load_split][split]["fidp"])
                faith_armonic = armonic(fidm, fidp)
                faith_gmean = gmean(fidm, fidp)
                print_metric("Char. Score = \t", faith_armonic)
                print_metric("Char. Score GMean = \t", faith_gmean)
        print(f"Computed for split load_split = {load_split}\n\n\n")
    print("Completed in ", datetime.now() - startTime, f" for {config.model.model_name} {config.dataset.dataset_name}")
    print("\n\n")
                    
def gmean(a,b):
    return (a*b).sqrt()
def armonic(a,b):
    return 2 * (a*b) / (a+b)
def gstd(a):
    return a.log().std().exp()
def print_metric(name, data):
    avg = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    print(name, " = ", ", ".join([f"{avg[i]:.3f} +- {std[i]:.3f}" for i in range(len(avg))]))
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
    if args.task == 'test_motif':
        test_motif(args)
        exit(0)
    if args.task == 'permute_attention':
        permute_attention_scores(args)
        exit(0)

    test_scores = defaultdict(list)
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
            pipeline.load_task() # train model
            pipeline.task = 'test'
            test_score, test_loss = pipeline.load_task()
            test_scores["trained"].append(test_score)
        elif config.task == 'test':
            test_score, test_loss = pipeline.load_task(load_param=True, load_split="id")
            for s in ["train", "id_val", "val", "test"]:
                sa = pipeline.evaluate(s, compute_suff=False)
                test_scores[s].append(sa['score'])
            test_score, test_loss = pipeline.load_task(load_param=True, load_split="ood")
            for s in ["train", "id_val", "val", "test"]:
                sa = pipeline.evaluate(s, compute_suff=False)
                test_scores["ood_" + s].append(sa['score'])
            print(f"Printing obtained and stored scores: {sa['score']} !=? {test_score}")
    print()
    print()
    print("Final scores: ")
    for s in test_scores.keys():
        print(f"{s.upper()} = {round(np.mean(test_scores[s]), 3)} +- {round(np.std(test_scores[s]), 3)}")
    print()


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

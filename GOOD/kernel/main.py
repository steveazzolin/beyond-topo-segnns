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
    print(dataset["id_val"].data)

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


def evaluate_metric(args):
    load_splits = ["id"]
    splits = ["id_val", "val"]
    startTime = datetime.now()
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)
        print(f"USING LOAD SPLIT = {load_split}\n\n")

        metrics_score = {s: defaultdict(list) for s in splits + ["test", "test_R"]}
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

            test_score, test_loss = pipeline.load_task(load_param=True, load_split=load_split)            
            # for e in ["train", "id_val", "val", "test"]:
            #     sa = pipeline.evaluate(e, compute_suff=False)
            #     print(e, sa)
            # exit()

            edge_scores, graphs = {s: None for s in splits}, {s: None for s in splits}
            for metric in args.metrics.split("/"):
                print(f"\n\nEvaluating {metric.upper()} for seed {seed} with load_split {load_split}\n")

                if metric == "acc":
                    assert not (config.acc_givenR and config.mask)
                    if not config.acc_givenR:
                        for split in splits + ["test"]:
                            acc, xai = pipeline.compute_accuracy_binarizing(split, givenR=False)
                            metrics_score[split]["acc"].append(acc)
                            metrics_score[split]["plaus"].append([e[1] for e in xai])
                            metrics_score[split]["wiou"].append([e[0] for e in xai])
                    print("\n\nComputing now with givenR...\n")
                    acc, xai = pipeline.compute_accuracy_binarizing("test", givenR=True)
                    metrics_score["test_R"]["acc"].append(acc)
                    metrics_score["test_R"]["plaus"].append([e[1] for e in xai])
                    metrics_score["test_R"]["wiou"].append([e[0] for e in xai])
                    continue

                for split in splits:
                    score, results, edge_scores_split, graphs_split = pipeline.compute_metric_ratio(
                        split,
                        metric=metric,
                        intervention_distrib=config.intervention_distrib,
                        edge_scores=edge_scores[split],
                        graphs=graphs[split]
                    )
                    edge_scores[split] = edge_scores_split
                    graphs[split] = graphs_split
                    metrics_score[split][metric].append(score)

                    if metric.lower() in ("suff", "nec", "nec++", "fidp", "fidm") and config.save_metrics:
                        expname = f"{config.load_split}_{config.util_model_dirname}_{config.dataset.dataset_name}_{config.random_seed}_{metric}_{split}_budgetsamples{config.numsamples_budget}_expbudget{config.expval_budget}"
                        with open(f"storage/metric_results/{expname}.json", "w") as f:
                            json.dump(results, f)

        print("\n\n", "-"*50, "\nPrinting evaluation results")
        for split in splits:
            print(f"\nEval split {split}")
            for metric in args.metrics.split("/"):
                print(f"{metric} = {metrics_score[split][metric]}")
                if metric == "acc":
                    for a in ["plaus", "wiou"]:
                        print(f"{a} = {metrics_score[split][a]}")
        if "acc" in args.metrics.split("/"):
            for split in ["test", "test_R"]:
                print(f"\nEval split {split}")
                for metric in ["acc", "plaus", "wiou"]:
                    print(f"{metric} = {metrics_score[split][metric]}")
        print(f"Computed for split load_split = {load_split}\n\n\n")

        print("\n\n", "-"*50, "\nPrinting evaluation averaged per seed")
        for split in splits:
            print(f"\nEval split {split}")
            for metric in args.metrics.split("/"):
                avg_per_seed = torch.tensor(metrics_score[split][metric]).mean(0)
                std_per_seed = torch.tensor(metrics_score[split][metric]).std(0)
                print(f"{metric} = {avg_per_seed} +- {std_per_seed}")
                if metric == "acc":
                    for a in ["plaus", "wiou"]:
                        avg_per_seed = torch.tensor(metrics_score[split][a]).mean(0)
                        std_per_seed = torch.tensor(metrics_score[split][a]).std(0)
                        print(f"{a} = {avg_per_seed} +- {std_per_seed}")
        if "acc" in args.metrics.split("/"):
            for split in ["test", "test_R"]:
                print(f"\nEval split {split}")
                for a in ["acc", "plaus", "wiou"]:
                    avg_per_seed = torch.tensor(metrics_score[split][a]).mean(0)
                    std_per_seed = torch.tensor(metrics_score[split][a]).std(0)
                    print(f"{a} = {avg_per_seed} +- {std_per_seed}")

        print("\n\n", "-"*50, "\nComputing faithfulness")
        for split in splits:
            print(f"\nEval split {split}")
            
            if "suff" in args.metrics.split("/") and "nec" in args.metrics.split("/"):
                suff = torch.tensor(metrics_score[split]["suff"]) #.mean(0) #1xratio
                nec  = torch.tensor(metrics_score[split]["nec"])[:, :suff.shape[1]] #.mean(0) #1xratio
                
                faith_armonic = armonic(suff, nec)
                faith_gmean = gmean(suff, nec)

                print(f"Faith = \t{torch.round(faith_armonic.mean(0), decimals=3)} +- {torch.round(faith_armonic.std(0), decimals=3)}")
                print(f"Faith GMean = \t{torch.round(faith_gmean.mean(0), decimals=3)} +- {torch.round(faith_gmean.std(0), decimals=3)}")

                if "nec++" in args.metrics.split("/"):
                    necpp = torch.tensor(metrics_score[split]["nec++"])[:, :suff.shape[1]] #.mean(0) #1xratio
                    faith_armonic = armonic(suff, necpp)
                    faith_gmean = gmean(suff, necpp)
                    print(f"Faith = \t{torch.round(faith_armonic.mean(0), decimals=3)} +- {torch.round(faith_armonic.std(0), decimals=3)}")
                    print(f"Faith GMean = \t{torch.round(faith_gmean.mean(0), decimals=3)} +- {torch.round(faith_gmean.std(0), decimals=3)}")

            if "fidp" in args.metrics.split("/") and "fidm" in args.metrics.split("/"):
                fidm = torch.tensor(metrics_score[split]["fidm"]) #.mean(0) #1xratio
                fidp  = torch.tensor(metrics_score[split]["fidp"]) #.mean(0) #1xratio
                
                faith_armonic = armonic(fidm, fidp)
                faith_gmean = gmean(fidm, fidp)

                print(f"Char. Score = {torch.round(faith_armonic.mean(0), decimals=3)} +- {torch.round(faith_armonic.std(0), decimals=3)}")
                print(f"Char. Score GMean = {torch.round(faith_gmean.mean(0), decimals=3)} +- {torch.round(faith_gmean.std(0), decimals=3)}")
        print("Completed in ", datetime.now() - startTime)
        print("\n\n")
                    
def gmean(a,b):
    return (a*b).sqrt()
def armonic(a,b):
    return 2 * (a*b) / (a+b)
def gstd(a):
    return a.log().std().exp()


def main():
    args = args_parser()

    assert not args.seeds is None, args.seeds
    # assert args.metrics != ""

    if args.task == 'eval_metric':
        evaluate_metric(args)
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
            test_score, test_loss = pipeline.load_task(load_param=True)
            for s in ["train", "id_val", "val", "test"]:
                sa = pipeline.evaluate(s, compute_suff=False)
                test_scores[s].append(sa['score'])
            print(f"Printing obtained and stored scores: {sa['score']} !=? {test_score}")
    print()
    print()
    print("Final scores: ")
    for s in ["train", "id_val", "val", "test"]:
        print(f"{s.upper()} = {round(np.mean(test_scores[s]), 4)} +- {round(np.std(test_scores[s]), 4)}")
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

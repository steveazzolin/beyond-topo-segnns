r"""Kernel pipeline: main pipeline, initialization, task loading, etc.
"""
import os
import time
from typing import Tuple, Union
import json

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


def evaluate_acc(args):
    load_splits = ["id"]
    for l, load_split in enumerate(load_splits):
        print("\n\n")
        print("-"*50)
        print(f"USING LOAD SPLIT = {load_split}")
        print("\n\n")

        test_scores = []
        test_acc_id, test_acc_ood = [], []
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
            sa = pipeline.evaluate("test", compute_suff=False)
            test_scores.append((sa['score'], test_score))
            print(sa)

            if not config.acc_givenR:
                acc_id, _ = pipeline.compute_accuracy_binarizing("id_val", givenR=config.acc_givenR)
                acc_id, _ = pipeline.compute_accuracy_binarizing("val", givenR=config.acc_givenR)
            acc_ood, _ = pipeline.compute_accuracy_binarizing("test", givenR=config.acc_givenR)
            # test_acc_ood.append((acc_ood, 0.))
        print()
        print()
        print("Final OOD Test scores: ", test_scores)
        print(f"Computed for split load_split = {load_split}")
        print("Final ACC_ID scores: ", test_acc_id)
        print("Final ACC_ID scores: ", test_acc_ood)
        print()


def evaluate_metric(args):
    load_splits = ["id"]
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)
        print(f"USING LOAD SPLIT = {load_split}\n\n")

        test_scores = []
        test_suff_id, test_suff_ood = [], []
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
            
            for e in ["train", "id_val", "val", "test"]:
                sa = pipeline.evaluate(e, compute_suff=False)
                print(e, sa)
            exit()

            for metric in args.metrics.split("/"):
                print(f"\n\nEvaluating {metric.upper()} for seed {seed} with load_split {load_split}\n")

                if metric == "acc":
                    assert not (config.acc_givenR and config.mask)

                    # if not config.acc_givenR:
                    #     acc_id, _ = pipeline.compute_accuracy_binarizing("id_val", givenR=config.acc_givenR)
                    #     acc_id, _ = pipeline.compute_accuracy_binarizing("val", givenR=config.acc_givenR)
                    acc_ood, _ = pipeline.compute_accuracy_binarizing("test", givenR=config.acc_givenR)
                    continue

                suff_id, suff_devstd_id, results_id, edge_scores_id, graphs_id = pipeline.compute_metric_ratio("test", metric=metric, intervention_distrib=config.intervention_distrib)
                suff_ood, suff_devstd_ood, results_ood, edge_scores_ood, graphs_ood = pipeline.compute_metric_ratio("val", metric=metric, intervention_distrib=config.intervention_distrib)

                test_suff_id.append((suff_id, suff_devstd_id))
                test_suff_ood.append((suff_ood, suff_devstd_ood))

                if metric.lower() in ("suff", "nec") and config.save_metrics:
                    expname = f"{config.load_split}_{config.util_model_dirname}_{config.random_seed}_suff_idval_budgetsamples{config.numsamples_budget}_expbudget{config.expval_budget}"
                    with open(f"storage/metric_results/{expname}.json", "w") as f:
                        json.dump(results_id, f)
                    expname = f"{config.load_split}_{config.util_model_dirname}_{config.random_seed}_suff_val_budgetsamples{config.numsamples_budget}_expbudget{config.expval_budget}"
                    with open(f"storage/metric_results/{expname}.json", "w") as f:
                        json.dump(results_ood, f)

        print("\n\nFinal OOD Test scores: ", test_scores)
        print("Final SUFF_ID scores: ", test_suff_id)
        print("Final SUFF_OOD scores: ", test_suff_ood)
        print(f"Computed for split load_split = {load_split}")


def evaluate_suff(args):
    load_splits = ["id"]
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)
        print(f"USING LOAD SPLIT = {load_split}\n\n")

        test_scores = []
        test_suff_id, test_suff_ood = [], []
        for i, seed in enumerate(args.seeds.split("/")):
            seed = int(seed)        
            args.random_seed = seed
            args.exp_round = seed
            
            config = config_summoner(args)
            config["mitigation_backbone"] = args.mitigation_backbone
            config["mitigation_sampling"] = args.mitigation_sampling
            config["task"] = "test"
            config["load_split"] = load_split
            config["device"] = "cuda"
            if l == 0 and i == 0:
                load_logger(config)
            
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)

            test_score, test_loss = pipeline.load_task(load_param=True, load_split=load_split)
            # sa = pipeline.evaluate("test", compute_suff=False)
            # test_scores.append((sa['score'], test_score))

            if "LECI" in config.model.model_name:
                suff_id, suff_devstd_id, results_id = pipeline.compute_metric_ratio("id_val", metric="suff", intervention_distrib=config.intervention_distrib)
                suff_ood, suff_devstd_ood, results_ood = pipeline.compute_metric_ratio("val", metric="suff", intervention_distrib=config.intervention_distrib)
            else:
                suff_id, suff_devstd_id, results_id = pipeline.compute_metric_ratio("id_val", metric="suff", intervention_distrib=config.intervention_distrib)
                suff_ood, suff_devstd_ood, results_ood = pipeline.compute_metric_ratio("val", metric="suff", intervention_distrib=config.intervention_distrib)

            test_suff_id.append((suff_id, suff_devstd_id))
            test_suff_ood.append((suff_ood, suff_devstd_ood))

            if config.save_metrics:
                expname = f"{config.load_split}_{config.util_model_dirname}_{config.random_seed}_suff_idval_budgetsamples{config.numsamples_budget}_expbudget{config.expval_budget}"
                with open(f"storage/metric_results/{expname}.json", "w") as f:
                    json.dump(results_id, f)
                expname = f"{config.load_split}_{config.util_model_dirname}_{config.random_seed}_suff_val_budgetsamples{config.numsamples_budget}_expbudget{config.expval_budget}"
                with open(f"storage/metric_results/{expname}.json", "w") as f:
                    json.dump(results_ood, f)

        print("\n\nFinal OOD Test scores: ", test_scores)
        print("Final SUFF_ID scores: ", test_suff_id)
        print("Final SUFF_OOD scores: ", test_suff_ood)
        print(f"Computed for split load_split = {load_split}")

def evaluate_nec(args):
    load_splits = ["id"]
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)
        print(f"USING LOAD SPLIT = {load_split}\n\n")

        test_scores = []
        test_nec_id, test_nec_ood = [], []
        for i, seed in enumerate(args.seeds.split("/")):
            seed = int(seed)        
            args.random_seed = seed
            args.exp_round = seed
            
            config = config_summoner(args)
            config["mitigation_backbone"] = args.mitigation_backbone
            config["mitigation_sampling"] = args.mitigation_sampling
            config["task"] = "test"
            config["load_split"] = load_split
            config["device"] = "cuda"
            if l == 0 and i == 0:
                load_logger(config)
            
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)

            test_score, test_loss = pipeline.load_task(load_param=True, load_split=load_split)
            # sa = pipeline.evaluate("test", compute_suff=False)
            # test_scores.append((sa['score'], test_score))

            nec_id, nec_devstd_id, results_id = pipeline.compute_metric_ratio("id_val", metric="nec")
            nec_ood, nec_devstd_ood, results_ood = pipeline.compute_metric_ratio("val", metric="nec")  

            test_nec_id.append((nec_id, nec_devstd_id))
            test_nec_ood.append((nec_ood, nec_devstd_ood))

            if config.save_metrics:
                expname = f"{config.load_split}_{config.util_model_dirname}_{config.random_seed}_nec_idval_budgetsamples{config.numsamples_budget}_expbudget{config.expval_budget}"
                with open(f"storage/metric_results/{expname}.json", "w") as f:
                    json.dump(results_id, f)
                expname = f"{config.load_split}_{config.util_model_dirname}_{config.random_seed}_nec_val_budgetsamples{config.numsamples_budget}_expbudget{config.expval_budget}_alpha{config.nec_alpha_1}"
                with open(f"storage/metric_results/{expname}.json", "w") as f:
                    json.dump(results_ood, f)

        print("\n\nFinal OOD Test scores: ", test_scores)
        print("Final SUFF_ID scores: ", test_nec_id)
        print("Final SUFF_OOD scores: ", test_nec_ood)
        print(f"Computed for split load_split = {load_split}")

def evaluate_fid(args):
    load_splits = ["id"]
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)
        print(f"USING LOAD SPLIT = {load_split}\n\n")

        test_scores = []
        test_fid_id, test_fid_ood = [], []
        for i, seed in enumerate(args.seeds.split("/")):
            seed = int(seed)        
            args.random_seed = seed
            args.exp_round = seed
            
            config = config_summoner(args)
            config["mitigation_backbone"] = args.mitigation_backbone
            config["mitigation_sampling"] = args.mitigation_sampling
            config["task"] = "test"
            config["load_split"] = load_split
            config["device"] = "cpu"
            if l == 0 and i == 0:
                load_logger(config)
            
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)

            test_score, test_loss = pipeline.load_task(load_param=True, load_split=load_split)
            sa = pipeline.evaluate("test", compute_suff=False)
            test_scores.append((sa['score'], test_score))

            if "LECI" in config.model.model_name:
                fid_id, fid_devstd_id, _ = pipeline.compute_metric_ratio("id_val", metric="fid")
                fid_ood, fid_devstd_ood, _ = pipeline.compute_metric_ratio("val", metric="fid")
            else:
                fid_id, fid_devstd_id, _ = pipeline.compute_metric_ratio("id_val", metric="fid")
                fid_ood, fid_devstd_ood, _ = pipeline.compute_metric_ratio("val", metric="fid")

            test_fid_id.append((fid_id, fid_devstd_id))
            test_fid_ood.append((fid_ood, fid_devstd_ood))

        print("\n\nFinal OOD Test scores: ", test_scores)
        print("Final FID_ID scores: ", test_fid_id)
        print("Final FID_OOD scores: ", test_fid_ood)
        print(f"Computed for split load_split = {load_split}")
        print()


def main():
    args = args_parser()

    assert not args.seeds is None, args.seeds
    assert args.metrics != ""

    if args.task == 'eval_suff':
        evaluate_suff(args)
        exit(0)
    if args.task == 'eval_fid':
        evaluate_fid(args)
        exit(0)
    if args.task == 'eval_nec':
        evaluate_nec(args)
        exit(0)
    if args.task == 'eval_acc':
        evaluate_acc(args)
        exit(0)
    if args.task == 'eval_metric':
        evaluate_metric(args)
        exit(0)

    test_scores = []
    test_suff, test_fid = [], []
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
            test_scores.append(test_score)
        elif config.task == 'test':
            test_score, test_loss = pipeline.load_task(load_param=True)
            sa = pipeline.evaluate("test", compute_suff=False)
            test_scores.append(sa['score'])
            # test_suff.append(sa["suff"])
            # test_fid.append(sa["fid"])
            print(f"Printing obtained and stored scores: {sa['score']} !=? {test_score}")
            # print(f"SUFF = {sa['suff']} +- {sa['suff_devstd']}")
            # print(f"FID_ = {sa['fid']} +- {sa['fid_devstd']}")
        elif config.task == 'debug_suff':
            config.task = 'test'
            
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
            pipeline.load_task(load_param=True)

            for sample_budget, explval_budget in zip([100, 1000, 1000, "all", "all"], [5, 5, 20, 5, 20]):
                config["expval_budget"] = explval_budget
                config["numsamples_budget"] = sample_budget

                sa = pipeline.evaluate("test")
                test_suff.append(sa["suff"])
                test_fid.append(sa["fid"])
                print(f"SUFF = {sa['suff']} +- {sa['suff_devstd']}")
            print(test_suff)
    print()
    print()
    print("Final OOD Test scores: ", round(np.mean(test_scores), 4), "+-", round(np.std(test_scores), 4))
    print("Final SUFF scores: ", round(np.mean(test_suff), 4), "+-", round(np.std(test_suff), 4))
    print("Final ROB FID_ scores: ", round(np.mean(test_fid), 4), "+-", round(np.std(test_fid), 4))
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

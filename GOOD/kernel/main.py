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

import warnings
import numpy as np
import wandb

if pyg_v == "2.4.0":
    torch.set_num_threads(4)

warnings.simplefilter(action='ignore', category=FutureWarning)

def initialize_model_dataset(config: Union[CommonArgs, Munch]) -> Tuple[torch.nn.Module, Union[dict, DataLoader]]:
    r"""
    Fix random seeds and initialize a GNN and a dataset. (For project use only)

    Returns:
        A GNN and a data loader.
    """
    # Initial
    reset_random_seed(config)

    print(f'#IN#\n-----------------------------------\n Task: {config.task}\n'
          f'{time.asctime(time.localtime(time.time()))}')
    # Load dataset
    print(f'#IN#Load Dataset {config.dataset.dataset_name}')
    dataset = load_dataset(config.dataset.dataset_name, config)
    print(f"#D#Dataset: {dataset}")
    print('#D#', dataset['train'][0] if type(dataset) is dict else dataset[0])
    print(dataset["id_val"].get(0))

    loader = create_dataloader(dataset, config)

    # Load model
    print('#IN#Loading model...')
    model = load_model(config.model.model_name, config)
    return model, loader


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
    print("Using ratios = ", ratios)

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

            for split in splits:
                pipeline.generate_explanation_examples(
                    seed,
                    ratios,
                    split,
                    metric=args.metrics,
                    edge_scores=edge_scores[split],
                    graphs=graphs[split],
                    labels=labels[split],
                    avg_graph_size=avg_graph_size[split],
                )


def generate_panel(args):
    load_splits = ["id"]
    for l, load_split in enumerate(load_splits):
        print("\n\n" + "-"*50)

        edge_scores_seed = []
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

            edge_scores = pipeline.generate_panel()
            edge_scores_seed.append(edge_scores)
        pipeline.generate_panel_all_seeds(edge_scores_seed)

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

def evaluate_metric(args):
    load_splits = ["id"]

    if args.splits != "":
        splits = args.splits.split("/")
    else:
        splits = ["id_val", "val", "test"]
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
            if l == 0 and i == 0:
                load_logger(config)
            
            model, loader = initialize_model_dataset(config)
            ood_algorithm = load_ood_alg(config.ood.ood_alg, config)
            pipeline = load_pipeline(config.pipeline, config.task, model, loader, ood_algorithm, config)
            pipeline.load_task(load_param=True, load_split=load_split)            

            if not (len(args.metrics.split("/")) == 1 and args.metrics.split("/")[0] == "acc"):
                (edge_scores, graphs, graphs_nx, labels, avg_graph_size, \
                causal_subgraphs_r, spu_subgraphs_r, expl_accs_r, causal_masks_r)  = pipeline.compute_scores_and_graphs(
                    ratios,
                    splits,
                    convert_to_nx=("suff" in args.metrics) and (not "suff_simple" in args.metrics),
                    is_weight="weight" in config.log_id
                )

            for metric in args.metrics.split("/"):
                print(f"\n\nEvaluating {metric.upper()} for seed {seed} with load_split {load_split}\n")
                for split in splits:
                    score, acc_int, results = pipeline.compute_metric_ratio(
                        ratios,
                        split,
                        metric=metric,
                        edge_scores=edge_scores[split],
                        graphs=graphs[split],
                        labels=labels[split],
                        avg_graph_size=avg_graph_size[split],
                        causal_masks_r=causal_masks_r[split]
                    )
                    
                    metrics_score[load_split][split][metric].append(score)
                    metrics_score[load_split][split][metric + "_acc_int"].append(acc_int)              

    if config.save_metrics:
        save_path = f"storage/metric_results/aggregated_{load_split}_results_necalpha{config.nec_alpha_1}" \
                    f"_numsamples{config.numsamples_budget}_randomexpl{config.random_expl}_ratios{args.ratios.replace('/','-')}" \
                    f"_metrics{args.metrics.replace('/','-')}" \
                    f"_{config.log_id}.json"
        if not os.path.exists(save_path):
            with open(save_path, 'w') as file:
                file.write("{}")
        with open(save_path, "r") as jsonFile:
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
                for metric in ["acc"]:
                    print(f"{metric} = {metrics_score[load_split][split][metric]}")

        print("\n\n", "-"*50, "\nPrinting evaluation averaged per seed")
        for split in splits:
            print(f"\nEval split {split}")
            for metric in args.metrics.split("/"):
                if "acc" == metric:
                    continue
                for div in ["L1", "KL"]:
                    s = [
                        metrics_score[load_split][split][metric][i][f"all_{div}"] for i in range(len(metrics_score[load_split][split][metric]))
                    ]
                    print_metric(metric + f" class all_{div}", s, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, metric+f"_{div}"])
                print(metrics_score[load_split][split][metric + "_acc_int"])
                print(s)
                print_metric(metric + "_acc_int", metrics_score[load_split][split][metric + "_acc_int"], results_aggregated, key=[config.dataset.dataset_name+" "+config.dataset.domain, config.complete_dirname, split, metric+"_acc_int"])
                
        if "acc" in args.metrics.split("/"):
            for split in splits + ["test", "test_R"]:
                print(f"\nEval split {split}")
                print_metric("acc", metrics_score[load_split][split]["acc"])

        print("\n\n", "-"*50, "\nComputing faithfulness")
        for split in splits:
            print(f"\nEval split {split}")            
            for div in ["L1", "KL"]:                
                suff = get_tensorized_metric(metrics_score[load_split][split]["suff_simple"], f"all_{div}")
                nec = get_tensorized_metric(metrics_score[load_split][split]["nec"], f"all_{div}")[:, :suff.shape[1]]             
                faith_aritm = aritm(suff, nec)
                faith_armonic = armonic(suff, nec)
                faith_gmean = gmean(suff, nec)
                print_metric(f"Faith. Aritm ({div})= \t\t", faith_aritm, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, f"faith_aritm_{div}"])
                print_metric(f"Faith. Armon ({div})= \t\t", faith_armonic, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, f"faith_armon_{div}"])
                print_metric(f"Faith. GMean ({div})= \t", faith_gmean, results_aggregated, key=[config.dataset.dataset_name + " " + config.dataset.domain, config.complete_dirname, split, f"faith_gmean_{div}"])

        print(f"Computed for split load_split = {load_split}\n\n\n")
    
    if config.save_metrics:
        with open(save_path, "w") as f:
            json.dump(results_aggregated, f)     
    
    print("Completed in ", datetime.now() - startTime, f" for {config.complete_dirname} {config.dataset.dataset_name}/{config.dataset.domain}")
    print("\n\n")
    sys.stdout.flush()

def print_faith(args):
    load_split = "id"
    config = config_summoner(args)

    split_metrics = ["id_val", "val", "test"]
    metrics = ["suff_simple_L1", "nec_L1", "faith_armon_L1"]
    model = config.complete_dirname
    dataset = config.dataset.dataset_name + " " + config.dataset.domain

    print("\nMODEL = \t", model)
    print("DATASET = \t", dataset)
    print("\n\n")
    print("Printing final metric values, separated according to their ratio k value:\n")

    results = {
        True: {split: {} for split in split_metrics},
        False: {split: {} for split in split_metrics},
    }
    big_rows = {s: "" for s in split_metrics}
    for split_metric in split_metrics:
        print(f"{split_metric}")
        for j, metric in enumerate(metrics):
            for i, random_expl in enumerate([False, True]):
                save_path = f"storage/metric_results/aggregated_{load_split}_results_necalpha{config.nec_alpha_1}" \
                            f"_numsamples{config.numsamples_budget}_randomexpl{random_expl}_ratios{args.ratios.replace('/','-')}" \
                            f"_metrics{args.metrics.replace('/','-')}" \
                            f"_{config.log_id}.json"
                if split_metric == split_metrics[0] and i == 0 and j == 0:
                    print("METRIC FILE = \t", save_path)

                with open(save_path, "r") as jsonFile:
                    data = json.load(jsonFile)
                if not model in data[dataset].keys() or not split_metric in data[dataset][model].keys():
                    continue
                
                results[random_expl][split_metric][f"values_{metric}"]     = [f"{d:.2f}" for d in data[dataset][model][split_metric][metric]]
                results[random_expl][split_metric][f"stds_{metric}"]       = [f"{d:.2f}" for d in data[dataset][model][split_metric][metric + "_std"]]

                if config.log_id == "isweight" and len(results[random_expl][split_metric][f"values_{metric}"]) == 4:
                    # append an empty entry for easy formatting on Sheet
                    results[random_expl][split_metric][f"values_{metric}"].append("-")
                    results[random_expl][split_metric][f"stds_{metric}"].append("-")

                row = "; ".join([f"{v} +- {s}" for v,s in zip(results[random_expl][split_metric][f"values_{metric}"], results[random_expl][split_metric][f"stds_{metric}"])])
                
                print(f"\t{metric + (' Rnd' if random_expl else ''):<25}:\t{row}")
                big_rows[split_metric] = big_rows[split_metric] + ";" + row

                if "faith" in metric and random_expl == True:
                    ratio_values = [ f"{float(results[True][split_metric][f'values_{metric}'][k]) / float(results[False][split_metric][f'values_{metric}'][k]):.2f}" for k in range(len(data[dataset][model][split_metric][metric]))]
                    
                    # ratio_stds = [ f"0.00" for _ in range(len(data[dataset][model][split_metric][metric]))]
                    # Z = Y/X = Rnd/Orig
                    # Computed using the Delta method (assuming R.V. asymptotically Gaussian and X and Y independent)
                    # https://stats.stackexchange.com/questions/291594/estimation-of-population-ratio-using-delta-method
                    # http://www.senns.uk/Stats_Notes/Variance_of_a_ratio.pdf
                    ratio_vars = [
                            float(results[True][split_metric][f'stds_{metric}'][k])**2 / float(results[False][split_metric][f'values_{metric}'][k])**2 + \
                                (float(results[True][split_metric][f'values_{metric}'][k])**2 * float(results[False][split_metric][f'stds_{metric}'][k])**2 / float(results[False][split_metric][f'values_{metric}'][k])**4)
                        for k in range(len(data[dataset][model][split_metric][metric]))
                    ]
                    ratio_stds = [ f"{d**0.5:.2f}" for d in ratio_vars]


                    row = "; ".join([f"{v} +- {s}" for v,s in zip(ratio_values, ratio_stds)])
                    
                    print(f"\t{metric + ' ratio':<25}:\t{row}")
                    big_rows[split_metric] = big_rows[split_metric] + ";" + row
    print("\n\nPrinting big rows (for ease of parsing on Excel):\n")
    for split_metric in split_metrics:
        print(f"{split_metric}: {big_rows[split_metric]}\n\n")

                    
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

    if args.task == 'eval_metric':
        evaluate_metric(args)
        exit(0)
    if args.task == 'plot_panel':
        generate_panel(args)
        exit(0)
    if args.task == 'plot_global':
        generate_global_explanation(args)
        exit(0)
    if args.task == 'plot_explanations':
        plot_explanation_examples(args)
        exit(0)
    if args.task == 'print_faith':
        print_faith(args)
        exit(0)
        

    run = None
    test_scores = defaultdict(list)
    channel_relevances, global_coeffs, global_weights = [], [], []
    elapsed_time_seed = []
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
                    project="your_name",
                    config=config,
                    entity="your_name",
                    name=f'{config.dataset.dataset_name}_{config.dataset.domain}{config.ood_dirname}_{config.util_model_dirname}_{config.random_seed}'
                )
                wandb.watch(pipeline.model, log="all", log_freq=10)

            # Train model
            pipeline.load_task()     
            elapsed_time_seed.append((datetime.now() - startTrainTime).total_seconds())
            print(f'\nTraining end ({elapsed_time_seed[-1]}).\n')

            # Eval model
            pipeline.task = 'test'
            test_score, test_loss = pipeline.load_task(load_param=True, load_split="id")
            test_scores["saved_score"].append(test_score)
            for s in ["id_val", "id_test", "val", "test"]:
                sa = pipeline.evaluate(s)
                test_scores[s].append(sa['score'])
            
            if config.global_side_channel and "simple_concept" in config.global_side_channel:
                channel_relevances.append(model.combinator.classifier[0].alpha_norm.cpu().numpy())
                print("\nConcept relevance scores for this run:\n", channel_relevances[-1], "\n")
        elif config.task == 'test':
            test_score, test_loss = pipeline.load_task(load_param=True, load_split="id")
            
            # Set weights manually for DEBUG
            # model.global_side_channel.classifier.classifier[0].weight = torch.nn.Parameter(
            #     torch.tensor([[1.0, 0., 0.]], device=config.device)
            # )
            # model.global_side_channel.classifier.classifier[0].bias = torch.nn.Parameter(
            #     torch.tensor([[-4.9]], device=config.device)
            # )
            # print(model.global_side_channel.classifier.classifier[0].weight)
            # model.global_side_channel.classifier.classifier[0].reset_parameters() # Remove Global Channel
            # model.gnn
            # model.extractor
            # model.classifier.classifier[0].reset_parameters() # Remove Local Channel

            for s in ["train", "id_val", "id_test", "val", "test"]:
                sa = pipeline.evaluate(s)
                test_scores[s].append(sa['score'])

            if config.global_side_channel:
                w = model.global_side_channel.classifier.classifier[0].weight.detach().cpu().numpy()
                b = model.global_side_channel.classifier.classifier[0].bias.detach().cpu().numpy()
                global_coeffs.append(-b / w[0][0])
                global_weights.append(w[0])
                
                if config.dataset.dataset_name in ("BAColor", "TopoFeature", "AIDS", "AIDSC1", "MUTAG0"):
                    print(f"\nWeight vector of global side channel:\nW: {w} b:{b}")
                    if config.dataset.dataset_name in ("AIDS", "AIDSC1"):
                        print(f"\nCoeff rule on x1: num_elements >= {-b / w[0][-1]}")
                    else:
                        print(f"\nCoeff rule on x1: x1 >= {global_coeffs[-1]}")

                if "simple_concept" in config.global_side_channel:
                    channel_relevances.append(model.combinator.classifier[0].alpha_norm.cpu().numpy())
                    print("\nConcept relevance scores for this run:\n", channel_relevances[-1], "\n")

                if "simple_linear" in config.global_side_channel:
                    channel_relevances.append(model.combinator.weight.detach().cpu().numpy())
                    print("\nConcept relevance scores for this run:\n", channel_relevances[-1], "\n")
            
    print(f"Average time elapsed = {np.mean(elapsed_time_seed):.2f} +- {np.std(elapsed_time_seed):.2f}")
    
    print("\n\nFinal accuracies: ")
    for s in test_scores.keys():
        print(f"{s.upper():<10} = {np.mean(test_scores[s]):.3f} +- {np.std(test_scores[s]):.3f}")

    if config.global_side_channel and config.model.model_name != "GIN":
        print(f"\n\nFinal accuracies")
        for s in test_scores.keys():
            tmp = np.array(test_scores[s])
            print(f"{s.upper():<10} = {np.mean(tmp):.3f} +- {np.std(tmp):.3f}")

        if "simple_concept" in config.global_side_channel or config.global_side_channel == "simple_linear":
            channel_relevances = np.concatenate(channel_relevances, axis=0)
            print(f"\n\nAveraged channel relevance scores")
            print(f"{channel_relevances.mean(0)} +- {channel_relevances.std(0)}")

        if config.dataset.dataset_name in ("BAColor", "TopoFeature", "AIDS", "AIDSC1"):
            print(f"\n\nGlobal side channel coefficient wrt x1")
            tmp = np.array(global_coeffs)
            print(f"{tmp.mean(0)} +- {tmp.std(0)}")

            print(f"\n\nAverage global channel weights")
            tmp = np.array(global_weights)
            print(f"{tmp.mean(0)} +- {tmp.std(0)}") 

    if config.global_side_channel in ("simple", "simple_filternode", "simple_concept", "simple_concept2"):
        with torch.no_grad():
            # Print weights of global channel
            if config.global_side_channel in ("simple", "simple_filternode"):
                w = model.global_side_channel.classifier.classifier[0].weight.detach().cpu().numpy()
                b = model.global_side_channel.classifier.classifier[0].bias.detach().cpu().numpy()
                print(f"\nWeight vector of global side channel:\nW: {w}\nb:{b}")
                print(f"\nBeta combination parameter of global side channel:{model.beta.sigmoid().item():.4f}\n")   
            elif config.global_side_channel in ("simple_concept", "simple_concept2"):
                print("\nConcept relevance scores:\n", model.combinator.classifier[0].alpha_norm.cpu().numpy(), "\n")

            # Print attention filter score for each unique node feature
            if config.global_side_channel == "simple_filternode":
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

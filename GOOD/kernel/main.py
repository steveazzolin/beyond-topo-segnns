r"""Kernel pipeline: main pipeline, initialization, task loading, etc.
"""
import os
import time
from typing import Tuple, Union

import torch.nn
from torch.utils.data import DataLoader

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

    loader = create_dataloader(dataset, config)

    # Load model
    print('#IN#Loading model...')
    model = load_model(config.model.model_name, config)

    return model, loader




def main():
    args = args_parser()

    assert not args.seeds is None, args.seeds

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
            sa = pipeline.evaluate("test", compute_suff=True)
            test_scores.append(sa['score'])
            test_suff.append(sa["suff"])
            test_fid.append(sa["fid"])
            print(f"Printing obtained and stored scores: {sa['score']} !=? {test_score}")
            print(f"SUFF = {sa['suff']} +- {sa['suff_devstd']}")
            print(f"FID_ = {sa['fid']} +- {sa['fid_devstd']}")
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
    print("Final SUFF scores: ", round(np.mean(test_fid), 4), "+-", round(np.std(test_fid), 4))
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

r"""Training pipeline: training/evaluation structure, batch training.
"""
import datetime
import os
import shutil
from typing import Dict
from typing import Union
import random
from collections import defaultdict
from scipy.stats import pearsonr

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_std
from munch import Munch
# from torch.utils.data import DataLoader
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data, InMemoryDataset
from torch_geometric.utils import to_networkx, from_networkx, to_undirected, sort_edge_index
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from GOOD.ood_algorithms.algorithms.BaseOOD import BaseOODAlg
from GOOD.utils.args import CommonArgs
from GOOD.utils.evaluation import eval_data_preprocess, eval_score
from GOOD.utils.logger import pbar_setting
from GOOD.utils.register import register
from GOOD.utils.train import nan2zero_get_mask
from GOOD.utils.initial import reset_random_seed
import GOOD.kernel.pipelines.xai_metric_utils as xai_utils
from GOOD.utils.splitting import split_graph, sparse_sort

pbar_setting["disable"] = False

class CustomDataset(InMemoryDataset):
    def __init__(self, root, samples, belonging, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.edge_types = {
            "inv": 0,
            "spu": 1,
            "added": 2,
            "BA": 3
        }
        
        data_list = []
        for i , G in enumerate(samples):
            data = from_networkx(G)
            # if data.ori_x is None:
            #     print(i)
            #     print(data)
            # if len(data.x.shape) == 1:
            #     data.x = data.x.unsqueeze(1)
            if len(data.ori_x.shape) == 1:
                data.ori_x = data.ori_x.unsqueeze(1)
            data.x = data.ori_x
            data.belonging = belonging[i]
            # data.origin = torch.tensor(list(map(lambda x: self.edge_types[x], data.origin)), dtype=int)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.data, self.slices = self.collate(data_list)


@register.pipeline_register
class Pipeline:
    r"""
    Kernel pipeline.

    Args:
        task (str): Current running task. 'train' or 'test'
        model (torch.nn.Module): The GNN model.
        loader (Union[DataLoader, Dict[str, DataLoader]]): The data loader.
        ood_algorithm (BaseOODAlg): The OOD algorithm.
        config (Union[CommonArgs, Munch]): Please refer to :ref:`configs:GOOD Configs and command line Arguments (CA)`.

    """

    def __init__(self, task: str, model: torch.nn.Module, loader: Union[DataLoader, Dict[str, DataLoader]],
                 ood_algorithm: BaseOODAlg,
                 config: Union[CommonArgs, Munch]):
        super(Pipeline, self).__init__()
        self.task: str = task
        self.model: torch.nn.Module = model
        self.loader: Union[DataLoader, Dict[str, DataLoader]] = loader
        self.ood_algorithm: BaseOODAlg = ood_algorithm
        self.config: Union[CommonArgs, Munch] = config

    def train_batch(self, data: Batch, pbar) -> dict:
        r"""
        Train a batch. (Project use only)

        Args:
            data (Batch): Current batch of data.

        Returns:
            Calculated loss.
        """
        data = data.to(self.config.device)

        self.ood_algorithm.optimizer.zero_grad()

        mask, targets = nan2zero_get_mask(data, 'train', self.config)
        node_norm = data.get('node_norm') if self.config.model.model_level == 'node' else None
        data, targets, mask, node_norm = self.ood_algorithm.input_preprocess(data, targets, mask, node_norm,
                                                                             self.model.training,
                                                                             self.config)
        edge_weight = data.get('edge_norm') if self.config.model.model_level == 'node' else None

        model_output = self.model(data=data, edge_weight=edge_weight, ood_algorithm=self.ood_algorithm)
        raw_pred = self.ood_algorithm.output_postprocess(model_output)

        loss = self.ood_algorithm.loss_calculate(raw_pred, targets, mask, node_norm, self.config)
        loss = self.ood_algorithm.loss_postprocess(loss, data, mask, self.config)

        self.ood_algorithm.backward(loss)

        return {'loss': loss.detach()}

    def train(self):
        r"""
        Training pipeline. (Project use only)
        """
        # config model
        print('Config model')
        self.config_model('train')

        # Load training utils
        print('Load training utils')
        self.ood_algorithm.set_up(self.model, self.config)

        # train the model
        for epoch in range(self.config.train.ctn_epoch, self.config.train.max_epoch):
            self.config.train.epoch = epoch
            print(f'Epoch {epoch}:')

            mean_loss = 0
            spec_loss = 0

            self.ood_algorithm.stage_control(self.config)

            pbar = tqdm(enumerate(self.loader['train']), total=len(self.loader['train']), **pbar_setting)
            for index, data in pbar:
                if data.batch is not None and (data.batch[-1] < self.config.train.train_bs - 1):
                    continue

                # Parameter for DANN
                p = (index / len(self.loader['train']) + epoch) / self.config.train.max_epoch
                self.config.train.alpha = 2. / (1. + np.exp(-10 * p)) - 1

                # train a batch
                train_stat = self.train_batch(data, pbar)
                mean_loss = (mean_loss * index + self.ood_algorithm.mean_loss) / (index + 1)

                if self.ood_algorithm.spec_loss is not None:
                    if isinstance(self.ood_algorithm.spec_loss, dict):
                        desc = f'ML: {mean_loss:.4f}|'
                        for loss_name, loss_value in self.ood_algorithm.spec_loss.items():
                            if not isinstance(spec_loss, dict):
                                spec_loss = dict()
                            if loss_name not in spec_loss.keys():
                                spec_loss[loss_name] = 0
                            spec_loss[loss_name] = (spec_loss[loss_name] * index + loss_value) / (index + 1)
                            desc += f'{loss_name}: {spec_loss[loss_name]:.4f}|'
                        pbar.set_description(desc[:-1])
                    else:
                        spec_loss = (spec_loss * index + self.ood_algorithm.spec_loss) / (index + 1)
                        pbar.set_description(f'M/S Loss: {mean_loss:.4f}/{spec_loss:.4f}')
                else:
                    pbar.set_description(f'Loss: {mean_loss:.4f}')

            # Eval training score

            # Epoch val
            print('Evaluating...')
            if self.ood_algorithm.spec_loss is not None:
                if isinstance(self.ood_algorithm.spec_loss, dict):
                    desc = f'ML: {mean_loss:.4f}|'
                    for loss_name, loss_value in self.ood_algorithm.spec_loss.items():
                        desc += f'{loss_name}: {spec_loss[loss_name]:.4f}|'
                    print(f'Approximated ' + desc[:-1])
                else:
                    print(f'Approximated average M/S Loss {mean_loss:.4f}/{spec_loss:.4f}')
            else:
                print(f'Approximated average training loss {mean_loss.cpu().item():.4f}')

            epoch_train_stat = self.evaluate('eval_train')
            id_val_stat = self.evaluate('id_val')
            id_test_stat = self.evaluate('id_test')
            val_stat = self.evaluate('val')
            test_stat = self.evaluate('test')

            # checkpoints save
            self.save_epoch(epoch, epoch_train_stat, id_val_stat, id_test_stat, val_stat, test_stat, self.config)

            # --- scheduler step ---
            self.ood_algorithm.scheduler.step()

        print('\nTraining end.\n')


    @torch.no_grad()
    def compute_robust_fidelity_m(self, split: str, debug=False):
        print(f"\n\n#D#Computing ROBUST FIDELITY MINUS over {split}")
        reset_random_seed(self.config)
        self.model.to("cpu")
        self.model.eval()        
        
        if self.config.numsamples_budget == "all":
            self.config.numsamples_budget = len(loader)

        loader = DataLoader(self.loader[split].dataset, batch_size=1, shuffle=False)
        pbar = tqdm(loader, desc=f'Eval {split.capitalize()}', total=len(loader), **pbar_setting)
        preds_all, labels, graphs = [], [], []
        causal_subgraphs, spu_subgraphs = [], []
        for data in pbar:
            data: Batch = data.to(self.config.device)
            output = self.model.probs(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm)
            preds_all.append(output[0].detach().cpu().numpy().tolist())
            labels.extend(data.y.detach().cpu().numpy().tolist())

            (causal_edge_index, causal_x, causal_batch, causal_edge_weight), \
                (spu_edge_index, spu_x, spu_batch, causal_edge_weight), edge_score = self.model.get_subgraph(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm, do_relabel=False)

            graphs.append(data.detach().cpu())
            causal_subgraphs.append(causal_edge_index.detach().cpu())
            spu_subgraphs.append(spu_edge_index.detach().cpu())
        labels = torch.tensor(labels)

        # eval_samples = [to_networkx(graphs[i],node_attrs=["x"]) for i in range(len(graphs))]
        # dataset = CustomDataset("", eval_samples, range(len(eval_samples)))
        # loader = DataLoader(dataset, batch_size=1, shuffle=False)       
        # pbar = tqdm(loader, desc=f'Eval {split.capitalize()}', total=len(loader), **pbar_setting)
        # preds_all2 = []
        # for data in pbar:
        #     data: Batch = data.to(self.config.device)
        #     output2 = self.model.probs(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm)
        #     preds_all2.append(output2[0].detach().cpu().numpy().tolist())

        # preds_all2 = torch.tensor(preds_all2)
        # preds_all = torch.tensor(preds_all)
        # tmp = torch.abs(preds_all.gather(1, labels.unsqueeze(1)) - preds_all2.gather(1, labels.unsqueeze(1)))
        # print(tmp.mean())
        # exit("SA")

        ##
        # Create interventional distribution
        ##
        
        eval_samples, belonging = [], []
        preds_ori, labels_ori = [], []
        pbar = tqdm(range(self.config.numsamples_budget), desc=f'Subsamling explanations', total=self.config.numsamples_budget, **pbar_setting)
        for i in pbar:
            preds_ori.append(preds_all[i])
            labels_ori.append(labels[i])

            G = to_networkx(
                graphs[i],
                node_attrs=["x"]
            )
            xai_utils.mark_edges(G, causal_subgraphs[i], spu_subgraphs[i])

            for m in range(self.config.expval_budget):
                G_c = xai_utils.sample_edges(G, "spu", self.config.fidelity_alpha_2)
                belonging.append(i)
                eval_samples.append(G_c)
                # xai_utils.draw(G_c, name=f"plots_rob_fid_examples/graph_{i}_{m}.png")

        ##
        # Compute new prediction and evaluate KL
        ##
        dataset = CustomDataset("", eval_samples, belonging)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
            
        pbar = tqdm(loader, desc=f'Eval intervened graphs', total=len(loader), **pbar_setting)
        preds_eval = []
        belonging = []
        for data in pbar:
            data: Batch = data.to(self.config.device)
            output = self.model.probs(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm)
            preds_eval.extend(output.detach().cpu().numpy().tolist())
            belonging.extend(data.belonging.detach().cpu().numpy().tolist())
        

        labels_ori_ori = torch.tensor(labels_ori)
        preds_eval = torch.tensor(preds_eval)
        preds_ori_ori = torch.tensor(preds_ori)
        preds_ori = preds_ori_ori.repeat_interleave(self.config.expval_budget, dim=0)
        labels_ori = labels_ori_ori.repeat_interleave(self.config.expval_budget, dim=0)

        print(preds_ori.shape, preds_eval.shape)        
        l1 = torch.abs(preds_eval.gather(1, labels_ori.unsqueeze(1)) - preds_ori.gather(1, labels_ori.unsqueeze(1)))
        l1_aggr = scatter_mean(l1, torch.tensor(belonging), dim=0)
        l1_aggr_std = scatter_std(l1, torch.tensor(belonging), dim=0)

        print(f"Robust Fidelity with L1 = {l1_aggr.mean()} +- {l1_aggr.std()} (in-sample avg dev_std = {(l1_aggr_std**2).mean().sqrt()})")
        return l1_aggr.mean(), l1_aggr.std()


    @torch.no_grad()
    def compute_debug(self, split: str, debug=False):
        reset_random_seed(self.config)
        self.model.eval()

        print(f"\n\n#D#Computing ROBUST FIDELITY MINUS over {split}")
        print(self.loader[split].dataset)
        if torch_geometric.__version__ == "2.4.0":
            print("Label distribution: ", self.loader[split].dataset.y.unique(return_counts=True))
        else:
            print("Label distribution: ", self.loader[split].dataset.data.y.unique(return_counts=True))

        loader = DataLoader(self.loader[split].dataset[:10], batch_size=1, shuffle=False)
        if self.config.numsamples_budget == "all":
            self.config.numsamples_budget = len(loader)
       
        pbar = tqdm(loader, desc=f'Eval {split.capitalize()}', total=len(loader), **pbar_setting)
        preds_all = []
        graphs = []
        causal_subgraphs = []
        spu_subgraphs = []
        causal_edge_weights, spu_edge_weights = [], []
        expl_accs = []
        labels = []
        edge_scores = []
        for data in pbar:
            data: Batch = data.to(self.config.device)
            output = self.model.probs(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm)
            preds_all.append(output[0].detach().cpu().numpy().tolist())
            labels.extend(data.y.detach().cpu().numpy().tolist())
            graphs.append(data.detach().cpu())

            (causal_edge_index, causal_x, causal_batch, causal_edge_weight), \
                (spu_edge_index, spu_x, spu_batch, spu_edge_weight), edge_score = self.model.get_subgraph(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm, do_relabel=False)
            edge_score = (edge_score - edge_score.min()) / (edge_score.max() - edge_score.min())
            spu_edge_weight = - spu_edge_weight # to compensate the '-' in CIGA split_graph(.)
            
            causal_subgraphs.append(causal_edge_index.detach().cpu())
            spu_subgraphs.append(spu_edge_index.detach().cpu())
            causal_edge_weights.append(causal_edge_weight.detach().cpu())
            spu_edge_weights.append(spu_edge_weight.detach().cpu())
            expl_accs.append(xai_utils.expl_acc(causal_subgraphs[-1], graphs[-1]))
            edge_scores.append({(u.item(), v.item()): edge_score[e].item() for e, (u,v) in enumerate(graphs[-1].edge_index.T)})
        labels = torch.tensor(labels)

        print(f"Explanation F1 score: {np.mean(expl_accs)}")
        plt.hist(torch.cat(causal_edge_weights, dim=0).numpy(), density=False)
        plt.title("causal_edge_weights")
        plt.savefig(f"GOOD/kernel/pipelines/plots/hist_causal_edge_weights_{self.config.util_model_dirname}.png")
        plt.close()

        plt.hist(torch.cat(spu_edge_weights, dim=0).numpy(), density=False)
        plt.title("spu_edge_weights")
        plt.savefig(f"GOOD/kernel/pipelines/plots/hist_spu_edge_weights_{self.config.util_model_dirname}.png")
        plt.close()

        # print("Plotting debug graphs")
        # k = 10
        # for i in range(10):
        #     G = to_networkx(graphs[i])
        #     xai_utils.mark_edges(G, causal_subgraphs[i], spu_subgraphs[i], causal_edge_weights[i], spu_edge_weights[i])
        #     xai_utils.draw_topk(self.config, G, subfolder="plots_rob_fid_examples", name=f"top{k}_graph_{i}", k=k)
        #     xai_utils.draw_gt(self.config, G, subfolder="plots_rob_fid_examples", name=f"gt_graph_{i}", gt=graphs[i].edge_gt, edge_index=graphs[i].edge_index)
        #     xai_utils.draw(self.config, G, subfolder="plots_rob_fid_examples", name=f"graph_{i}")

        ##
        # Create interventional distribution
        ##
        eval_samples = []
        belonging = []
        preds_ori, labels_ori, edge_scores_ori = [], [], []
        pbar = tqdm(range(self.config.numsamples_budget), desc=f'Subsamling explanations', total=self.config.numsamples_budget, **pbar_setting)
        i = 1
        for alpha in [0.7,]: # 0.7, 0.5, 0.3
            preds_ori.append(preds_all[i])
            labels_ori.append(labels[i])
            edge_scores_ori.append(edge_scores[i])

            G = to_networkx(
                graphs[i],
                node_attrs=["x"]
            )
            xai_utils.mark_edges(G, causal_subgraphs[i], spu_subgraphs[i], causal_edge_weights[i], spu_edge_weights[i])
            # xai_utils.draw(self.config, G, subfolder="plots_rob_fid_examples", name=f"graph_{i}")

            for m in range(self.config.expval_budget):
                G_c = xai_utils.sample_edges(G, "spu", alpha)
                belonging.append(i)
                eval_samples.append(G_c)
                # xai_utils.draw(self.config, G_c, subfolder="plots_rob_fid_examples", name=f"sample_{alpha}_{m}")
                

        ##
        # Compute new prediction and evaluate KL
        ##
        dataset = CustomDataset("", eval_samples, belonging)
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
            
        pbar = tqdm(loader, desc=f'Eval intervened graphs', total=len(loader), **pbar_setting)
        preds_eval = []
        belonging = []
        edge_scores_eval = []
        for data in pbar:
            data: Batch = data.to(self.config.device)
            data_ori = data.clone()
            output = self.model.probs(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm)
            preds_eval.extend(output.detach().cpu().numpy().tolist())
            belonging.extend(data.belonging.detach().cpu().numpy().tolist())

            (causal_edge_index, causal_x, causal_batch, causal_edge_weight), \
                (spu_edge_index, spu_x, spu_batch, spu_edge_weight), edge_score = self.model.get_subgraph(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm, do_relabel=False)
            edge_score = (edge_score - edge_score.min()) / (edge_score.max() - edge_score.min())
            edge_scores_eval.append({(u.item(), v.item()): edge_score[e].item() for e, (u,v) in enumerate(data_ori.edge_index.T)})

        labels_ori_ori = torch.tensor(labels_ori)
        preds_eval = torch.tensor(preds_eval)
        preds_ori_ori = torch.tensor(preds_ori)
        preds_ori = preds_ori_ori.repeat_interleave(self.config.expval_budget, dim=0)
        labels_ori = labels_ori_ori.repeat_interleave(self.config.expval_budget, dim=0)

        print(preds_ori.shape, preds_eval.shape)

        l1 = torch.abs(preds_eval.gather(1, labels_ori.unsqueeze(1)) - preds_ori.gather(1, labels_ori.unsqueeze(1)))
        l1_aggr = scatter_mean(l1, torch.tensor(belonging), dim=0)
        l1_aggr_std = scatter_std(l1, torch.tensor(belonging), dim=0)
        print(f"Robust Fidelity with L1 = {l1_aggr.mean()} +- {l1_aggr.std()} (in-sample avg dev_std = {(l1_aggr_std**2).mean().sqrt()})")

        c, d = 0, 0
        for i in range(len(edge_scores_ori)):
            for k, v in edge_scores_ori[i].items():
                if k in edge_scores_eval[i]:
                    d += abs(v - edge_scores_eval[i][k])
                    c += 1
        d = d / c
        print(f"Average L1 over edge_scores = {d}")
        exit()
        return l1_aggr.mean(), l1_aggr.std()
            
            


    @torch.no_grad()
    def compute_edge_score_divergence(self, split: str, debug=False):
        reset_random_seed(self.config)
        self.model.to("cpu")
        self.model.eval()

        print(f"\n\n#D#Computing L1 Divergence of Detector over {split}")
        print(self.loader[split].dataset)
        if torch_geometric.__version__ == "2.4.0":
            print("Label distribution: ", self.loader[split].dataset.y.unique(return_counts=True))
        else:
            print("Label distribution: ", self.loader[split].dataset.data.y.unique(return_counts=True))

        loader = DataLoader(self.loader[split].dataset, batch_size=1, shuffle=False)
        if self.config.numsamples_budget == "all":
            self.config.numsamples_budget = len(loader)
       
        pbar = tqdm(loader, desc=f'Eval {split.capitalize()}', total=len(loader), **pbar_setting)
        preds_all = []
        graphs = []
        causal_subgraphs = []
        spu_subgraphs = []
        causal_edge_weights, spu_edge_weights = [], []
        expl_accs = []
        labels = []
        edge_scores = []
        for data in pbar:
            data: Batch = data.to(self.config.device)
            graphs.append(data.detach().cpu())

            (causal_edge_index, causal_x, causal_batch, causal_edge_weight), \
                (spu_edge_index, spu_x, spu_batch, spu_edge_weight), edge_score = self.model.get_subgraph(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm, do_relabel=False)

            min_val = edge_score.min()
            edge_score = (edge_score - min_val) / (edge_score.max() - min_val)
            spu_edge_weight = - spu_edge_weight # to compensate the '-' in CIGA split_graph(.)
            
            causal_subgraphs.append(causal_edge_index.detach().cpu())
            spu_subgraphs.append(spu_edge_index.detach().cpu())
            causal_edge_weights.append(causal_edge_weight.detach().cpu())
            spu_edge_weights.append(spu_edge_weight.detach().cpu())
            expl_accs.append(xai_utils.expl_acc(causal_subgraphs[-1], graphs[-1]))
            edge_scores.append({(u.item(), v.item()): edge_score[e].item() for e, (u,v) in enumerate(graphs[-1].edge_index.T)})
        
        ##
        # Create interventional distribution
        ##
        eval_samples = []
        preds_ori, labels_ori, edge_scores_ori = [], [], []
        pbar = tqdm(range(self.config.numsamples_budget), desc=f'Subsamling explanations', total=self.config.numsamples_budget, **pbar_setting)
        for i in pbar:
            for alpha in [0.9]: # 0.7, 0.5, 0.3
                edge_scores_ori.append(edge_scores[i])

                G = to_networkx(
                    graphs[i],
                    node_attrs=["x"]
                )
                xai_utils.mark_edges(G, causal_subgraphs[i], spu_subgraphs[i], causal_edge_weights[i], spu_edge_weights[i])
                for m in range(self.config.expval_budget):
                    G_c = xai_utils.sample_edges(G, "spu", alpha)
                    eval_samples.append(G_c)
                

        ##
        # Compute new prediction and evaluate KL
        ##
        dataset = CustomDataset("", eval_samples, torch.arange(len(eval_samples)))
        loader = DataLoader(dataset, batch_size=1, shuffle=False)            
        pbar = tqdm(loader, desc=f'Eval intervened graphs', total=len(loader), **pbar_setting)
        edge_scores_eval = []
        for data in pbar:
            data: Batch = data.to(self.config.device)
            data_ori = data.clone()
            (causal_edge_index, causal_x, causal_batch, causal_edge_weight), \
                (spu_edge_index, spu_x, spu_batch, spu_edge_weight), edge_score = self.model.get_subgraph(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm, do_relabel=False)
            
            min_val = edge_score.min()
            edge_score = (edge_score - min_val) / (edge_score.max() - min_val)
            edge_scores_eval.append({(u.item(), v.item()): edge_score[e].item() for e, (u,v) in enumerate(data_ori.edge_index.T)})

        tmp = []
        for i in range(len(edge_scores_ori)):
            for k, v in edge_scores_ori[i].items():
                if k in edge_scores_eval[i]:
                    tmp.append(abs(v - edge_scores_eval[i][k]))
        print(f"Average L1 over edge_scores = {np.nanmean(tmp)} {np.nanstd(tmp)}")
        return np.nanmean(tmp), np.nanstd(tmp)                


    def plot_attn_distrib(self, attn_distrib, edge_scores=None):
        arrange_attn_distrib = []
        for l in range(len(attn_distrib[0])):
            arrange_attn_distrib.append([])
            for i in range(len(attn_distrib)):
                arrange_attn_distrib[l].extend(attn_distrib[i][l])
        
        path = f'GOOD/kernel/pipelines/plots/attn_distrib/{self.config.load_split}_{self.config.dataset.dataset_name}_{self.config.dataset.domain}_{self.config.util_model_dirname}_{self.config.random_seed}/'
        if not os.path.exists(path):
            os.makedirs(path)
        
        for l in range(len(arrange_attn_distrib)):
            plt.hist(arrange_attn_distrib[l], density=False)
            plt.savefig(path + f"l{l}.png")
            plt.close()
        if not edge_scores is None:
            scores = []
            for e in edge_scores:
                scores.extend(e.numpy().tolist())
            self.plot_hist_score(scores, density=False, log=True, name="edge_scores.png")

    def plot_hist_score(self, data, density=False, log=False, name="noname.png"):        
        path = f'GOOD/kernel/pipelines/plots/attn_distrib/{self.config.load_split}_{self.config.dataset.dataset_name}_{self.config.dataset.domain}_{self.config.util_model_dirname}_{self.config.random_seed}/'            
        plt.hist(data, density=density, bins=100, log=log)
        plt.xlim(0.0,1.1)
        plt.title(f"distrib. edge_scores (min={round(min(data), 2)}, max={round(max(data), 2)})")
        plt.savefig(path + name)
        plt.close()

    @torch.no_grad()
    def compute_sufficiency(self, split: str, debug=False):
        """
            Algorithm:
            1. compute and store P(Y|G')
            2. extract explanation and complement for each sample
            3. for each sample (or subset thereof)
                3.1 for a certain budget
                    3.1.1 replace its complement with the complement of another sample
                    3.1.2 compute P(Y|G')
                    3.1.3 compute d_i = d(P(Y|G'), P(Y|G))
            4. average d_i across all samples
        """
        reset_random_seed(self.config)
        self.model.to("cpu")
        self.model.eval()

        print(f"\n\n#D#Computing SUFF over {split}")
        print(self.loader[split].dataset)
        if torch_geometric.__version__ == "2.4.0":
            print("Label distribution: ", self.loader[split].dataset.y.unique(return_counts=True))
        else:
            print("Label distribution: ", self.loader[split].dataset.data.y.unique(return_counts=True))

        loader = DataLoader(self.loader[split].dataset, batch_size=1, shuffle=False)
        if self.config.numsamples_budget == "all":
            self.config.numsamples_budget = len(loader)

        pbar = tqdm(loader, desc=f'Extracting subgraphs {split.capitalize()}', total=len(loader), **pbar_setting)
        preds_all = []
        graphs = []
        causal_subgraphs = []
        spu_subgraphs = []
        causal_edge_weights, spu_edge_weights = [], []
        attn_distrib, edge_scores = [], []
        labels = []
        expl_accs = []
        for data in pbar:
            data: Batch = data.to(self.config.device)
            output = self.model.log_probs(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm)
            preds_all.append(output[0].detach().cpu().numpy().tolist())
            labels.extend(data.y.detach().cpu().numpy().tolist())
            graphs.append(data.detach().cpu())

            (causal_edge_index, causal_x, causal_batch, causal_edge_weight), \
                (spu_edge_index, spu_x, spu_batch, spu_edge_weight), edge_score = self.model.get_subgraph(
                    data=data,
                    edge_weight=None,
                    ood_algorithm=self.ood_algorithm,
                    do_relabel=False,
                    return_attn=True,
                    ratio=0.3
                )
            
            attn_distrib.append(self.model.attn_distrib)
            edge_scores.extend(edge_score.detach().cpu().numpy().tolist())
            causal_subgraphs.append(causal_edge_index.detach().cpu())
            spu_subgraphs.append(spu_edge_index.detach().cpu())
            causal_edge_weights.append(causal_edge_weight.detach().cpu())
            spu_edge_weights.append(spu_edge_weight.detach().cpu())
            expl_accs.append(xai_utils.expl_acc(causal_subgraphs[-1], graphs[-1]))
        labels = torch.tensor(labels)
        

        ##
        # Log attention distribution
        ## 
        self.plot_attn_distrib(attn_distrib, edge_scores)


        ##
        # Create interventional distribution
        ##        
        eval_samples = []
        belonging = []
        preds_ori, labels_ori, expl_acc_ori = [], [], []
        pbar = tqdm(range(self.config.numsamples_budget), desc=f'Int. distrib', total=self.config.numsamples_budget, **pbar_setting)
        for i in pbar:
            preds_ori.append(preds_all[i])
            labels_ori.append(labels[i])
            expl_acc_ori.append(expl_accs[i])

            G = to_networkx(
                graphs[i],
                node_attrs=["x"]
            )
            xai_utils.mark_edges(G, causal_subgraphs[i], spu_subgraphs[i])
            G_filt = xai_utils.remove_from_graph(G, "spu")
            num_elem = xai_utils.mark_frontier(G, G_filt)

            if num_elem == 0:
                print("\nZero frontier here ", i)
            if debug and i < 3:
                pos = xai_utils.draw(self.config, G, subfolder="plots_of_suff_scores", name=f"graph_{i}")                
                xai_utils.draw(self.config, G_filt, subfolder="plots_of_suff_scores", name=f"inv_graph_{i}", pos=pos)
            
            z, c = -1, 0
            idxs = np.random.permutation(np.arange(len(labels))[labels == labels[i]]) #pick random from same class
            invalid_idxs = set()
            while c < self.config.expval_budget:
                z += 1
                j = idxs[z]
                if z == len(idxs) - 1:
                    z = -1
                if j in invalid_idxs:
                    continue

                G_union = self.get_intervened_graph(
                        graphs[j],
                        invalid_idxs,
                        causal_subgraphs[j],
                        spu_subgraphs[j],
                        G_filt,
                        debug,
                        (i, j, c)
                )
                if G_union is None:
                    continue
                eval_samples.append(G_union)
                belonging.append(i)
                c += 1
        ##
        # Compute new prediction and evaluate KL
        ##
        dataset = CustomDataset("", eval_samples, belonging)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
            
        pbar = tqdm(loader, desc=f'Eval intervened graphs', total=len(loader), **pbar_setting)
        preds_eval = []
        belonging = []
        for data in pbar:
            data: Batch = data.to(self.config.device)
            output = self.model.log_probs(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm)
            preds_eval.extend(output.detach().cpu().numpy().tolist())
            belonging.extend(data.belonging.detach().cpu().numpy().tolist())

        expl_acc_ori = torch.tensor(expl_acc_ori)
        labels_ori = torch.tensor(labels_ori)
        preds_eval = torch.tensor(preds_eval)
        preds_ori_ori = torch.tensor(preds_ori)
        preds_ori = preds_ori_ori.repeat_interleave(self.config.expval_budget, dim=0)

        print(preds_ori.shape, preds_eval.shape)

        div = torch.nn.KLDivLoss(reduction="none", log_target=True)(preds_ori, preds_eval).sum(-1)
        div_aggr = scatter_mean(div, torch.tensor(belonging), dim=0)
        div_aggr_std = scatter_std(div, torch.tensor(belonging), dim=0)
        # print("Mean val. of div_aggr = ", div_aggr)
        # print("Dev std. of div_aggr = ", div_aggr_std)

        print(div_aggr)

        correct_samples = preds_ori_ori.argmax(-1) == labels_ori        
        print(f"for correct samples: {div_aggr[correct_samples].mean()} +- {div_aggr[correct_samples].std()}")

        incorrect_samples = preds_ori_ori.argmax(-1) != labels_ori
        print(f"for incorrect samples: {div_aggr[incorrect_samples].mean()} +- {div_aggr[incorrect_samples].std()}")

        for c in range(3):
            class_c = labels_ori == c
            acc = (preds_ori_ori.argmax(-1)[class_c] == labels_ori[class_c]).sum() / class_c.sum()
            print(f"for samples of class {c} (acc={acc:.2f}): {div_aggr[class_c].mean()} +- {div_aggr[class_c].std()}")

        div_aggr_mean = div_aggr.mean()
        print(f"Pearson corr between SUFF and expl_acc = {pearsonr(expl_acc_ori, div_aggr)}")
        print(f"Mean expl_acc for above avg SUFF = {expl_acc_ori[div_aggr >= div_aggr_mean].mean():.3f} +- {expl_acc_ori[div_aggr >= div_aggr_mean].std():.3f}", )
        print(f"Mean expl_acc for below avg SUFF = {expl_acc_ori[div_aggr < div_aggr_mean].mean():.3f} +- {expl_acc_ori[div_aggr < div_aggr_mean].std():.3f}", )
        print(f"Explanation F1 score: {np.mean(expl_accs)}")

        print(f"Mean of the dev_std computed for the int_distrib of each sample = {(div_aggr_std**2).mean().sqrt()}")
        print(f"SUFF results: ", div_aggr.mean().item(), div_aggr.std().item())

        # plt.hist(div_aggr.numpy(), density=False)
        # plt.savefig("GOOD/kernel/pipelines/plots/plots_of_suff_scores/hist_div_aggr.png")
        # plt.close()

        # c = 0
        # for j in range(500):
        #     if div_aggr[j] > 0.2 and labels_ori[j] == 1:
        #         G = to_networkx(graphs[j])
        #         xai_utils.mark_edges(G, causal_subgraphs[j], spu_subgraphs[j])
        #         xai_utils.draw(G, f"plots_of_suff_scores/bad_{j}.png")
        #         print(f"suff bad sample {j} (with pred={preds_ori_ori.argmax(-1)[j]}) = {div_aggr[j]} +- {div_aggr_std[j]}")
        #         c += 1
        #     if c == 10:
        #         break
        # c = 0
        # for j in range(100):
        #     if div_aggr[j] < 0.09 and labels_ori[j] == 1:
        #         G = to_networkx(graphs[j])
        #         xai_utils.mark_edges(G, causal_subgraphs[j], spu_subgraphs[j])
        #         xai_utils.draw(G, f"plots_of_suff_scores/good_{j}.png")
        #         c += 1
        #     if c == 10:
        #         break
        return div_aggr.mean().item(), div_aggr.std().item()
    

    @torch.no_grad()
    def get_subragphs_ratio(self, graphs, ratio, edge_scores):
        # # DEBUG OF SPARSE TOPK
        # i = 1
        # print("Weights:")
        # for j, (u,v) in enumerate(graphs[i].edge_index.T):
        #     if u < v:
        #         print((u.item(), v.item()), edge_scores[i][j])
        # print()

        # # case with twice the same graph (separated): Equal result        
        # print("\nSeparate and independent case (bs=1)\n")
        # (causal_edge_index, causal_edge_attr, causal_edge_weight), \
        #     (spu_edge_index, spu_edge_attr, spu_edge_weight) = split_graph(
        #         graphs[i],
        #         edge_scores[i],
        #         ratio
        #     )
        # print(causal_edge_index)
        # (causal_edge_index, causal_edge_attr, causal_edge_weight), \
        #     (spu_edge_index, spu_edge_attr, spu_edge_weight) = split_graph(
        #         graphs[i],
        #         edge_scores[i],
        #         ratio
        #     )
        # print(causal_edge_index)        
                
        # # case with twice the same graph (joined in batch): Equal result
        # print("\nJoined case (bs=3)\n")
        # data_joined = Batch().from_data_list([graphs[i], graphs[i], graphs[i]])

        # (causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch), \
        #     (spu_edge_index, spu_edge_attr, spu_edge_weight) = split_graph(
        #         data_joined,
        #         torch.cat((edge_scores[i], edge_scores[i], edge_scores[i]), dim=0),
        #         ratio,
        #         return_batch=True
        #     )

        # causal_edge_index = sort_edge_index(causal_edge_index)
        # for j, (u,v) in enumerate(causal_edge_index.T):
        #     num = graphs[i].num_nodes            
        #     if j % (len(causal_edge_index[0]) / 3) == 0:
        #         print("-"*20)
        #     u, v = int(u.item() - num * (j // (len(causal_edge_index[0]) / 3))) , int(v.item() - num * (j // (len(causal_edge_index[0]) / 3)))
        #     print((u, v), causal_edge_weight[j])
        # exit()
        # # END OF DEBUG
        

        spu_subgraphs, causal_subgraphs, expl_accs = [], [], []
        if "CIGA" in self.config.model.model_name:
            norm_edge_scores = [e.sigmoid() for e in edge_scores]
        else:
            norm_edge_scores = edge_scores

        # spu_subgraphs2, causal_subgraphs2, expl_accs2 = [], [], []
        # Select relevant subgraph (bs = 1)
        # for i in range(len(graphs)):
        #     (causal_edge_index, causal_edge_attr, causal_edge_weight), \
        #         (spu_edge_index, spu_edge_attr, spu_edge_weight) = split_graph(
        #             graphs[i],
        #             edge_scores[i],
        #             ratio
        #         )
        #     causal_subgraphs2.append(causal_edge_index.detach().cpu())
        #     spu_subgraphs2.append(spu_edge_index.detach().cpu())
        #     expl_accs2.append(xai_utils.expl_acc(causal_subgraphs2[-1], graphs[i], norm_edge_scores[i]) if hasattr(graphs[i], "edge_gt") else np.nan)

        # Select relevant subgraph (bs = all)
        big_data = Batch().from_data_list(graphs)        
        (causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch), \
            (spu_edge_index, spu_edge_attr, spu_edge_weight) = split_graph(
                big_data,
                torch.cat(edge_scores, dim=0),
                ratio,
                return_batch=True
            )
        
        cumnum = torch.tensor([g.num_nodes for g in graphs]).cumsum(0)
        cumnum[-1] = 0
        for j in range(causal_batch.max() + 1):
            causal_subgraphs.append(causal_edge_index[:, big_data.batch[causal_edge_index[0]] == j] - cumnum[j-1])
            spu_subgraphs.append(spu_edge_index[:, big_data.batch[spu_edge_index[0]] == j] - cumnum[j-1])
            expl_accs.append(xai_utils.expl_acc(causal_subgraphs[-1], graphs[j], norm_edge_scores[j]) if hasattr(graphs[j], "edge_gt") else np.nan)


        # assert torch.allclose(torch.tensor(expl_accs), torch.tensor(expl_accs2), atol=1e-5)
        # for k in range(causal_batch.max()):
        #     assert torch.all(causal_subgraphs[k] == causal_subgraphs2[k]), f"\n{causal_subgraphs[k]}\n{causal_subgraphs2[k]}"
        #     assert torch.all(spu_subgraphs[k] == spu_subgraphs2[k]), f"\n{spu_subgraphs[k]}\n{spu_subgraphs2[k]}"

        # big_data = Batch().from_data_list(graphs[:10])        
        # (causal_edge_index2, causal_edge_attr, causal_edge_weight, causal_batch2), \
        #     (spu_edge_index, spu_edge_attr, spu_edge_weight) = split_graph(
        #         big_data,
        #         torch.cat(edge_scores[:10], dim=0),
        #         ratio,
        #         return_batch=True
        #     )            
        # print(edge_scores[0].dtype)
        # for k in range(5):
        #     print(sort_edge_index(causal_subgraphs[k]))
        #     print(sort_edge_index(causal_edge_index[:, causal_batch == k]) - sum([graphs[j].num_nodes for j in range(k)]))
        #     print(sort_edge_index(causal_edge_index2[:, causal_batch2 == k]) - sum([graphs[j].num_nodes for j in range(k)]))
        #     print("-"*20)
        # exit()
        
        return causal_subgraphs, spu_subgraphs, expl_accs
    
    @torch.no_grad()
    def get_subragphs_weight(self, graphs, weight, edge_scores):
        spu_subgraphs, causal_subgraphs, cau_idxs, spu_idxs = [], [], [], []
        expl_accs = []
        # Select relevant subgraph
        for i in range(len(graphs)):
            cau_idxs.append(edge_scores[i] >= weight)
            spu_idxs.append(edge_scores[i] < weight)

            spu = (graphs[i].edge_index.T[spu_idxs[-1]]).T
            cau = (graphs[i].edge_index.T[cau_idxs[-1]]).T

            causal_subgraphs.append(cau)
            spu_subgraphs.append(spu)
            expl_accs.append(xai_utils.expl_acc(cau, graphs[i]) if hasattr(graphs[i], "edge_gt") else np.nan)
        return causal_subgraphs, spu_subgraphs, expl_accs, cau_idxs, spu_idxs

    @torch.no_grad()
    def evaluate_graphs(self, loader, log=False, **kwargs):
        pbar = tqdm(loader, desc=f'Eval intervened graphs', total=len(loader), **pbar_setting)
        preds_eval, belonging = [], []
        for data in pbar:
            data: Batch = data.to(self.config.device)            
            if log:
                output = self.model.log_probs(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm, **kwargs)
            else:
                output = self.model.probs(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm, **kwargs)
            preds_eval.extend(output.detach().cpu().numpy().tolist())
            belonging.extend(data.belonging.detach().cpu().numpy().tolist())
        preds_eval = torch.tensor(preds_eval)
        belonging = torch.tensor(belonging, dtype=int)
        return preds_eval, belonging

    def get_intervened_graph(self, metric, intervention_distrib, graph, empty_idx=None, causal=None, spu=None, source=None, debug=None, idx=None, bank=None, feature_intervention=False, feature_bank=None):
        i, j, c = idx
        if metric == "fid" or (metric == "suff" and intervention_distrib == "model_dependent" and causal is None):
            return xai_utils.sample_edges(graph, "spu", self.config.fidelity_alpha_2)
        elif metric == "nec":
            alpha = max(self.config.nec_alpha_1 - 0.1 * (j // 3), 0.1)
            return xai_utils.sample_edges(graph, "inv", alpha)
        elif metric == "suff" and intervention_distrib == "bank":
            G = graph.copy()
            I = bank[j].copy()
            ret = nx.union(G, I, rename=("", "T"))
            for n in range(random.randint(3, max(10, int(len(I) / 2)))):
                s_idx = random.randint(0, len(G) - 1)
                t_idx = random.randint(0, len(I) - 1)
                u = str(list(G.nodes())[s_idx])
                v = "T" + str(list(I.nodes())[t_idx])
                ret.add_edge(u, v, origin="added")
                ret.add_edge(v, u, origin="added")
            return ret
        elif metric == "suff" and intervention_distrib == "fixed":
            # random attach fixed graph to the explanation
            G = graph.copy()
            
            I = nx.DiGraph(nx.barabasi_albert_graph(random.randint(5, max(len(G), 8)), random.randint(1, 3)), seed=42) #BA1 -> nx.barabasi_albert_graph(randint(5, max(len(G), 8)), randint(1, 3))
            nx.set_edge_attributes(I, name="origin", values="spu")
            nx.set_node_attributes(I, name="x", values=[1.0])
            print("remebder to check values here for non-motif datasets")
            # nx.set_node_attributes(I, name="frontier", values=False)

            ret = nx.union(G, I, rename=("", "T"))
            for n in range(random.randint(3, max(10, int(len(G) / 2)))):
                s_idx = random.randint(0, len(G) - 1)
                t_idx = random.randint(0, len(I) - 1)
                u = str(list(G.nodes())[s_idx])
                v = "T" + str(list(I.nodes())[t_idx])
                ret.add_edge(u, v, origin="added")
                ret.add_edge(v, u, origin="added")
            return ret
        else:
            # G_t = to_networkx(
            #     graph,
            #     node_attrs=["ori_x"]
            # )
            G_t = graph.copy()
            xai_utils.mark_edges(G_t, causal, spu)
            G_t_filt = xai_utils.remove_from_graph(G_t, "inv")
            num_elem = xai_utils.mark_frontier(G_t, G_t_filt)

            if len(G_t_filt) == 0:
                empty_idx.add(j)
                # pos = xai_utils.draw(self.config, G_t, subfolder="plots_of_suff_scores", name=f"debug_graph_{j}")
                # xai_utils.draw(self.config, G_t_filt, subfolder="plots_of_suff_scores", name=f"spu_graph_{j}", pos=pos)
                return None

            if feature_intervention:
                if i == 0 and j == 0:
                    print(f"Applying feature interventions with alpha = {self.config.feat_int_alpha}")
                G_t_filt = xai_utils.feature_intervention(G_t_filt, feature_bank, self.config.feat_int_alpha)

            # G_union = xai_utils.random_attach(source, G_t_filt)
            G_union = xai_utils.random_attach_no_target_frontier(source, G_t_filt)
            if debug:
                if c <= 3 and i < 3:
                    xai_utils.draw(self.config, source, subfolder="plots_of_suff_scores", name=f"graph_{i}")
                    pos = xai_utils.draw(self.config, G_t, subfolder="plots_of_suff_scores", name=f"graph_{j}")
                    xai_utils.draw(self.config, G_t_filt, subfolder="plots_of_suff_scores", name=f"spu_graph_{j}", pos=pos)
                    xai_utils.draw(self.config, G_union, subfolder="plots_of_suff_scores", name=f"joined_graph_{i}_{j}")
                else:
                    exit()
        return G_union


    @torch.no_grad()
    def compute_metric_ratio(self, split: str, metric: str, intervention_distrib:str = "model_dependent", debug=False, edge_scores=None, graphs=None):
        assert metric in ["suff", "fid", "nec"]

        do_feature_intervention = False
        if "CIGA" in self.config.model.model_name and ("motif" in self.config.dataset.dataset_name.lower() or "twitter" in self.config.dataset.dataset_name.lower()):
            is_ratio = True
            weights = [0.6]
            assert weights[0] == self.model.att_net.ratio
        else:
            is_ratio = True
            weights = [0.3, 0.6, 0.9, 1.0]

        print(f"\n\n")
        print("-"*50)
        print(f"\n\n#D#Computing {metric.upper()} over {split} across ratios")
        print(self.loader[split].dataset)
        print(self.loader[split].dataset.data)
        reset_random_seed(self.config)
        self.model.eval()        

        
        intervent_bank = None
        features_bank = None
        if intervention_distrib == "bank":
            if torch_geometric.__version__ == "2.4.0": 
                features_bank = self.loader[split].dataset.x.unique(dim=0).cpu()
            else:
                features_bank = self.loader[split].dataset.data.x.unique(dim=0).cpu()
            print(f"Shape of feature bank = {features_bank.shape}")
            print(f"Creating interventional bank with {self.config.expval_budget} elements")
            intervent_bank = []
            max_g_size = max([d.num_nodes for d in self.loader[split].dataset])
            for i in range(self.config.expval_budget):
                I = nx.DiGraph(nx.barabasi_albert_graph(random.randint(5, max(int(max_g_size/2), 8)), 1), seed=42) #BA1 -> nx.barabasi_albert_graph(randint(5, max(len(G), 8)), randint(1, 3))
                nx.set_edge_attributes(I, name="origin", values="BA")
                if "motif" in self.config.dataset.dataset_name.lower():
                    nx.set_node_attributes(I, name="ori_x", values=1.0)
                else:
                    nx.set_node_attributes(I, name="ori_x", values=features_bank[random.randint(0, features_bank.shape[0]-1)].tolist())
                intervent_bank.append(I)  
        
        if self.config.numsamples_budget == "all" or self.config.numsamples_budget >= len(self.loader[split].dataset):
            self.config.numsamples_budget = len(self.loader[split].dataset)
            idx = np.arange(len(self.loader[split].dataset))        
        if self.config.numsamples_budget < len(self.loader[split].dataset):        
            idx, _ = train_test_split(
                np.arange(len(self.loader[split].dataset)),
                train_size=min(self.config.numsamples_budget, len(self.loader[split].dataset)) / len(self.loader[split].dataset),
                random_state=42,
                shuffle=True,
                stratify=self.loader[split].dataset.y if torch_geometric.__version__ == "2.4.0" else self.loader[split].dataset.data.y
            )
        

        loader = DataLoader(self.loader[split].dataset[idx], batch_size=1, shuffle=False)
        pbar = tqdm(loader, desc=f'Exctracting edge_scores {split.capitalize()}', total=len(loader), **pbar_setting)
        labels, attn_distrib = [], []
        if not edge_scores is None and not graphs is None:
            print("Using previous edge_scores and graphs")
            precomputed_scores = True
        else:
            precomputed_scores = False
            edge_scores, graphs = [], []
        for data in pbar:
            if not precomputed_scores:
                data: Batch = data.to(self.config.device)
                edge_score = self.model.get_subgraph(
                    data=data,
                    edge_weight=None,
                    ood_algorithm=self.ood_algorithm,
                    do_relabel=False,
                    return_attn=True,
                    ratio=None
                )
                edge_scores.append(edge_score.cpu())            
                graphs.append(data.detach().cpu())
                attn_distrib.append(self.model.attn_distrib)
            labels.extend(data.y.detach().cpu().numpy().tolist())
        labels = torch.tensor(labels)
        graphs_nx = [to_networkx(g, node_attrs=["ori_x"]) for g in graphs]

        # plot attn_distrib and compute the ratio between gt edges and all edges (gold cut ratio)
        # self.plot_attn_distrib(attn_distrib, edge_scores)
        if hasattr(graphs[0], "edge_gt"):
            num_gt_edges = torch.tensor([data.edge_gt.sum() for data in graphs])
            num_all_edges = torch.tensor([data.edge_index.shape[1] for data in graphs])
            print("\nGold ratio = ", torch.mean(num_gt_edges / num_all_edges), torch.std(num_gt_edges / num_all_edges))

        scores, results = [], {}
        for ratio in weights: #[0.0, 0.3, 0.5, 0.8, 1.0] for LECI
            reset_random_seed(self.config)
            if is_ratio:
                print(f"\n\nratio={ratio}\n\n")
            else:
                print(f"\n\nweight={ratio}\n\n")

            eval_samples, belonging, reference = [], [], []
            preds_ori, labels_ori, expl_acc_ori = [], [], []
            empty_idx = set()

            if is_ratio:
                causal_subgraphs, spu_subgraphs, expl_accs = self.get_subragphs_ratio(graphs, ratio, edge_scores)  
            else:
                causal_subgraphs, spu_subgraphs, expl_accs, causal_idxs, spu_idxs = self.get_subragphs_weight(graphs, ratio, edge_scores)

            pbar = tqdm(range(self.config.numsamples_budget), desc=f'Creating Intervent. distrib.', total=self.config.numsamples_budget, **pbar_setting)
            for i in pbar:
                G = graphs_nx[i].copy()
                xai_utils.mark_edges(G, causal_subgraphs[i], spu_subgraphs[i])
                
                # if i < 5:
                #     xai_utils.draw(self.config, G, subfolder="plots_of_suff_scores_sparsetopk", name=f"graph_{i}_r_{ratio}")
                # else:
                #     continue
                
                if metric == "suff" and intervention_distrib == "model_dependent":
                    G_filt = xai_utils.remove_from_graph(G, "spu")
                    num_elem = xai_utils.mark_frontier(G, G_filt)
                    if len(G_filt) == 0 or num_elem == 0:
                        assert False
                        continue
                    # G = G_filt # P(Y|G) vs P(Y|R)
                
                eval_samples.append(G)
                reference.append(len(eval_samples) - 1)
                belonging.append(-1)
                labels_ori.append(labels[i])
                expl_acc_ori.append(expl_accs[i])

                for m in range(self.config.expval_budget):                        
                    belonging.append(i)
                    eval_samples.append(G)
                continue

                if metric in ("fid", "nec") or len(empty_idx) == len(graphs) or intervention_distrib in ("fixed", "bank"):
                    if metric == "suff" and intervention_distrib in ("fixed", "bank") and i == 0:
                        print(f"Using {intervention_distrib} interventional distribution")
                    elif metric == "suff" and intervention_distrib == "model_dependent" and i < 2:
                        print("Empty graphs for SUFF. Rolling-back to FID")

                    # pos = xai_utils.draw(self.config, G, subfolder="plots_of_suff_scores", name=f"graph_{i}")                
                    for m in range(self.config.expval_budget):                        
                        G_c = self.get_intervened_graph(metric, intervention_distrib, G, idx=(i,m,-1), bank=intervent_bank)
                        # xai_utils.draw(self.config, G_c, subfolder="plots_of_suff_scores", name=f"nec_{i}_{m}", pos=pos)                
                        belonging.append(i)
                        eval_samples.append(G_c)
                elif metric == "suff":
                    z, c = -1, 0
                    idxs = np.random.permutation(np.arange(len(labels))) #pick random from every class
                    while c < self.config.expval_budget:
                        if z == len(idxs) - 1:
                            break
                        z += 1
                        j = idxs[z]
                        if j in empty_idx:
                            continue

                        G_union = self.get_intervened_graph(
                            metric,
                            intervention_distrib,
                            graphs_nx[j],
                            empty_idx,
                            causal_subgraphs[j],
                            spu_subgraphs[j],
                            G_filt,
                            debug,
                            (i, j, c),
                            feature_intervention=do_feature_intervention,
                            feature_bank=features_bank
                        )
                        if G_union is None:
                            continue
                        eval_samples.append(G_union)
                        belonging.append(i)
                        c += 1
                    for k in range(c, self.config.expval_budget): # if not enough interventions, pad with sub-sampling
                        G_c = xai_utils.sample_edges(G, "spu", self.config.fidelity_alpha_2)
                        belonging.append(i)
                        eval_samples.append(G_c)

            if len(eval_samples) == 0:
                print(f"\nZero intervened samples, skipping weight={ratio}")
                continue

            int_dataset = CustomDataset("", eval_samples, belonging)

            # # Inspect edge_scores of intervened edges
            # self.debug_edge_scores(int_dataset, reference, ratio)            
            
            # Compute new prediction and evaluate KL
            loader = DataLoader(int_dataset, batch_size=128, shuffle=False)
            if self.config.mask:
                print("Computing with masking")
                preds_eval, belonging = self.evaluate_graphs(loader, log=False if metric == "fid" else True, weight=ratio, is_ratio=is_ratio)
            else:
                preds_eval, belonging = self.evaluate_graphs(loader, log=False if metric == "fid" else True)
            preds_ori = preds_eval[reference]
            
            mask = torch.ones(preds_eval.shape[0], dtype=bool)
            mask[reference] = False
            preds_eval = preds_eval[mask]
            belonging = belonging[mask]            
            assert torch.all(belonging >= 0), f"{torch.all(belonging >= 0)}"

            print(preds_ori[0])
            print(preds_eval[:5])

            labels_ori_ori = torch.tensor(labels_ori)
            preds_ori_ori = preds_ori
            preds_ori = preds_ori_ori.repeat_interleave(self.config.expval_budget, dim=0)
            labels_ori = labels_ori_ori.repeat_interleave(self.config.expval_budget, dim=0)

            if metric in ("suff", "nec") and preds_eval.shape[0] > 0:
                div = torch.nn.KLDivLoss(reduction="none", log_target=True)(preds_ori, preds_eval).sum(-1)
                results[ratio] = div.numpy().tolist()
                if metric == "suff":
                    div = torch.exp(-div)
                elif metric == "nec":
                    div = 1 - torch.exp(-div)
                aggr = scatter_mean(div, belonging, dim=0)
                aggr_std = scatter_std(div, belonging, dim=0)
                score = aggr.mean().item()
            elif metric == "fid" and preds_eval.shape[0] > 0:
                l1 = torch.abs(preds_eval.gather(1, labels_ori.unsqueeze(1)) - preds_ori.gather(1, labels_ori.unsqueeze(1)))
                aggr = scatter_mean(l1, belonging, dim=0)
                aggr_std = scatter_std(l1, belonging, dim=0)
                score = aggr.mean().item()
            else:
                score = 0.
            scores.append(score)

            print(f"\nModel Val Acc of binarized graphs for ratio={ratio} = ", (labels_ori_ori == preds_ori_ori.argmax(-1)).sum() / preds_ori_ori.shape[0])
            print(f"Model XAI Acc of binarized graphs for weight={ratio} = ", np.mean(expl_accs))
            print(f"len(reference) = {len(reference)}")
            if preds_eval.shape[0] > 0:
                print(f"Model Val Acc over intervened graphs for ratio={ratio} = ", (labels_ori == preds_eval.argmax(-1)).sum() / preds_eval.shape[0])
                print(f"{metric.upper()} for ratio={ratio} = {score} +- {aggr.std()} (in-sample avg dev_std = {(aggr_std**2).mean().sqrt()})")
        return np.mean(scores), np.std(scores), results, edge_scores, graphs


    def debug_edge_scores(self, int_dataset, reference, ratio):
        loader = DataLoader(int_dataset[:1000], batch_size=1, shuffle=False)

        int_edge_scores, int_samples, ref_samples = [], [], []
        for i, data in enumerate(loader):
            if i in reference:
                ref_samples.append(i)
            else:
                int_samples.append(i)
            data: Batch = data.to(self.config.device)
            edge_score = self.model.get_subgraph(
                data=data,
                edge_weight=None,
                ood_algorithm=self.ood_algorithm,
                do_relabel=False,
                return_attn=False,
                ratio=None
            )
            int_edge_scores.append(edge_score)

        c = 0
        for k1, s1 in enumerate(ref_samples):
            assert c in ref_samples
            num_inv_ref = sum(int_dataset[s1].origin == 0)
            c+=1
            for k2 in range(self.config.expval_budget):
                assert c in int_samples
                assert sum(int_dataset[c].origin == 0) == num_inv_ref
                c+=1
                    
        attns = defaultdict(list)
        for i in range(len(int_samples)):
            for key, val in int_dataset.edge_types.items():
                original_mask = torch.zeros(int_edge_scores[int_samples[i]].shape[0], dtype=bool)
                original_mask[int_dataset[int_samples[i]].origin == val] = True

                attns[key].extend(int_edge_scores[int_samples[i]][original_mask].numpy().tolist())
        for key, _ in int_dataset.edge_types.items():
            self.plot_hist_score(attns[key], density=False, log=False, name=f"{key}_edge_scores_w{ratio}.png")
        
        attns = defaultdict(list)
        for i in range(len(ref_samples)):
            for key, val in int_dataset.edge_types.items():
                original_mask = torch.zeros(int_edge_scores[ref_samples[i]].shape[0], dtype=bool)
                original_mask[int_dataset[ref_samples[i]].origin == val] = True

                attns[key].extend(int_edge_scores[ref_samples[i]][original_mask].numpy().tolist())
        for key, _ in int_dataset.edge_types.items():
            self.plot_hist_score(attns[key], density=False, log=False, name=f"ref_{key}_edge_scores_w{ratio}.png")



    @torch.no_grad()
    def compute_accuracy_binarizing(self, split: str, givenR, debug=False):
        """
            Either computes the Accuracy of P(Y|R) or P(Y|G) under different weight/ratio binarizations
        """
        print(self.config.device)
        print(self.loader[split].dataset)
        if torch_geometric.__version__ == "2.4.0":
            print(self.loader[split].dataset.data)
            print(self.loader[split].dataset.y.unique(return_counts=True))

        if "CIGA" in self.config.model.model_name and ("motif" in self.config.dataset.dataset_name.lower() or "twitter" in self.config.dataset.dataset_name.lower()):
            is_ratio = True
            weights = [0.6]
            assert weights[0] == self.model.att_net.ratio
        else:
            is_ratio = True
            weights = [0.3, 0.6, 0.9, 1.0]

        reset_random_seed(self.config)
        self.model.eval()

        print(f"#D#Computing accuracy under post-hoc binarization for {split}")
        if givenR:
            print("Accuracy computed as P(Y|R)\n")
        else:
            print("Accuracy computed as P(Y|G)\n")
        print(self.loader[split].dataset)
        if self.config.numsamples_budget == "all" or self.config.numsamples_budget >= len(self.loader[split].dataset):
            self.config.numsamples_budget = len(self.loader[split].dataset)


        # loader = DataLoader(self.loader[split].dataset[:], batch_size=1, shuffle=False)
        # # print("\nFirst sample:\n")
        # # print(self.loader[split].dataset[0])
        # # print(self.loader[split].dataset[0].edge_index)
        # pbar = tqdm(loader, desc=f'Extracting edge_scores {split.capitalize()}', total=len(loader), **pbar_setting)
        # graphs = []
        # edge_scores = []
        # labels = []
        # for data in pbar:
        #     data: Batch = data.to(self.config.device)            
        #     edge_score = self.model.get_subgraph(
        #                 data=data,
        #                 edge_weight=None,
        #                 ood_algorithm=self.ood_algorithm,
        #                 do_relabel=False,
        #                 return_attn=False,
        #                 ratio=None
        #             )   
        #     edge_scores.append(edge_score.detach().cpu())
        #     labels.extend(data.y.detach().cpu().numpy().tolist())
        #     graphs.append(data.detach().cpu())
        # labels = torch.tensor(labels)


        #Test batching
        # print()
        
        # swap = self.loader[split].dataset[0].edge_index[:, 0].clone()
        # self.loader[split].dataset[0].edge_index[:, 0] = self.loader[split].dataset[0].edge_index[:, -1]
        # self.loader[split].dataset[0].edge_index[:, -1] = swap

        loader = DataLoader(self.loader[split].dataset[:], batch_size=256, shuffle=False, num_workers=2)
        pbar = tqdm(loader, desc=f'Extracting edge_scores {split.capitalize()} batched', total=len(loader), **pbar_setting)
        graphs = []
        edge_scores = []
        labels = []
        for data in pbar:
            data: Batch = data.to(self.config.device)            
            edge_score = self.model.get_subgraph(
                            data=data,
                            edge_weight=None,
                            ood_algorithm=self.ood_algorithm,
                            do_relabel=False,
                            return_attn=False,
                            ratio=None
                    )   
            labels.extend(data.y.detach().cpu().numpy().tolist())

            # print(data, data.x[0])
            # for i in range(data.batch.max()):
            #     g = data.get_example(i)
            #     print(self.loader[split].dataset[0], self.loader[split].dataset[0].x[0])
            #     print(g, g.x[0])
            #     exit()
            # print("-------------------------------")
            for j, g in enumerate(data.to_data_list()):
                g.ori_x = data.ori_x[data.batch == j]
                g.ori_edge_index = data.ori_edge_index[:, data.batch[data.ori_edge_index[0]] == j]
                graphs.append(g.detach().cpu())
                edge_scores.append(edge_score[data.batch[data.ori_edge_index[0]] == j].detach().cpu())
        labels = torch.tensor(labels)
        graphs_nx = [to_networkx(g, node_attrs=["ori_x"]) for g in graphs]

        # for j in range(len(graphs)):
        #     assert labels[j] == labels2[j]
        #     assert torch.allclose(edge_scores[j], edge_scores2[j], atol=1e-05), f"\n{edge_scores[j]}\n{edge_scores2[j]}\n{abs(edge_scores[j] - edge_scores2[j])}"
        #     assert torch.all(graphs[j].edge_index == graphs2[j].edge_index)
        #     assert torch.allclose(graphs[j].x, graphs2[j].x, atol=1e-05)         
        
        
        # self.plot_attn_distrib([[]], edge_scores)

        acc_scores = []
        for weight in weights:
            print(f"\n\nweight={weight}\n")
            eval_samples = []
            labels_ori = []
            empty_graphs = 0
            
            # Select relevant subgraph based on ratio
            if is_ratio:
                causal_subgraphs, spu_subgraphs, expl_accs = self.get_subragphs_ratio(graphs, weight, edge_scores)
            else:
                causal_subgraphs, spu_subgraphs, expl_accs, causal_idxs, spu_idxs = self.get_subragphs_weight(graphs, weight, edge_scores)            

            # Create interventional distribution     
            pbar = tqdm(range(self.config.numsamples_budget), desc=f'Int. distrib', total=self.config.numsamples_budget, **pbar_setting)
            for i in pbar:                
                G = graphs_nx[i].copy() #to_networkx(graphs[i], node_attrs=["ori_x"])
                xai_utils.mark_edges(G, causal_subgraphs[i], spu_subgraphs[i])
                G_filt = xai_utils.remove_from_graph(G, "spu")
                # xai_utils.draw(self.config, G, subfolder="plots_of_suff_scores", name=f"graph_{i}_topk")

                if len(G_filt) == 0:
                    empty_graphs += 1
                    continue

                eval_samples.append(G_filt if givenR else G) #G_filt for P(Y|R), G for P(Y|G)
                labels_ori.append(labels[i])

            # Compute accuracy
            if len(eval_samples) == 0:
                acc = 0.
            else:
                loader = DataLoader(CustomDataset("", eval_samples, torch.arange(len(eval_samples))), batch_size=128, shuffle=False)
                if self.config.mask:
                    print("Computing with masking")
                    preds, _ = self.evaluate_graphs(loader, log=True, weight=None if givenR else weight, is_ratio=is_ratio)
                else:                    
                    preds, _ = self.evaluate_graphs(loader, log=True)
                acc = (torch.tensor(labels_ori) == preds.argmax(-1)).sum() / (preds.shape[0] + empty_graphs)
            acc_scores.append(acc)
            print(f"\nModel Acc of binarized graphs for weight={weight} = ", acc)
            print(f"Model XAI Acc of binarized graphs for weight={weight} = ", np.mean(expl_accs))
            print("Num empty graphs = ", empty_graphs)
        return np.mean(acc_scores), np.std(acc_scores)


    @torch.no_grad()
    def evaluate(self, split: str, compute_suff=False):
        r"""
        This function is design to collect data results and calculate scores and loss given a dataset subset.
        (For project use only)

        Args:
            split (str): A split string for choosing the corresponding dataloader. Allowed: 'train', 'id_val', 'id_test',
                'val', and 'test'.

        Returns:
            A score and a loss.

        """
        stat = {'score': None, 'loss': None}
        if self.loader.get(split) is None:
            return stat
        self.model.eval()

        loss_all = []
        mask_all = []
        pred_all = []
        target_all = []
        pbar = tqdm(self.loader[split], desc=f'Eval {split.capitalize()}', total=len(self.loader[split]),
                    **pbar_setting)
        for data in pbar:
            data: Batch = data.to(self.config.device)

            mask, targets = nan2zero_get_mask(data, split, self.config)
            if mask is None:
                return stat
            node_norm = torch.ones((data.num_nodes,),
                                   device=self.config.device) if self.config.model.model_level == 'node' else None
            data, targets, mask, node_norm = self.ood_algorithm.input_preprocess(data, targets, mask, node_norm,
                                                                                 self.model.training,
                                                                                 self.config)
            model_output = self.model(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm)
            raw_preds = self.ood_algorithm.output_postprocess(model_output)

            # --------------- Loss collection ------------------
            loss: torch.tensor = self.config.metric.loss_func(raw_preds, targets, reduction='none') * mask
            mask_all.append(mask)
            loss_all.append(loss)

            # ------------- Score data collection ------------------
            pred, target = eval_data_preprocess(data.y, raw_preds, mask, self.config)
            pred_all.append(pred)
            target_all.append(target)

        # ------- Loss calculate -------
        loss_all = torch.cat(loss_all)
        mask_all = torch.cat(mask_all)
        stat['loss'] = loss_all.sum() / mask_all.sum()

        # --------------- Metric calculation including ROC_AUC, Accuracy, AP.  --------------------
        stat['score'] = eval_score(pred_all, target_all, self.config)
        # --------------- Metric SUFF  --------------------
        if compute_suff:
            suff, suff_devstd = 0, 0
            fid, fid_devstd = 0, 0
            suff, suff_devstd = self.compute_sufficiency("id_val")            
            fid, fid_devstd = self.compute_robust_fidelity_m("val")
        else:
            suff, suff_devstd = 0,0
            fid, fid_devstd = 0,0

        print(f'{split.capitalize()} {self.config.metric.score_name}: {stat["score"]:.4f}'
              f'{split.capitalize()} Loss: {stat["loss"]:.4f}')

        self.model.train()

        return {
            'score': stat['score'],
            'loss': stat['loss'],
            'suff': suff,
            'suff_devstd': suff_devstd,
            'fid': fid,
            'fid_devstd': fid_devstd
        }

    def load_task(self, load_param=False, load_split="ood"):
        r"""
        Launch a training or a test.
        """
        if self.task == 'train':
            self.train()
            return None, None
        elif self.task == 'test':
            # config model
            print('#D#Config model and output the best checkpoint info...')
            test_score, test_loss = self.config_model('test', load_param=load_param, load_split=load_split)
            return test_score, test_loss

    def config_model(self, mode: str, load_param=False, load_split="ood"):
        r"""
        A model configuration utility. Responsible for transiting model from CPU -> GPU and loading checkpoints.
        Args:
            mode (str): 'train' or 'test'.
            load_param: When True, loading test checkpoint will load parameters to the GNN model.

        Returns:
            Test score and loss if mode=='test'.
        """
        self.model.to(self.config.device)
        self.model.train()

        # load checkpoint
        if mode == 'train' and self.config.train.tr_ctn:
            ckpt = torch.load(os.path.join(self.config.ckpt_dir, f'last.ckpt'))
            self.model.load_state_dict(ckpt['state_dict'])
            best_ckpt = torch.load(os.path.join(self.config.ckpt_dir, f'best.ckpt'))
            self.config.metric.best_stat['score'] = best_ckpt['val_score']
            self.config.metric.best_stat['loss'] = best_ckpt['val_loss']
            self.config.train.ctn_epoch = ckpt['epoch'] + 1
            print(f'#IN#Continue training from Epoch {ckpt["epoch"]}...')

        if mode == 'test':
            try:
                ckpt = torch.load(self.config.test_ckpt, map_location=self.config.device)
            except FileNotFoundError:
                print(f'#E#Checkpoint not found at {os.path.abspath(self.config.test_ckpt)}')
                exit(1)
            if os.path.exists(self.config.id_test_ckpt):
                id_ckpt = torch.load(self.config.id_test_ckpt, map_location=self.config.device)
                # model.load_state_dict(id_ckpt['state_dict'])
                print(f'#IN#Loading best In-Domain Checkpoint {id_ckpt["epoch"]}...')
                print(f'#IN#Checkpoint {id_ckpt["epoch"]}: \n-----------------------------------\n'
                      f'Train {self.config.metric.score_name}: {id_ckpt["train_score"]:.4f}\n'
                      f'Train Loss: {id_ckpt["train_loss"].item():.4f}\n'
                      f'ID Validation {self.config.metric.score_name}: {id_ckpt["id_val_score"]:.4f}\n'
                      f'ID Validation Loss: {id_ckpt["id_val_loss"].item():.4f}\n'
                      f'ID Test {self.config.metric.score_name}: {id_ckpt["id_test_score"]:.4f}\n'
                      f'ID Test Loss: {id_ckpt["id_test_loss"].item():.4f}\n'
                      f'OOD Validation {self.config.metric.score_name}: {id_ckpt["val_score"]:.4f}\n'
                      f'OOD Validation Loss: {id_ckpt["val_loss"].item():.4f}\n'
                      f'OOD Test {self.config.metric.score_name}: {id_ckpt["test_score"]:.4f}\n'
                      f'OOD Test Loss: {id_ckpt["test_loss"].item():.4f}\n')
                print(f'#IN#Loading best Out-of-Domain Checkpoint {ckpt["epoch"]}...')
                print(f'#IN#Checkpoint {ckpt["epoch"]}: \n-----------------------------------\n'
                      f'Train {self.config.metric.score_name}: {ckpt["train_score"]:.4f}\n'
                      f'Train Loss: {ckpt["train_loss"].item():.4f}\n'
                      f'ID Validation {self.config.metric.score_name}: {ckpt["id_val_score"]:.4f}\n'
                      f'ID Validation Loss: {ckpt["id_val_loss"].item():.4f}\n'
                      f'ID Test {self.config.metric.score_name}: {ckpt["id_test_score"]:.4f}\n'
                      f'ID Test Loss: {ckpt["id_test_loss"].item():.4f}\n'
                      f'OOD Validation {self.config.metric.score_name}: {ckpt["val_score"]:.4f}\n'
                      f'OOD Validation Loss: {ckpt["val_loss"].item():.4f}\n'
                      f'OOD Test {self.config.metric.score_name}: {ckpt["test_score"]:.4f}\n'
                      f'OOD Test Loss: {ckpt["test_loss"].item():.4f}\n')

                print(f'#IN#ChartInfo {id_ckpt["id_test_score"]:.4f} {id_ckpt["test_score"]:.4f} '
                      f'{ckpt["id_test_score"]:.4f} {ckpt["test_score"]:.4f} {ckpt["id_val_score"]:.4f} {ckpt["val_score"]:.4f}', end='')

            else:
                print(f'#IN#No In-Domain checkpoint.')
                # model.load_state_dict(ckpt['state_dict'])
                print(f'#IN#Loading best Checkpoint {ckpt["epoch"]}...')
                print(f'#IN#Checkpoint {ckpt["epoch"]}: \n-----------------------------------\n'
                      f'Train {self.config.metric.score_name}: {ckpt["train_score"]:.4f}\n'
                      f'Train Loss: {ckpt["train_loss"].item():.4f}\n'
                      f'Validation {self.config.metric.score_name}: {ckpt["val_score"]:.4f}\n'
                      f'Validation Loss: {ckpt["val_loss"].item():.4f}\n'
                      f'Test {self.config.metric.score_name}: {ckpt["test_score"]:.4f}\n'
                      f'Test Loss: {ckpt["test_loss"].item():.4f}\n')

                print(
                    f'#IN#ChartInfo {ckpt["test_score"]:.4f} {ckpt["val_score"]:.4f}', end='')
            if load_param:
                if self.config.ood.ood_alg != 'EERM':
                    if load_split == "ood":
                        self.model.load_state_dict(ckpt['state_dict'])
                    elif load_split == "id":
                        self.model.load_state_dict(id_ckpt['state_dict'])
                    else:
                        raise ValueError(f"{load_split} not supported")
                else:
                    self.model.gnn.load_state_dict(ckpt['state_dict'])
            return ckpt["test_score"], ckpt["test_loss"]

    def save_epoch(self, epoch: int, train_stat: dir, id_val_stat: dir, id_test_stat: dir, val_stat: dir,
                   test_stat: dir, config: Union[CommonArgs, Munch]):
        r"""
        Training util for checkpoint saving.

        Args:
            epoch (int): epoch number
            train_stat (dir): train statistics
            id_val_stat (dir): in-domain validation statistics
            id_test_stat (dir): in-domain test statistics
            val_stat (dir): ood validation statistics
            test_stat (dir): ood test statistics
            config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.ckpt_dir`, :obj:`config.dataset`, :obj:`config.train`, :obj:`config.model`, :obj:`config.metric`, :obj:`config.log_path`, :obj:`config.ood`)

        Returns:
            None

        """
        state_dict = self.model.state_dict() if config.ood.ood_alg != 'EERM' else self.model.gnn.state_dict()
        ckpt = {
            'state_dict': state_dict,
            'train_score': train_stat['score'],
            'train_loss': train_stat['loss'],
            'id_val_score': id_val_stat['score'],
            'id_val_loss': id_val_stat['loss'],
            'id_test_score': id_test_stat['score'],
            'id_test_loss': id_test_stat['loss'],
            'val_score': val_stat['score'],
            'val_loss': val_stat['loss'],
            'test_score': test_stat['score'],
            'test_loss': test_stat['loss'],
            'time': datetime.datetime.now().strftime('%b%d %Hh %M:%S'),
            'model': {
                'model name': f'{config.model.model_name} {config.model.model_level} layers',
                'dim_hidden': config.model.dim_hidden,
                'dim_ffn': config.model.dim_ffn,
                'global pooling': config.model.global_pool
            },
            'dataset': config.dataset.dataset_name,
            'train': {
                'weight_decay': config.train.weight_decay,
                'learning_rate': config.train.lr,
                'mile stone': config.train.mile_stones,
                'shift_type': config.dataset.shift_type,
                'Batch size': f'{config.train.train_bs}, {config.train.val_bs}, {config.train.test_bs}'
            },
            'OOD': {
                'OOD alg': config.ood.ood_alg,
                'OOD param': config.ood.ood_param,
                'number of environments': config.dataset.num_envs
            },
            'log file': config.log_path,
            'epoch': epoch,
            'max epoch': config.train.max_epoch
        }
        if epoch < config.train.pre_train:
            return

        if not (config.metric.best_stat['score'] is None or config.metric.lower_better * val_stat[
            'score'] < config.metric.lower_better *
                config.metric.best_stat['score']
                or (id_val_stat.get('score') and (
                        config.metric.id_best_stat['score'] is None or config.metric.lower_better * id_val_stat[
                    'score'] < config.metric.lower_better * config.metric.id_best_stat['score']))
                or epoch % config.train.save_gap == 0):
            return

        if not os.path.exists(config.ckpt_dir):
            os.makedirs(config.ckpt_dir)
            print(f'#W#Directory does not exists. Have built it automatically.\n'
                  f'{os.path.abspath(config.ckpt_dir)}')
        saved_file = os.path.join(config.ckpt_dir, f'{epoch}.ckpt')
        torch.save(ckpt, saved_file)
        shutil.copy(saved_file, os.path.join(config.ckpt_dir, f'last.ckpt'))

        # --- In-Domain checkpoint ---
        if id_val_stat.get('score') and (
                config.metric.id_best_stat['score'] is None or config.metric.lower_better * id_val_stat[
            'score'] < config.metric.lower_better * config.metric.id_best_stat['score']):
            config.metric.id_best_stat['score'] = id_val_stat['score']
            config.metric.id_best_stat['loss'] = id_val_stat['loss']
            shutil.copy(saved_file, os.path.join(config.ckpt_dir, f'id_best.ckpt'))
            print('#IM#Saved a new best In-Domain checkpoint.')

        # --- Out-Of-Domain checkpoint ---
        # if id_val_stat.get('score'):
        #     if not (config.metric.lower_better * id_val_stat['score'] < config.metric.lower_better * val_stat['score']):
        #         return
        if config.metric.best_stat['score'] is None or config.metric.lower_better * val_stat[
            'score'] < config.metric.lower_better * \
                config.metric.best_stat['score']:
            config.metric.best_stat['score'] = val_stat['score']
            config.metric.best_stat['loss'] = val_stat['loss']
            shutil.copy(saved_file, os.path.join(config.ckpt_dir, f'best.ckpt'))
            print('#IM#Saved a new best checkpoint.')
        if config.clean_save:
            os.unlink(saved_file)

r"""Training pipeline: training/evaluation structure, batch training.
"""
import datetime
import os
import shutil
from typing import Dict
from typing import Union
from random import randint
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
from torch_geometric.utils import to_networkx, from_networkx
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

from GOOD.ood_algorithms.algorithms.BaseOOD import BaseOODAlg
from GOOD.utils.args import CommonArgs
from GOOD.utils.evaluation import eval_data_preprocess, eval_score
from GOOD.utils.logger import pbar_setting
from GOOD.utils.register import register
from GOOD.utils.train import nan2zero_get_mask
from GOOD.utils.initial import reset_random_seed
import GOOD.kernel.pipelines.xai_metric_utils as xai_utils
from GOOD.utils.splitting import split_graph

pbar_setting["disable"] = True

class CustomDataset(InMemoryDataset):
    def __init__(self, root, samples, belonging, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        data_list = []
        for i , G in enumerate(samples):
            data = from_networkx(G)
            if len(data.x.shape) == 1:
                data.x = data.x.unsqueeze(1)
            data.belonging = belonging[i]
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
        print(f"#D#Computing ROBUST FIDELITY MINUS over {split}")
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

        print(f"#D#Computing ROBUST FIDELITY MINUS over {split}")
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

        print(f"#D#Computing L1 Divergence of Detector over {split}")
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
        
        path = f'GOOD/kernel/pipelines/plots/attn_distrib/{self.config.load_split}_{self.config.util_model_dirname}/'
        if not os.path.exists(path):
            os.makedirs(path)
        
        for l in range(len(arrange_attn_distrib)):
            plt.hist(arrange_attn_distrib[l], density=False)
            plt.savefig(path + f"l{l}.png")
            plt.close()
        if not edge_scores is None:
            plt.hist(edge_scores, density=True)
            plt.savefig(path + f"edge_scores.png")
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

        print(f"#D#Computing SUFF over {split}")
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
        spu_subgraphs, causal_subgraphs = [], []
        expl_accs = []
        # Select relevant subgraph
        for i in range(len(graphs)):
            (causal_edge_index, causal_edge_attr, causal_edge_weight), \
                (spu_edge_index, spu_edge_attr, spu_edge_weight) = split_graph(
                    graphs[i],
                    edge_scores[i],
                    ratio
                )
            causal_subgraphs.append(causal_edge_index.detach().cpu())
            spu_subgraphs.append(spu_edge_index.detach().cpu())
            expl_accs.append(xai_utils.expl_acc(causal_subgraphs[-1], graphs[i]))
        return causal_subgraphs, spu_subgraphs, expl_accs
    

    @torch.no_grad()
    def evaluate_graphs(self, loader, log=False):
        pbar = tqdm(loader, desc=f'Eval intervened graphs', total=len(loader), **pbar_setting)
        preds_eval, belonging = [], []
        for data in pbar:
            data: Batch = data.to(self.config.device)
            if log:
                output = self.model.log_probs(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm)
            else:
                output = self.model.probs(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm)
            preds_eval.extend(output.detach().cpu().numpy().tolist())
            belonging.extend(data.belonging.detach().cpu().numpy().tolist())
        preds_eval = torch.tensor(preds_eval)
        return preds_eval, belonging

    def get_intervened_graph(self, graph, invalid_idxs, causal, spu, source, debug, idx):
        i, j, c = idx
        
        G_t = to_networkx(
            graph,
            node_attrs=["x"]
        )
        xai_utils.mark_edges(G_t, causal, spu)
        G_t_filt = xai_utils.remove_from_graph(G_t, "inv")
        num_elem = xai_utils.mark_frontier(G_t, G_t_filt)

        if num_elem == 0:
            invalid_idxs.add(j)
            return None

        # G_union = xai_utils.random_attach(source, G_t_filt)
        G_union = xai_utils.random_attach_no_target_frontier(source, G_t_filt)
        if debug and c <= 3 and i < 3:
            pos = xai_utils.draw(self.config, G_t, subfolder="plots_of_suff_scores", name=f"graph_{j}")
            xai_utils.draw(self.config, G_t_filt, subfolder="plots_of_suff_scores", name=f"spu_graph_{j}", pos=pos)
            xai_utils.draw(self.config, G_union, subfolder="plots_of_suff_scores", name=f"joined_graph_{i}_{j}")
        return G_union

    @torch.no_grad()
    def compute_sufficiency_ratio(self, split: str, debug=False):
        reset_random_seed(self.config)
        self.model.to("cpu")
        self.model.eval()

        print(f"#D#Computing SUFF over {split} divided by ratio")
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
        attn_distrib, edge_scores = [], []
        labels = []
        for data in pbar:
            data: Batch = data.to(self.config.device)
            output = self.model.log_probs(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm)
            preds_all.append(output[0].detach().cpu().numpy().tolist())
            labels.extend(data.y.detach().cpu().numpy().tolist())
            graphs.append(data.detach().cpu())

            edge_score = self.model.get_subgraph(
                        data=data,
                        edge_weight=None,
                        ood_algorithm=self.ood_algorithm,
                        do_relabel=False,
                        return_attn=True,
                        ratio=None
                    )
            
            edge_scores.append(edge_score.detach().cpu())
            attn_distrib.append(self.model.attn_distrib)
        labels = torch.tensor(labels)

        # Log attention distribution
        self.plot_attn_distrib(attn_distrib, edge_scores)

        suff_scores = []
        for ratio in [0.3, 0.6, 0.9]:
            print(f"\n\nratio={ratio}\n\n")

            eval_samples, belonging = [], []
            preds_ori, labels_ori, expl_acc_ori = [], [], []
            
            # Select relevant subgraph
            causal_subgraphs, spu_subgraphs, expl_accs = self.get_subragphs_ratio(graphs, ratio, edge_scores)

            # Create interventional distribution     
            pbar = tqdm(range(self.config.numsamples_budget), desc=f'Int. distrib', total=self.config.numsamples_budget, **pbar_setting)
            for i in pbar:                
                G = to_networkx(graphs[i], node_attrs=["x"])
                xai_utils.mark_edges(G, causal_subgraphs[i], spu_subgraphs[i])
                G_filt = xai_utils.remove_from_graph(G, "spu")
                num_elem = xai_utils.mark_frontier(G, G_filt)

                if num_elem == 0:
                    print(f"\nZero frontier here. Idx {i} Num nodes {len(G_filt.nodes())}")
                    continue
                if debug and i < 3:
                    pos = xai_utils.draw(self.config, G, subfolder="plots_of_suff_scores", name=f"graph_{i}")                
                    xai_utils.draw(self.config, G_filt, subfolder="plots_of_suff_scores", name=f"inv_graph_{i}", pos=pos)
                
                preds_ori.append(preds_all[i])
                labels_ori.append(labels[i])
                expl_acc_ori.append(expl_accs[i])

                z, c = -1, 0
                idxs = np.random.permutation(np.arange(len(labels))) #pick random from every class
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
            loader = DataLoader(CustomDataset("", eval_samples, belonging), batch_size=1, shuffle=False, num_workers=0)
            preds_eval, belonging = self.evaluate_graphs(loader, log=True)

            expl_acc_ori = torch.tensor(expl_acc_ori)
            labels_ori_ori = torch.tensor(labels_ori)            
            preds_ori_ori = torch.tensor(preds_ori)
            preds_ori = preds_ori_ori.repeat_interleave(self.config.expval_budget, dim=0)
            labels_ori = labels_ori_ori.repeat_interleave(self.config.expval_budget, dim=0)

            div = torch.nn.KLDivLoss(reduction="none", log_target=True)(preds_ori, preds_eval).sum(-1)
            div_aggr = scatter_mean(div, torch.tensor(belonging), dim=0)
            suff = div_aggr.mean().item()
            suff_scores.append(suff)

            print(div_aggr)
            print(f"Model Accuracy for ratio={ratio}: ", (labels_ori == preds_eval.argmax(-1)).sum() / preds_eval.shape[0])
            print(f"Explanation F1 score for ratio={ratio}: {np.mean(expl_accs)}")
            print(f"SUFF results for ratio={ratio}: ", suff, div_aggr.std().item())
        return np.mean(suff_scores), np.std(suff_scores)


    @torch.no_grad()
    def compute_robust_fidelity_m_ratio(self, split: str, debug=False):
        print(f"#D#Computing ROBUST FIDELITY MINUS over {split} across ratios")
        reset_random_seed(self.config)
        self.model.to("cpu")
        self.model.eval()        
        
        if self.config.numsamples_budget == "all":
            self.config.numsamples_budget = len(loader)

        loader = DataLoader(self.loader[split].dataset, batch_size=1, shuffle=False)
        pbar = tqdm(loader, desc=f'Eval {split.capitalize()}', total=len(loader), **pbar_setting)
        preds_all, labels, graphs = [], [], []
        edge_scores = []
        for data in pbar:
            data: Batch = data.to(self.config.device)
            output = self.model.probs(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm)
            graphs.append(data.detach().cpu())
            preds_all.append(output[0].detach().cpu().numpy().tolist())
            labels.extend(data.y.detach().cpu().numpy().tolist())

            edge_score = self.model.get_subgraph(
                data=data,
                edge_weight=None,
                ood_algorithm=self.ood_algorithm,
                do_relabel=False,
                ratio=None
            )
            edge_scores.append(edge_score)            
        labels = torch.tensor(labels)

        fid_scores = []
        for ratio in [0.3, 0.6, 0.9]:
            print(f"\n\nratio={ratio}\n\n")

            eval_samples, belonging = [], []
            preds_ori, labels_ori = [], []            
            # Select relevant subgraph
            causal_subgraphs, spu_subgraphs, _ = self.get_subragphs_ratio(graphs, ratio, edge_scores)

            # Create interventional distribution     
            pbar = tqdm(range(self.config.numsamples_budget), desc=f'Int. distrib', total=self.config.numsamples_budget, **pbar_setting)
            for i in pbar:                
                G = to_networkx(graphs[i], node_attrs=["x"])
                xai_utils.mark_edges(G, causal_subgraphs[i], spu_subgraphs[i])                
                preds_ori.append(preds_all[i])
                labels_ori.append(labels[i])

                for m in range(self.config.expval_budget):
                    G_c = xai_utils.sample_edges(G, "spu", self.config.fidelity_alpha_2)
                    belonging.append(i)
                    eval_samples.append(G_c)

            # Compute new prediction and evaluate KL
            loader = DataLoader(CustomDataset("", eval_samples, belonging), batch_size=1, shuffle=False, num_workers=2)
            preds_eval, belonging = self.evaluate_graphs(loader, log=False)

            labels_ori_ori = torch.tensor(labels_ori)
            preds_ori_ori = torch.tensor(preds_ori)
            preds_ori = preds_ori_ori.repeat_interleave(self.config.expval_budget, dim=0)
            labels_ori = labels_ori_ori.repeat_interleave(self.config.expval_budget, dim=0)

            print(preds_ori.shape, preds_eval.shape)        
            l1 = torch.abs(preds_eval.gather(1, labels_ori.unsqueeze(1)) - preds_ori.gather(1, labels_ori.unsqueeze(1)))
            l1_aggr = scatter_mean(l1, torch.tensor(belonging), dim=0)
            l1_aggr_std = scatter_std(l1, torch.tensor(belonging), dim=0)
            fid = l1_aggr.mean().item()
            fid_scores.append(fid)

            print(f"Robust Fidelity with L1 for ratio={ratio} = {fid} +- {l1_aggr.std()} (in-sample avg dev_std = {(l1_aggr_std**2).mean().sqrt()})")
        return np.mean(fid_scores), np.std(fid_scores)


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

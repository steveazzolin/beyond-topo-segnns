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
from torch_scatter import scatter_mean, scatter_std, scatter_min, scatter_max, scatter_add
from munch import Munch
# from torch.utils.data import DataLoader
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data, InMemoryDataset
from torch_geometric.utils import to_networkx, from_networkx, to_undirected, sort_edge_index, shuffle_node, is_undirected, contains_self_loops, contains_isolated_nodes, coalesce
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score as sk_roc_auc, f1_score, accuracy_score
from imblearn.under_sampling import RandomUnderSampler

from GOOD.ood_algorithms.algorithms.BaseOOD import BaseOODAlg
from GOOD.utils.args import CommonArgs
from GOOD.utils.evaluation import eval_data_preprocess, eval_score
from GOOD.utils.logger import pbar_setting
from GOOD.utils.register import register
from GOOD.utils.train import nan2zero_get_mask
from GOOD.utils.initial import reset_random_seed
import GOOD.kernel.pipelines.xai_metric_utils as xai_utils
from GOOD.utils.splitting import split_graph, sparse_sort, relabel

import wandb

pbar_setting["disable"] = True

class CustomDataset(InMemoryDataset):
    def __init__(self, root, samples, belonging, add_fake_edge_gt=False, dataset_name=None, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.edge_types = {
            "inv": 0,
            "spu": 1,
            "added": 2,
            "BA": 3
        }
        
        data_list = []
        for i , G in enumerate(samples):
            if type(G) is nx.classes.digraph.DiGraph:
                data = from_networkx(G)

                data.num_edge_removed = torch.tensor(0, dtype=torch.long)
            else:
                if G.edge_index.shape[1] == 0:
                    raise ValueError("Empty intervened graph")
                data = Data(ori_x=G.ori_x.clone(), edge_index=G.edge_index.clone())
                
                # Comment for FAITH
                # if hasattr(G, "edge_gt"): # added for stability of detector analysis
                    # data.edge_gt = G.edge_gt
                # elif add_fake_edge_gt:
                    # data.edge_gt = torch.zeros((data.edge_index.shape[1]), dtype=torch.long, device=data.edge_index.device)
                # if hasattr(G, "node_gt"): # added for stability of detector analysis
                    # data.node_gt = G.node_gt                

            if not hasattr(data, "ori_x"):
                print(i, data, type(data))
                print(G.nodes(data=True))
            if len(data.ori_x.shape) == 1:
                data.ori_x = data.ori_x.unsqueeze(1)

            edge_index_no_duplicates = coalesce(data.edge_index, None, is_sorted=False)[0]
            if edge_index_no_duplicates.shape[1] != data.edge_index.shape[1]:
                if dataset_name:
                    assert dataset_name == "GOODCMNIST"
                # edge_index contains duplicates. Remove them now to avoid proplems later
                if hasattr(data, "edge_attr"):
                    _, data.edge_attr = coalesce(data.edge_index, data.edge_attr, is_sorted=False)
                if hasattr(data, "edge_gt"):
                    _, data.edge_gt = coalesce(data.edge_index, data.edge_gt, is_sorted=False)
                if hasattr(data, "causal_mask"):
                    _, data.causal_mask = coalesce(data.edge_index, data.causal_mask, is_sorted=False)
                data.edge_index = edge_index_no_duplicates
                
            data.x = data.ori_x
            data.belonging = belonging[i]
            data.idx = i
            
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

    def train_batch(self, data: Batch, pbar, epoch:int) -> dict:
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

        model_output = self.model(
            data=data,
            edge_weight=edge_weight,
            ood_algorithm=self.ood_algorithm,
            max_num_epoch=self.config.train.max_epoch,
            curr_epoch=epoch
        )

        raw_pred = self.ood_algorithm.output_postprocess(model_output)

        warmup = 20 #if self.config.dataset.dataset_name != "MNIST" else 0
        if self.config.global_side_channel and self.config.dataset.dataset_name != "BAColor" and epoch < warmup:
            # pre-train the individual channels
            loss_global = self.ood_algorithm.loss_calculate(self.ood_algorithm.logit_global, targets, mask, node_norm, self.config, batch=data.batch)
            loss_global = loss_global.mean()
            loss_gnn    = self.ood_algorithm.loss_calculate(self.ood_algorithm.logit_gnn, targets, mask, node_norm, self.config, batch=data.batch)
            loss_gnn    = self.ood_algorithm.loss_postprocess(loss_gnn, data, mask, self.config, epoch)
            loss = loss_gnn + loss_global
        else:
            loss = self.ood_algorithm.loss_calculate(raw_pred, targets, mask, node_norm, self.config, batch=data.batch)
            loss = self.ood_algorithm.loss_postprocess(loss, data, mask, self.config, epoch)
        
        loss_clf_global_side_channel = torch.tensor(0.)

        if self.config.global_side_channel and self.config.dataset.dataset_name == "AIDSC1":
            loss += 0.01 * self.model.global_side_channel.classifier.classifier[0].weight.abs().sum()

        self.ood_algorithm.backward(loss)
        
        pred, target = eval_data_preprocess(data.y, raw_pred, mask, self.config)

        return {
            'loss': loss.detach(),
            'score': eval_score([pred], [target], self.config, pos_class=self.loader["train"].dataset.minority_class), 
            'clf_loss': self.ood_algorithm.clf_loss,
            'clf_global_side_loss': loss_clf_global_side_channel.item(),
            'l_norm_loss': self.ood_algorithm.l_norm_loss.item(),
            'entr_loss': self.ood_algorithm.entr_loss.item(),
        }


    def train(self):
        r"""
        Training pipeline. (Project use only)
        """
        if self.config.wandb:
            wandb.login()

        # config model
        print('Config model')
        self.config_model('train')

        # Load training utils
        print('Load training utils')
        self.ood_algorithm.set_up(self.model, self.config)

        print("Before training:")
        epoch_train_stat = self.evaluate('eval_train')
        id_val_stat = self.evaluate('id_val')
        id_test_stat = self.evaluate('id_test')

        if self.config.global_side_channel in ("simple_concept", "simple_concept2"):
            with torch.no_grad():
                print("Concept relevance scores:\n", self.model.combinator.classifier[0].alpha_norm.cpu().numpy())

        if self.config.wandb:
            wandb.log({
                    "epoch": -1,
                    "all_train_loss": epoch_train_stat["loss"],
                    "all_id_val_loss": id_val_stat["loss"],
                    "train_score": epoch_train_stat["score"],
                    "id_val_score": id_val_stat["score"],
                    "id_test_score": id_test_stat["score"],
                    "val_score": np.nan,
                    "test_score": np.nan,
            }, step=0)


        # train the model
        counter = 1
        for epoch in range(self.config.train.ctn_epoch, self.config.train.max_epoch):
            self.config.train.epoch = epoch
            print(f'\nEpoch {epoch}:')

            mean_loss = 0
            spec_loss = 0

            self.ood_algorithm.stage_control(self.config)

            pbar = tqdm(enumerate(self.loader['train']), total=len(self.loader['train']), **pbar_setting)
            loss_per_batch_dict = defaultdict(list)
            edge_scores = []
            node_feat_attn = torch.tensor([])
            raw_global_only, raw_gnn_only, raw_targets = [], [], []
            train_batch_score, clf_batch_loss, clf_global_batch_loss, l_norm_batch_loss, entr_batch_loss  = [], [], [], [], []
            for index, data in pbar:
                if data.batch is not None and (data.batch[-1] < self.config.train.train_bs - 1):
                    continue

                # train a batch
                train_stat = self.train_batch(data, pbar, epoch)
                train_batch_score.append(train_stat["score"])
                clf_batch_loss.append(train_stat["clf_loss"])
                clf_global_batch_loss.append(train_stat["clf_global_side_loss"])
                l_norm_batch_loss.append(train_stat["l_norm_loss"])
                entr_batch_loss.append(train_stat["entr_loss"])

                mean_loss = (mean_loss * index + self.ood_algorithm.mean_loss) / (index + 1)

                if self.config.model.model_name != "GIN":
                    edge_scores.append(self.ood_algorithm.edge_att.detach().cpu())

                if self.config.wandb:                    
                    for l in ("mean_loss", "spec_loss", "entropy_filternode_loss", "side_channel_loss"):
                        loss_per_batch_dict[l].append(getattr(self.ood_algorithm, l, np.nan))

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

                if self.config.global_side_channel:
                    pred_gnn, _ = eval_data_preprocess(data.y, self.ood_algorithm.logit_gnn.detach(), ~torch.isnan(data.y), self.config)
                    pred_global, targets = eval_data_preprocess(data.y, self.ood_algorithm.logit_global.detach(), ~torch.isnan(data.y), self.config)

                    raw_global_only.append(pred_global)
                    raw_gnn_only.append(pred_gnn)
                    raw_targets.append(targets)

            # Epoch val
            print('Evaluating...')
            print("Clf loss: ", np.mean(clf_batch_loss))
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

            epoch_train_stat = self.evaluate(
                'eval_train',
                compute_wiou=(self.config.dataset.dataset_name == "TopoFeature" or self.config.dataset.dataset_name == "GOODMotif") 
                                and 
                             self.config.model.model_name != "GIN"
            )
            id_val_stat = self.evaluate('id_val')
            id_test_stat = self.evaluate('id_test')

            if self.config.global_side_channel:
                global_only_score = eval_score(raw_global_only, raw_targets, self.config, pos_class=self.loader["train"].dataset.minority_class)
                gnn_only_score = eval_score(raw_gnn_only, raw_targets, self.config, pos_class=self.loader["train"].dataset.minority_class)
                print(f"Global only Acc: {global_only_score:.3f}")
                print(f"GNN only Acc: {gnn_only_score:.3f}")

                print(f"Beta param = {self.model.beta.sigmoid().item():.3f}")
                if self.config.global_side_channel in ("simple_filternode", ):
                    with torch.no_grad():
                        # Print attention filter score for each unique node feature
                        feats = self.loader["test"].dataset.x.unique(dim=0).to(self.config.device)
                        node_feat_attn = self.model.global_side_channel.node_filter(feats)
                        print(torch.cat((feats, node_feat_attn), dim=1))
                if self.config.global_side_channel in ("simple_concept", "simple_concept2"):
                    with torch.no_grad():
                        print("Concept relevance scores:\n", self.model.combinator.classifier[0].alpha_norm.cpu().numpy())

                if self.config.dataset.dataset_name in ("BAColor", "TopoFeature"):
                    w = self.model.global_side_channel.classifier.classifier[0].weight.detach().cpu().numpy()
                    b = self.model.global_side_channel.classifier.classifier[0].bias.detach().cpu().numpy()
                    print(f"\nWeight vector of global side channel:\nW: {w} b:{b}")
            
            if self.config.dataset.shift_type == "no_shift":
                val_stat = id_val_stat
                test_stat = id_test_stat
            else:
                val_stat = id_val_stat
                test_stat = id_test_stat

            if self.config.model.model_name != "GIN":
                print("edge_weight: ", torch.cat(edge_scores, dim=0).min(), torch.cat(edge_scores, dim=0).max(), torch.cat(edge_scores, dim=0).mean())

            if self.config.wandb:
                edge_scores = torch.cat(edge_scores, dim=0)
                log_dict = {
                    "epoch": epoch,
                    "clf_loss": np.mean(clf_batch_loss),
                    "clf_global_batch_loss": np.mean(clf_global_batch_loss),
                    "mean_loss": self.ood_algorithm.mean_loss,
                    "spec_loss": self.ood_algorithm.spec_loss,
                    "entropy_filternode_loss": getattr(self.ood_algorithm, "entropy_filternode_loss", np.nan),
                    "side_channel_loss": getattr(self.ood_algorithm, "side_channel_loss", np.nan),
                    "all_train_loss": epoch_train_stat["loss"],
                    "all_id_val_loss": id_val_stat["loss"],
                    "train_batch_score": np.mean(train_batch_score),
                    "train_score": epoch_train_stat["score"],
                    "id_val_score": id_val_stat["score"],
                    "id_test_score": id_test_stat["score"],
                    "val_score": val_stat["score"],
                    "test_score": test_stat["score"],
                    "beta_combination_param": self.model.beta.sigmoid().item() if self.config.global_side_channel else np.nan,
                    "edge_weight": wandb.Histogram(sequence=edge_scores, num_bins=100),
                    "filternode": wandb.Histogram(sequence=node_feat_attn.detach().cpu(), num_bins=100),
                    "GNN train score": gnn_only_score if self.config.global_side_channel else np.nan,
                    "global train score": global_only_score if self.config.global_side_channel else np.nan,
                    "wiou": epoch_train_stat["wiou"],
                    "l_norm_loss": np.mean(l_norm_batch_loss),
                    "entr_loss": np.mean(entr_batch_loss)
                }
                wandb.log(log_dict, step=counter)
                counter += 1

            # checkpoints save
            self.save_epoch(epoch, epoch_train_stat, id_val_stat, id_test_stat, val_stat, test_stat, self.config)

            # --- scheduler step ---
            self.ood_algorithm.scheduler.step()            

    @torch.no_grad()
    def get_subragphs_ratio(self, graphs, ratio, edge_scores, is_weight=False):
        """
            Cut graphs based on TopK or value thresholding strategy.
            If 'is_weight==False', use Top-'ratio'.
            Otherwise, use a thresholding with value 'ratio'
        """
        spu_subgraphs, causal_subgraphs, expl_accs, causal_masks = [], [], [], []
        norm_edge_scores = edge_scores

        # Select relevant subgraph (bs = all)
        big_data = Batch().from_data_list(graphs)        
        (causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch), \
            (spu_edge_index, spu_edge_attr, spu_edge_weight), mask = split_graph(
                big_data,
                torch.cat(edge_scores, dim=0),
                ratio,
                return_batch=True,
                is_weight=is_weight
            )
        
        cumnum = torch.tensor([g.num_nodes for g in graphs]).cumsum(0)
        cumnum[-1] = 0
        for j in range(causal_batch.max() + 1):
            causal_subgraphs.append(causal_edge_index[:, big_data.batch[causal_edge_index[0]] == j] - cumnum[j-1])
            spu_subgraphs.append(spu_edge_index[:, big_data.batch[spu_edge_index[0]] == j] - cumnum[j-1])
            expl_accs.append(xai_utils.expl_acc(causal_subgraphs[-1], graphs[j], norm_edge_scores[j]) if hasattr(graphs[j], "edge_gt") else (np.nan,np.nan))
            causal_masks.append(mask[big_data.batch[big_data.edge_index[0]] == j])        
        return causal_subgraphs, spu_subgraphs, expl_accs, causal_masks


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

    def get_indices_dataset(self, dataset, extract_all=False):
        if self.config.numsamples_budget == "all" or self.config.numsamples_budget >= len(dataset) or extract_all:
            idx = np.arange(len(dataset))        
        elif self.config.numsamples_budget < len(dataset):
            idx, _ = train_test_split(
                np.arange(len(dataset)),
                train_size=min(self.config.numsamples_budget, len(dataset)), # / len(dataset)
                random_state=42,
                shuffle=True,
                stratify=dataset.y if torch_geometric.__version__ == "2.4.0" else dataset.data.y
            )
        return idx
    
    @torch.no_grad()
    def compute_scores_and_graphs(self, ratios, splits, convert_to_nx=True, log=True, extract_all=False, is_weight=False):
        reset_random_seed(self.config)
        self.model.eval()

        edge_scores, graphs, labels = {"train": [], "id_val": [], "id_test": [], "test": [], "val": []}, {"train": [], "id_val": [], "id_test": [], "test": [], "val": []}, {"train": [], "id_val": [], "id_test": [], "test": [], "val": []}
        causal_subgraphs_r, spu_subgraphs_r, expl_accs_r, causal_masks = defaultdict(dict), defaultdict(dict), defaultdict(dict), defaultdict(dict)
        graphs_nx, avg_graph_size = dict(), dict()
        for SPLIT in splits:
            dataset = self.get_local_dataset(SPLIT, log=log)
            
            idx = self.get_indices_dataset(dataset, extract_all=extract_all)
            loader = DataLoader(dataset[idx], batch_size=512, shuffle=False, num_workers=2)
            for data in loader:
                data = data.to(self.config.device)
                edge_score = self.model.get_subgraph(
                    data=data,
                    edge_weight=None,
                    ood_algorithm=self.ood_algorithm,
                    do_relabel=False,
                    return_attn=False,
                    ratio=None
                )
                
                if self.config.random_expl:
                    edge_score = edge_score[
                        shuffle_node(torch.arange(edge_score.shape[0], device=edge_score.device), batch=data.batch[data.edge_index[0]])[1]
                    ]
                    data.edge_index, edge_score = to_undirected(data.edge_index, edge_score, reduce="mean")
                    
                for j, g in enumerate(data.to_data_list()):
                    g.ori_x = data.ori_x[data.batch == j]
                    edge_scores[SPLIT].append(edge_score[data.batch[data.edge_index[0]] == j].detach().cpu())
                    graphs[SPLIT].append(g.detach().cpu())
                labels[SPLIT].extend(data.y.detach().cpu().numpy().tolist())
            labels[SPLIT] = torch.tensor(labels[SPLIT])
            avg_graph_size[SPLIT] = np.mean([g.edge_index.shape[1] for g in graphs[SPLIT]])

            if convert_to_nx:                
                edge_attr_tokeep = ([] if g.edge_attr is None else ["edge_attr"]) + ([] if g.edge_gt is None else ["edge_gt"])
                graphs_nx[SPLIT] = [to_networkx(g, node_attrs=["ori_x"], edge_attrs=edge_attr_tokeep if edge_attr_tokeep != [] else None) for g in graphs[SPLIT]]
            else:
                graphs_nx[SPLIT] = list()

            for ratio in ratios:
                reset_random_seed(self.config)
                causal_subgraphs_r[SPLIT][ratio], spu_subgraphs_r[SPLIT][ratio], expl_accs_r[SPLIT][ratio], causal_masks[SPLIT][ratio] = self.get_subragphs_ratio(graphs[SPLIT], ratio, edge_scores[SPLIT], is_weight=is_weight)
                if log:
                    mask = torch.ones_like(labels[SPLIT], dtype=torch.bool)
                    print(f"F1 for r={ratio} = {np.mean([e[1] for e in expl_accs_r[SPLIT][ratio]]):.3f}")
                    print(f"WIoU for r={ratio} = {np.mean([e[0] for e in expl_accs_r[SPLIT][ratio]]):.3f}")
        return (edge_scores, graphs, graphs_nx, labels, \
                avg_graph_size, causal_subgraphs_r, spu_subgraphs_r,  expl_accs_r, causal_masks)


    @torch.no_grad()
    def compute_metric_ratio(
        self,
        ratios,
        split: str,
        metric: str,
        edge_scores,
        graphs,
        labels,
        avg_graph_size,
        causal_masks_r,
    ):
        assert metric in ["nec", "suff_simple"]
        
        if "sst2" in self.config.dataset.dataset_name.lower() and split in ("id_test", "id_val", "train"):
            weights = [0.6, 0.9, 1.0]
        else:
            weights = ratios

        print(f"\n\n")
        print("-"*50)
        print(f"\n\n#D#Computing {metric.upper()} over {split} across ratios (random_expl={self.config.random_expl})")

        reset_random_seed(self.config)
        self.model.eval()   

        scores, results, acc_ints = defaultdict(list), {}, []
        for ratio in weights:
            reset_random_seed(self.config)
            print(f"\n\nratio={ratio}\n\n")            

            eval_samples, belonging, reference = [], [], []
            preds_ori, labels_ori = [], []
            empty_idx = set()

            pbar = tqdm(range(len(edge_scores)), desc=f'Creating Intervent. distrib.', total=len(edge_scores), **pbar_setting)
            for i in pbar:
                if graphs[i].edge_index.shape[1] <= 6:
                    continue

                if metric == "nec" or len(empty_idx) == len(graphs):
                    intervened_graphs = xai_utils.sample_edges_tensorized_batched(
                        graphs[i],
                        nec_number_samples=self.config.nec_number_samples,
                        nec_alpha_1=self.config.nec_alpha_1,
                        avg_graph_size=avg_graph_size,
                        edge_index_to_remove=causal_masks_r[ratio][i],
                        sampling_type=self.config.samplingtype,
                        budget=self.config.expval_budget
                    )
                    if not intervened_graphs is None:
                        eval_samples.append(graphs[i])
                        reference.append(len(eval_samples) - 1)
                        belonging.append(-1)
                        labels_ori.append(labels[i])
                        belonging.extend([i] * len(intervened_graphs))
                        eval_samples.extend(intervened_graphs)
                elif metric == "suff_simple":
                    if ratio == 1.0:
                        eval_samples.extend([graphs[i]]*self.config.expval_budget)
                        belonging.extend([i]*self.config.expval_budget)
                    else:
                        intervened_graphs = xai_utils.sample_edges_tensorized_batched(
                            graphs[i],
                            nec_number_samples=self.config.nec_number_samples,
                            nec_alpha_1=self.config.nec_alpha_1*2,
                            avg_graph_size=avg_graph_size,
                            edge_index_to_remove=~causal_masks_r[ratio][i],
                            sampling_type=self.config.samplingtype,
                            budget=self.config.expval_budget
                        )
                        if not intervened_graphs is None:
                            eval_samples.append(graphs[i])
                            reference.append(len(eval_samples) - 1)
                            belonging.append(-1)
                            labels_ori.append(labels[i])
                            belonging.extend([i] * len(intervened_graphs))
                            eval_samples.extend(intervened_graphs)

            if len(eval_samples) <= 1:
                print(f"\nToo few intervened samples, skipping weight={ratio}")
                for c in labels_ori_ori.unique():
                    scores[c.item()].append(1.0)
                scores["all_KL"].append(1.0)
                scores["all_L1"].append(1.0)
                acc_ints.append(-1.0)
                continue
            
            # Compute new predictions and evaluate the difference with original predictions
            int_dataset = CustomDataset("", eval_samples, belonging)
            loader = DataLoader(int_dataset, batch_size=256, shuffle=False)
            preds_eval, belonging = self.evaluate_graphs(loader, log=False if "fid" in metric else True, eval_kl=True)
            preds_ori = preds_eval[reference]
            
            mask = torch.ones(preds_eval.shape[0], dtype=bool)
            mask[reference] = False
            preds_eval = preds_eval[mask]
            belonging = belonging[mask]            
            assert torch.all(belonging >= 0), f"{torch.all(belonging >= 0)}"

            labels_ori_ori = torch.tensor(labels_ori)
            preds_ori_ori = preds_ori
            preds_ori = preds_ori_ori.repeat_interleave(self.config.expval_budget, dim=0)
            labels_ori = labels_ori_ori.repeat_interleave(self.config.expval_budget, dim=0)

            aggr, aggr_std = self.get_aggregated_metric(metric, preds_ori, preds_eval, preds_ori_ori, labels_ori, belonging, results, ratio)
            
            for m in aggr.keys():
                assert aggr[m].shape[0] == labels_ori_ori.shape[0]
                for c in labels_ori_ori.unique():
                    idx_class = np.arange(labels_ori_ori.shape[0])[(labels_ori_ori == c).numpy()]
                    scores[c.item()].append(round(aggr[m][idx_class].mean().item(), 3))
                scores[f"all_{m}"].append(round(aggr[m].mean().item(), 3))

            assert preds_ori_ori.shape[1] > 1, preds_ori_ori.shape
            dataset_metric = self.loader["id_val"].dataset.metric
            if dataset_metric == "ROC-AUC":
                if not "fid" in metric:
                    preds_ori_ori = preds_ori_ori.exp() # undo the log
                    preds_eval = preds_eval.exp()
                acc = sk_roc_auc(labels_ori_ori.long(), preds_ori_ori[:, 1], multi_class='ovo')
                acc_int = sk_roc_auc(labels_ori.long(), preds_eval[:, 1], multi_class='ovo')
            elif dataset_metric == "F1":
                acc = f1_score(labels_ori_ori.long(), preds_ori_ori.argmax(-1), average="binary", pos_label=dataset.minority_class)
                acc_int = f1_score(labels_ori.long(), preds_eval.argmax(-1), average="binary", pos_label=dataset.minority_class)
            else:
                if preds_ori_ori.shape[1] == 1:
                    assert False
                acc = (labels_ori_ori == preds_ori_ori.argmax(-1)).sum() / (preds_ori_ori.shape[0])
                acc_int = (labels_ori == preds_eval.argmax(-1)).sum() / preds_eval.shape[0]

            acc_ints.append(acc_int.item())
            print(f"\nModel {dataset_metric} of binarized graphs for r={ratio} = ", round(acc.item(), 3))
            print(f"len(reference) = {len(reference)}")
            if preds_eval.shape[0] > 0:
                print(f"Model {dataset_metric} over intervened graphs for r={ratio} = ", round(acc_int.item(), 3))
                for c in labels_ori_ori.unique().numpy().tolist():
                    print(f"{metric.upper()} for r={ratio} class {c} = {scores[c][-1]} +- {aggr['KL'].std():.3f} (in-sample avg dev_std = {(aggr_std**2).mean().sqrt():.3f})")
                    del scores[c]
                for m in aggr.keys():
                    print(f"{metric.upper()} for r={ratio} all {m} = {scores[f'all_{m}'][-1]} +- {aggr[f'{m}'].std():.3f} (in-sample avg dev_std = {(aggr_std**2).mean().sqrt():.3f})")
        return scores, acc_ints, results


    def normalize_belonging(self, belonging):
        ret = []
        i = -1
        for j , elem in enumerate(belonging):
            if len(ret) > 0 and elem == belonging[j-1]:
                ret.append(i)
            else:
                i += 1
                ret.append(i)
        return ret

    def get_aggregated_metric(self, metric, preds_ori, preds_eval, preds_ori_ori, labels_ori, belonging, results, ratio):
        ret = {"KL": None, "L1": None}
        belonging = torch.tensor(self.normalize_belonging(belonging))

        if metric in ("nec", "suff_simple") and preds_eval.shape[0] > 0:
            div_kl = torch.nn.KLDivLoss(reduction="none", log_target=True)(preds_ori, preds_eval).sum(-1)
            div_l1 = torch.abs(preds_ori.exp() - preds_eval.exp()).sum(-1)

            if metric == "suff_simple":
                ret["KL"] = torch.exp(-scatter_mean(div_kl, belonging, dim=0))
                ret["L1"] = torch.exp(-scatter_mean(div_l1, belonging, dim=0))
            elif metric == "nec":
                ret["KL"] = 1 - torch.exp(-scatter_mean(div_kl, belonging, dim=0))
                ret["L1"] = 1 - torch.exp(-scatter_mean(div_l1, belonging, dim=0))
            aggr_std = scatter_std(div_l1, belonging, dim=0)               
        else:
            raise ValueError(metric)
        return ret, aggr_std

    def get_local_dataset(self, split, log=True):
        if torch_geometric.__version__ == "2.4.0" and log:
            print(self.loader[split].dataset)
            print(f"Data example from {split}: {self.loader[split].dataset.get(0)}")
            print(f"Label distribution from {split}: {self.loader[split].dataset.y.unique(return_counts=True)}")        

        dataset = self.loader[split].dataset
        
        if abs(dataset.y.unique(return_counts=True)[1].min() - dataset.y.unique(return_counts=True)[1].max()) > 1000:
            print(f"#D#Unbalanced warning for {self.config.dataset.dataset_name} ({split})")
        return dataset


    @torch.no_grad()
    def evaluate(self, split: str):
        r"""
        This function is design to collect data results and calculate scores and loss given a dataset subset.
        (For project use only)

        Args:
            split (str): A split string for choosing the corresponding dataloader. Allowed: 'train', 'id_val', 'id_test',
                'val', and 'test'.

        Returns:
            A score and a loss.

        """
        stat = {'score': None, 'loss': None, 'wiou': None}
        if self.loader.get(split) is None:
            return stat
        
        was_training = self.model.training
        self.model.eval()

        loss_all = []
        mask_all = []
        pred_all = []
        target_all = []
        wious_all = []
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
        stat['score'] = eval_score(pred_all, target_all, self.config, self.loader[split].dataset.minority_class)

        print(
            f'{split.capitalize()} {self.config.metric.score_name}: {stat["score"]:.4f} \t' + 
            f'{split.capitalize()} Loss: {stat["loss"]:.4f} \t'
        )


        if was_training:
            self.model.train()

        return {
            'score': stat['score'],
            'loss': stat['loss'],
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
            assert False
        if mode == 'test':
            try:
                ckpt = torch.load(self.config.test_ckpt, map_location=self.config.device)
            except FileNotFoundError:
                print(f'#E#Checkpoint not found at {os.path.abspath(self.config.test_ckpt)}')
                exit(1)
            if os.path.exists(self.config.id_test_ckpt):
                id_ckpt = torch.load(self.config.id_test_ckpt, map_location=self.config.device)
                # model.load_state_dict(id_ckpt['state_dict'])
                print(f'#IN#Loading best In-Domain Checkpoint {id_ckpt["epoch"]} in {self.config.id_test_ckpt}')
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

        # WARNING: Original reference metric is 'score'
        reference_metric = "loss"
        lower_better = 1 if reference_metric == "loss" else -1

        if not (config.metric.best_stat[reference_metric] is None or 
                lower_better * val_stat[reference_metric] < lower_better *
                config.metric.best_stat[reference_metric]
            or (id_val_stat.get(reference_metric) and (
                        config.metric.id_best_stat[reference_metric] is None or 
                        lower_better * id_val_stat[reference_metric] < lower_better * config.metric.id_best_stat[reference_metric]))
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
        if id_val_stat.get(reference_metric) and (
                config.metric.id_best_stat[reference_metric] is None or lower_better * id_val_stat[
            reference_metric] < lower_better * config.metric.id_best_stat[reference_metric]):
            config.metric.id_best_stat['score'] = id_val_stat['score']
            config.metric.id_best_stat['loss'] = id_val_stat['loss']
            shutil.copy(saved_file, os.path.join(config.ckpt_dir, f'id_best.ckpt'))
            print('#IM#Saved a new best In-Domain checkpoint.')

        
        if config.metric.best_stat[reference_metric] is None or lower_better * val_stat[
            reference_metric] < lower_better * \
                config.metric.best_stat[reference_metric]:
            config.metric.best_stat['score'] = val_stat['score']
            config.metric.best_stat['loss'] = val_stat['loss']
            shutil.copy(saved_file, os.path.join(config.ckpt_dir, f'best.ckpt'))
            print('#IM#Saved a new best checkpoint.')
        if config.clean_save:
            os.unlink(saved_file)

    def generate_panel(self):
        self.model.eval()

        splits = ["train", "id_val", "id_test"]
        n_row = 1
        fig, axs = plt.subplots(n_row, len(splits), figsize=(9,4))
        
        for i, split in enumerate(splits):            
            dataset = self.get_local_dataset(split)

            loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=2)
            edge_scores, effective_ratios = [], []
            for data in loader:
                data: Batch = data.to(self.config.device)   
                edge_score = self.model.get_subgraph(
                                data=data,
                                edge_weight=None,
                                ood_algorithm=self.ood_algorithm,
                                do_relabel=False,
                                return_attn=False,
                                ratio=None
                        )
                for j, g in enumerate(data.to_data_list()):
                    edge_scores.append(edge_score[data.batch[data.edge_index[0]] == j].detach().cpu().numpy().tolist())
                    if g.edge_index.shape[1] > 0:
                        effective_ratios.append(float((g.edge_gt.sum() if hasattr(g, "edge_gt") and not g.edge_gt is None else 0.) / (g.edge_index.shape[1])))
            return edge_scores            

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        path = f'GOOD/kernel/pipelines/plots/panels/{self.config.ood_dirname}/'
        if not os.path.exists(path):
            os.makedirs(path)

        path += f"{self.config.load_split}_{self.config.dataset.dataset_name}_{self.config.dataset.domain}_{self.config.util_model_dirname}_{self.config.random_seed}"
        plt.savefig(path + ".png")
        plt.savefig(f'GOOD/kernel/pipelines/plots/panels/pdfs/{self.config.load_split}_{self.config.dataset.dataset_name}_{self.config.dataset.domain}_{self.config.util_model_dirname}_{self.config.random_seed}.pdf')
        print("\n Saved plot ", path, "\n")
        plt.close()
        return edge_scores
    
    def generate_panel_all_seeds(self, edge_scores_seed):
        n_row, n_col = 5, 2
        fig, axs = plt.subplots(n_row, n_col, figsize=(20,16))
        for j in range(len(edge_scores_seed)):   
            ax = axs[j // n_col, j % n_col]

            ax.hist(np.concatenate(edge_scores_seed[j]), density=True, log=False, bins=100)
            ax.set_title(f"seed {j+1}", fontsize=15)
            ax.set_yticks([])
            ax.tick_params(axis='y', labelsize=12)
            ax.tick_params(axis='x', labelsize=12)
            ax.set_xlim((0.0,1.0))
        fig.supxlabel('explanation relevance scores', fontsize=18)
        fig.supylabel('density', fontsize=18)
        fig.suptitle(self.config.model.model_name.replace("GIN", ""), fontsize=22)
        
        path = f'GOOD/kernel/pipelines/plots/panels/{self.config.ood_dirname}/'
        if not os.path.exists(path):
            os.makedirs(path)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        path += f"{self.config.load_split}_{self.config.dataset.dataset_name}_{self.config.dataset.domain}_{self.config.util_model_dirname}_allseeds"
        plt.savefig(path + ".png")
        plt.savefig(f'GOOD/kernel/pipelines/plots/panels/pdfs_new/{self.config.load_split}_{self.config.dataset.dataset_name}_{self.config.dataset.domain}_{self.config.util_model_dirname}_allseeds.pdf')
        print("\n Saved plot ", path, "\n")
        plt.close()

    @torch.no_grad()
    def generate_global_explanation(self):
        self.model.eval()
        splits = ["id_val"]
        n_row = 1
        fig, axs = plt.subplots(n_row, len(splits), figsize=(4*len(splits),4))

        if len(splits) == 1:
            axs = [axs]
        
        w = self.model.global_side_channel.classifier.classifier[0].weight.cpu().numpy()
        b = self.model.global_side_channel.classifier.classifier[0].bias.cpu().numpy()
        print(f"\nWeight vector of global side channel:\nW: {w}\nb:{b}")
        print(f"\nBeta combination parameter of global side channel:{self.model.beta.sigmoid().item():.4f}\n")

        for i, split in enumerate(splits):
            dataset = self.get_local_dataset(split)

            loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
            samples, preds_global_only, preds_gnn_only = [], [], []
            for data in loader:
                data: Batch = data.to(self.config.device)   
                encoding, _ = self.model.global_side_channel.encode(data=data)
                logits_global_only, atnn = self.model.global_side_channel(data=data)
                logits_gnn_only = self.ood_algorithm.output_postprocess(
                    self.model(data=data, exclude_global=True)
                )
                samples.extend(encoding.cpu().tolist())

                if logits_global_only.shape[-1] > 1:
                    preds_global_only.extend(logits_global_only.argmax(dim=1).cpu().tolist())
                    preds_gnn_only.extend(logits_gnn_only.argmax(dim=1).cpu().tolist())
                else:
                    preds_global_only.extend((logits_global_only.sigmoid() >= 0.5).to(torch.long).cpu().tolist())
                    preds_gnn_only.extend((logits_gnn_only.sigmoid() >= 0.5).to(torch.long).cpu().tolist())
            
            samples = torch.tensor(samples)
            preds_global_only = torch.tensor(preds_global_only).reshape(-1)
            preds_gnn_only = torch.tensor(preds_gnn_only).reshape(-1)
            labels = dataset.y.reshape(-1)

            # Plot based on GT label
            axs[int(i/n_row)].scatter(samples[labels == 0, 2], samples[labels == 0, 0], c="orange", alpha=0.4, label="y=0")
            axs[int(i/n_row)].scatter(samples[labels == 1, 2], samples[labels == 1, 0], c="blue", alpha=0.4, label="y=1")
            
            acc = self.evaluate(split, compute_suff=False)["score"]
            print(f"Score overall model ({split}) =  {acc:.3f}%")
            acc_global = self.config.metric.score_func(labels, preds_global_only, pos_class=1)
            print(f"Score global channel only ({split}) =  {acc_global:.3f}%")
            acc_gnn = self.config.metric.score_func(labels, preds_gnn_only, pos_class=1)
            print(f"Score GNN only ({split}) =  {acc_gnn:.3f}%\n")

            # Plot classifier decision boundary
            x_min, x_max = samples[:, 0].min() - 0.5, samples[:, 2].max() + 0.5
            x_vals = np.linspace(-1, x_max, 100)
            for c in range(w.shape[0]):
                w_c = w[c]
                b_c = b[c]
                y_vals = -(w_c[2] * x_vals + b_c) / w_c[0]
                axs[int(i/n_row)].plot(x_vals, y_vals, color='black', label=f'Dec. Bound.', alpha=0.6)

            axs[0].set_xlabel('num uncolored nodes', fontsize=13)
            axs[0].set_ylabel('num red nodes', fontsize=13)

        plt.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        path = f'GOOD/kernel/pipelines/plots/global_explanations/{self.config.dataset.dataset_name}_{self.config.dataset.domain}/{self.config.ood_dirname}/'
        if not os.path.exists(path):
            os.makedirs(path)

        path += f"{self.config.load_split}_{self.config.util_model_dirname}_{self.config.random_seed}"
        plt.savefig(path + ".png")
        print("\n Saved plot ", path, "\n")
        plt.close()

    @torch.no_grad()
    def generate_explanation_examples(
        self,
        ratios,
        split: str,
        metric: str,
        edge_scores,
        graphs,
        labels,
        avg_graph_size,
    ):
        print(f"\n\n")
        print("-"*50)
        print(f"\n\n#D#Plotting examples of explanations for {metric.upper()} over {split} across ratios (random_expl={self.config.random_expl})")
        reset_random_seed(self.config)
        self.model.eval()   

        ratios = [1.0]
        for ratio in ratios:
            reset_random_seed(self.config)
            print(f"\n\nratio={ratio}\n\n")            

            eval_samples, belonging, reference = [], [], []
            labels_ori = []

            pbar = tqdm(range(len(edge_scores)), desc=f'Creating Intervent. distrib.', total=len(edge_scores), **pbar_setting)
            for i in pbar:
                if graphs[i].edge_index.shape[1] <= 6:
                    continue                
                eval_samples.append(graphs[i])
                reference.append(len(eval_samples) - 1)
                belonging.append(-1)
                labels_ori.append(labels[i])
            
            int_dataset = CustomDataset("", eval_samples, belonging)

            if self.config.model.model_name == "GSATGIN":
                if self.config.dataset.dataset_name == "TopoFeature":
                    thrs = 0.8
                if self.config.dataset.dataset_name == "AIDSC1":
                    thrs = 0.7
                if self.config.dataset.dataset_name == "AIDS":
                    thrs = 0.8
                if self.config.dataset.dataset_name == "BAColor":
                    thrs = 0.7
            elif self.config.model.model_name == "SMGNNGIN":
                if self.config.dataset.dataset_name == "TopoFeature":
                    if self.config.global_side_channel == "simple_concept2temperature":
                        thrs = 0.20
                if self.config.dataset.dataset_name == "BAColor":
                    thrs = 0.2

            # PLOT EXAMPLES OF EXPLANATIONS (ORIGINAL CLEAN SAMPLES)
            print(len(edge_scores), len(graphs), len(int_dataset), self.config.expval_budget, avg_graph_size)
            for i in range(25):
                if i > 25:
                    break
                data = int_dataset[reference[i]]
                g = to_networkx(data, node_attrs=["x"], to_undirected=True)
                xai_utils.mark_edges(g, data.edge_index, torch.tensor([[]]), inv_edge_w=edge_scores[i])
                xai_utils.draw_colored(
                    self.config,
                    g,
                    subfolder=f"plots_of_explanation_examples/{self.config.ood_dirname}/{self.config.dataset.dataset_name}_{self.config.dataset.domain}",
                    name=f"graph_{reference[i]}",
                    thrs=thrs,
                    title=f"Idx: {i} Class {labels_ori[reference[i]].long().item()}",
                    with_labels=False,
                    figsize=(12,10) if "AIDS" in self.config.dataset.dataset_name else (6.4, 4.8)
                )
                print(f"graph_{reference[i]} is of class {labels_ori[reference[i]]}")
        return 
    

    @torch.no_grad()
    def get_node_explanations(self):
        self.model.eval()

        splits = ["id_val"]
        ret = {
            split: {
                "scores": [],
                "samples": [],
                "pred": []
            } for split in splits
        }
                
        for i, split in enumerate(splits):
            dataset = self.get_local_dataset(split)

            loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=2)
            for data in loader:
                data: Batch = data.to(self.config.device)   
                edge_scores = self.model.get_subgraph(
                    data=data,
                    edge_weight=None,
                    ood_algorithm=self.ood_algorithm,
                    do_relabel=False,
                    return_attn=False,
                    ratio=None
                )

                for j, g in enumerate(data.to_data_list()):
                    edge_expl = edge_scores[data.batch[data.edge_index[0]] == j].detach().cpu()
                    ret[split]["scores"].append(edge_expl)
                    ret[split]["samples"].append(g)
        return ret
    

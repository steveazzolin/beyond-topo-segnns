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
                
                # Comment for FAITH (TODO: fix this)
                if hasattr(G, "edge_gt"): # added for stability of detector analysis
                    data.edge_gt = G.edge_gt
                elif add_fake_edge_gt:
                    data.edge_gt = torch.zeros((data.edge_index.shape[1]), dtype=torch.long, device=data.edge_index.device)
                # if hasattr(G, "node_gt"): # added for stability of detector analysis
                #     data.node_gt = G.node_gt
                # if hasattr(G, "causal_mask"): # added for stability of detector analysis
                #     data.causal_mask = G.causal_mask
                # if hasattr(G, "edge_attr"):
                #     data.edge_attr = G.edge_attr
                # if hasattr(G, "num_edge_removed"): # added for stability of detector analysis
                #     data.num_edge_removed = torch.tensor(G.num_edge_removed, dtype=torch.long)
                # else:
                #     data.num_edge_removed = torch.tensor(0, dtype=torch.long)

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

        if self.config.global_side_channel and self.config.dataset.dataset_name != "BAColor" and epoch < 20:
            # Little pre-train of individual channels
            loss_global = self.ood_algorithm.loss_calculate(self.ood_algorithm.logit_global, targets, mask, node_norm, self.config, batch=data.batch)
            loss_global = loss_global.mean()
            loss_gnn    = self.ood_algorithm.loss_calculate(self.ood_algorithm.logit_gnn, targets, mask, node_norm, self.config, batch=data.batch)
            loss_gnn    = self.ood_algorithm.loss_postprocess(loss_gnn, data, mask, self.config, epoch)
            loss = loss_gnn + loss_global
        else:
            loss = self.ood_algorithm.loss_calculate(raw_pred, targets, mask, node_norm, self.config, batch=data.batch)
            loss = self.ood_algorithm.loss_postprocess(loss, data, mask, self.config, epoch)
        
        if False and self.config.global_side_channel in ("simple", "simple_filternode"):
            loss_clf_global_side_channel = self.ood_algorithm.loss_global_side_channel(data.y, mask, self.config)
            loss += 0.005 * loss_clf_global_side_channel
        else:
            loss_clf_global_side_channel = torch.tensor(0.)

        if self.config.global_side_channel and self.config.dataset.dataset_name == "AIDSC1":
            assert False
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

        if self.config.global_side_channel == "dt":
            print(f"Training Decision Tree")
            data_list = [self.loader["train"].dataset[i] for i in range(len(self.loader["train"].dataset))]
            batch = Batch.from_data_list(data_list)
            self.model.global_side_channel.fit(batch)
            for split in ["train", "id_val", "id_test", "test"]:
                data_list = [self.loader[split].dataset[i] for i in range(len(self.loader[split].dataset))]
                batch = Batch.from_data_list(data_list)
                acc = self.model.global_side_channel.score(batch)
                print(f"DT {split} Acc={acc}")

        print("Before training:")
        epoch_train_stat = self.evaluate('eval_train')
        id_val_stat = self.evaluate('id_val')
        id_test_stat = self.evaluate('id_test')

        if self.config.global_side_channel in ("simple_concept", "simple_concept2"):
            with torch.no_grad():
                print("Concept relevance scores:\n", self.model.combinator.classifier[0].alpha_norm.cpu().numpy())
                # print("Gamma difference: \n", self.model.combinator.classifier[0].gamma.cpu().diff().item())

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
                    # "diff_concept_gamma": self.model.combinator.classifier[0].gamma.cpu().diff().item() 
                    #                                 if self.config.global_side_channel == "simple_concept" else np.nan
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

                # Parameter for DANN
                p = (index / len(self.loader['train']) + epoch) / self.config.train.max_epoch
                self.config.train.alpha = 2. / (1. + np.exp(-10 * p)) - 1

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
                compute_wiou=(self.config.dataset.dataset_name == "TopoFeature" or self.config.dataset.dataset_name == "SimpleMotif" or self.config.dataset.dataset_name == "GOODMotif") 
                                and 
                             self.config.model.model_name != "GIN"
            )
            id_val_stat = self.evaluate('id_val')
            id_test_stat = self.evaluate('id_test')

            if self.config.global_side_channel:
                # preds_gnn_only = torch.cat(preds_gnn_only, dim=0).cpu()
                # preds_global_only = torch.cat(preds_global_only, dim=0).cpu()
                # labels = torch.cat(labels, dim=0).cpu()
                # gnn_only_score = self.config.metric.score_func(labels, preds_gnn_only)
                # global_only_score = self.config.metric.score_func(labels, preds_global_only)
                # print(f"GNN only Acc: {gnn_only_score:.3f}")
                # print(f"Global only Acc: {global_only_score:.3f}")
                
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
                        # Print concept attention scores
                        print("Concept relevance scores:\n", self.model.combinator.classifier[0].alpha_norm.cpu().numpy())
                        # print("Gamma difference: \n", self.model.combinator.classifier[0].gamma.cpu().diff().item())

                if self.config.dataset.dataset_name in ("BAColor", "TopoFeature"):
                    w = self.model.global_side_channel.classifier.classifier[0].weight.detach().cpu().numpy()
                    b = self.model.global_side_channel.classifier.classifier[0].bias.detach().cpu().numpy()
                    print(f"\nWeight vector of global side channel:\nW: {w} b:{b}")
            
            if self.config.dataset.shift_type == "no_shift":
                val_stat = id_val_stat
                test_stat = id_test_stat
            else:
                # val_stat = self.evaluate('val')
                # test_stat = self.evaluate('test')
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
                    # "diff_concept_gamma": self.model.combinator.classifier[0].gamma.cpu().diff().item() 
                    #                                 if self.config.global_side_channel == "simple_concept" else np.nan,
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
        path = f'GOOD/kernel/pipelines/plots/attn_distrib/{self.config.load_split}_{self.config.dataset.dataset_name}_{self.config.dataset.domain}_{self.config.util_model_dirname}_{self.config.random_seed}/'
        if not os.path.exists(path):
            os.makedirs(path)

        if attn_distrib != []:
            arrange_attn_distrib = []
            for l in range(len(attn_distrib[0])):
                arrange_attn_distrib.append([])
                for i in range(len(attn_distrib)):
                    arrange_attn_distrib[l].extend(attn_distrib[i][l])
            
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
        plt.hist(data, bins=100, density=density, log=log)
        plt.xlim(0.0,1.1)
        plt.title(f"distrib. edge_scores (min={round(min(data), 2)}, max={round(max(data), 2)})")
        plt.savefig(path + name)
        plt.close()
    

    @torch.no_grad()
    def get_subragphs_ratio(self, graphs, ratio, edge_scores, is_weight=False):
        """
            Cut graphs based on TopK or value thresholding strategy.
            If 'is_weight==False', use Top-'ratio'.
            Otherwise, use a thresholding with value 'ratio'
        """
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
        

        spu_subgraphs, causal_subgraphs, expl_accs, causal_masks = [], [], [], []
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
        
        return causal_subgraphs, spu_subgraphs, expl_accs, causal_masks
    
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
        if metric == "fidm" or (metric == "suff" and intervention_distrib == "model_dependent" and causal is None):
            return xai_utils.sample_edges(graph, "spu", self.config.fidelity_alpha_2, spu)
        elif metric in ("nec", "nec++", "fidp"):
            if metric == "nec++":
                alpha = max(self.config.nec_alpha_1 - 0.1 * (j // 3), 0.1)
            else:
                alpha = self.config.nec_alpha_1
            return xai_utils.sample_edges(graph, alpha, deconfounded=True, edge_index_to_remove=causal)
            # return xai_utils.sample_edges_tensorized(graph, k=1, edge_index_to_remove=causal, sampling_type="deconfounded")
        elif metric == "suff" and intervention_distrib == "bank":
            assert False
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
            G_t = graph.copy()
            # xai_utils.mark_edges(G_t, causal, spu)
            G_t_filt = xai_utils.remove_from_graph(G_t, edge_index_to_remove=causal)
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
        if "weight" in self.config.log_id or "class1" in self.config.log_id:
            if self.config.numsamples_budget < len(dataset):
                return train_test_split(
                    np.arange(len(dataset))[dataset.y.reshape(-1) == 1],
                    train_size=min(self.config.numsamples_budget, len(dataset)), # / len(dataset)
                    random_state=42,
                    shuffle=True,
                )[0]
            else:
                return np.arange(len(dataset))[dataset.y.reshape(-1) == 1] # TODO: remove this
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
                    # max_val = scatter_max(edge_score.cpu(), index=data.batch[data.edge_index[0]].cpu())[0]
                    # edge_score = -edge_score.cpu()
                    # edge_score = edge_score + max_val[data.batch[data.edge_index[0]].cpu()]
                    # edge_score = edge_score.to(self.config.device)
                # attn_distrib.append(self.model.attn_distrib)
                for j, g in enumerate(data.to_data_list()):
                    g.ori_x = data.ori_x[data.batch == j]
                    edge_scores[SPLIT].append(edge_score[data.batch[data.edge_index[0]] == j].detach().cpu())
                    graphs[SPLIT].append(g.detach().cpu())
                labels[SPLIT].extend(data.y.detach().cpu().numpy().tolist())
            labels[SPLIT] = torch.tensor(labels[SPLIT])
            avg_graph_size[SPLIT] = np.mean([g.edge_index.shape[1] for g in graphs[SPLIT]])

            if convert_to_nx:
                # TODO: remove edge_gt for real-world experiments (added for stability analysis)
                # graphs_nx[SPLIT] = [to_networkx(g, node_attrs=["ori_x"], edge_attrs=["edge_gt"]) for g in graphs[SPLIT]]
                # graphs_nx[SPLIT] = [to_networkx(g, node_attrs=["ori_x"]) for g in graphs[SPLIT]]
                
                edge_attr_tokeep = ([] if g.edge_attr is None else ["edge_attr"]) + ([] if g.edge_gt is None else ["edge_gt"])
                graphs_nx[SPLIT] = [to_networkx(g, node_attrs=["ori_x"], edge_attrs=edge_attr_tokeep if edge_attr_tokeep != [] else None) for g in graphs[SPLIT]]
            else:
                graphs_nx[SPLIT] = list()

            # if hasattr(graphs[SPLIT][0], "edge_gt") and log:
            #     num_gt_edges = torch.tensor([data.edge_gt.sum() for data in graphs[SPLIT]])
            #     num_all_edges = torch.tensor([data.edge_index.shape[1] for data in graphs[SPLIT]])
            #     print(f"\nGold ratio ({SPLIT}) = ", torch.mean(num_gt_edges / num_all_edges), "+-", torch.std(num_gt_edges / num_all_edges))

            for ratio in ratios:
                reset_random_seed(self.config)
                causal_subgraphs_r[SPLIT][ratio], spu_subgraphs_r[SPLIT][ratio], expl_accs_r[SPLIT][ratio], causal_masks[SPLIT][ratio] = self.get_subragphs_ratio(graphs[SPLIT], ratio, edge_scores[SPLIT], is_weight=is_weight)
                if log:
                    if self.config.dataset.dataset_name == "SimpleMotif":
                        mask = labels[SPLIT] == 1
                    else:
                        mask = torch.ones_like(labels[SPLIT], dtype=torch.bool)
                    print(f"F1 for r={ratio} = {np.mean([e[1] for e in expl_accs_r[SPLIT][ratio]]):.3f}")
                    print(f"WIoU for r={ratio} = {np.mean([e[0] for e in expl_accs_r[SPLIT][ratio]]):.3f}")
        return (edge_scores, graphs, graphs_nx, labels, \
                avg_graph_size, causal_subgraphs_r, spu_subgraphs_r,  expl_accs_r, causal_masks)


    @torch.no_grad()
    def compute_intervention_bank(self, ratios, splits, graphs_nx, causal_subgraphs_r):
        interventional_bank = defaultdict(list)
        for SPLIT in splits:
            bank_idxs, _ = train_test_split(
                np.arange(len(graphs_nx[SPLIT])),
                train_size=min(500, len(graphs_nx[SPLIT])-1),
                random_state=42,
                shuffle=True,
                stratify=None
            )
            for ratio in ratios:
                if ratio == 1.0:
                    continue
                for bank_idx in bank_idxs:
                    int_graph = graphs_nx[SPLIT][bank_idx].copy()
                    int_graph_filt = xai_utils.remove_from_graph(int_graph, edge_index_to_remove=causal_subgraphs_r[SPLIT][ratio][bank_idx])
                    nx.set_node_attributes(int_graph_filt, name="frontier", values=False) # check if needed

                    if not (int_graph_filt is None or len(int_graph_filt.edges()) == 0 or len(int_graph_filt) == 0):
                        interventional_bank[ratio].append(int_graph_filt)
        for ratio in ratios:
            if ratio == 1.0:
                continue
            print(f"Size bank for r={ratio} -> {len(interventional_bank[ratio])}")
            assert len(interventional_bank[ratio]) > 100
        # intervent_bank = None
        # features_bank = None
        # if intervention_distrib == "bank":
        #     if torch_geometric.__version__ == "2.4.0": 
        #         features_bank = dataset.x.unique(dim=0).cpu()
        #     else:
        #         features_bank = dataset.data.x.unique(dim=0).cpu()
        #     print(f"Shape of feature bank = {features_bank.shape}")
        #     print(f"Creating interventional bank with {self.config.expval_budget} elements")
        #     intervent_bank = []
        #     max_g_size = max([d.num_nodes for d in dataset])
        #     for i in range(self.config.expval_budget):
        #         I = nx.DiGraph(nx.barabasi_albert_graph(random.randint(5, max(int(max_g_size/2), 8)), 1), seed=42) #BA1 -> nx.barabasi_albert_graph(randint(5, max(len(G), 8)), randint(1, 3))
        #         nx.set_edge_attributes(I, name="origin", values="BA")
        #         if "motif" in self.config.dataset.dataset_name.lower():
        #             nx.set_node_attributes(I, name="ori_x", values=1.0)
        #         else:
        #             nx.set_node_attributes(I, name="ori_x", values=features_bank[random.randint(0, features_bank.shape[0]-1)].tolist())
        #         intervent_bank.append(I)  
        return interventional_bank


    @torch.no_grad()
    def compute_metric_ratio(
        self,
        ratios,
        split: str,
        metric: str,
        edge_scores,
        graphs,
        graphs_nx,
        labels,
        avg_graph_size,
        causal_subgraphs_r,
        spu_subgraphs_r,
        expl_accs_r,
        causal_masks_r,
        intervention_bank,
        intervention_distrib:str = "model_dependent",
        debug=False,
    ):
        assert metric in ["suff", "fidm", "nec", "nec++", "fidp", "suff++", "suff_simple"]
        assert intervention_distrib == "model_dependent"

        if "CIGA" in self.config.model.model_name:
            is_ratio = True
            weights = [self.model.att_net.ratio]
        else:
            is_ratio = True
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
            preds_ori, labels_ori, expl_acc_ori = [], [], []
            effective_ratio = [causal_subgraphs_r[ratio][i].shape[1] / (causal_subgraphs_r[ratio][i].shape[1] + spu_subgraphs_r[ratio][i].shape[1] + 1e-5) for i in range(len(spu_subgraphs_r[ratio]))]
            empty_idx = set()

            pbar = tqdm(range(len(edge_scores)), desc=f'Creating Intervent. distrib.', total=len(edge_scores), **pbar_setting)
            for i in pbar:
                if graphs[i].edge_index.shape[1] <= 6:
                    continue                
                if metric in ("suff", "suff++") and intervention_distrib == "model_dependent":
                    G = graphs_nx[i].copy()
                    G_filt = xai_utils.remove_from_graph(G, edge_index_to_remove=spu_subgraphs_r[ratio][i])
                    num_elem = xai_utils.mark_frontier(G, G_filt)
                    if len(G_filt) == 0 or num_elem == 0:
                        continue
                    # G = G_filt # P(Y|G) vs P(Y|R)

                # i = 100
                # tmp = graphs_nx[i].copy()
                # xai_utils.mark_edges(tmp, causal_subgraphs_r[ratio][i], spu_subgraphs_r[ratio][i])
                # xai_utils.draw(self.config, tmp, subfolder="plots_of_suff_scores", name=f"graph_{i}")
                # exit("SA")

                if metric in ("fidm", "fidp", "nec", "nec++") or len(empty_idx) == len(graphs) or intervention_distrib in ("fixed", "bank"):
                    if metric in ("suff", "suff++", "suff_simple") and intervention_distrib in ("fixed", "bank") and i == 0:
                        print(f"Using {intervention_distrib} interventional distribution")
                    elif metric in ("suff", "suff++", "suff_simple") and intervention_distrib == "model_dependent":
                        pass

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
                        expl_acc_ori.append(expl_accs_r[ratio][i])
                        belonging.extend([i] * len(intervened_graphs))
                        eval_samples.extend(intervened_graphs)

                    # for m in range(self.config.expval_budget): 
                    #     G_c = xai_utils.sample_edges_tensorized(
                    #         graphs[i],
                    #         nec_number_samples=self.config.nec_number_samples,
                    #         nec_alpha_1=self.config.nec_alpha_1,
                    #         avg_graph_size=avg_graph_size,
                    #         edge_index_to_remove=causal_masks_r[ratio][i],
                    #         sampling_type="bernoulli" if metric in ("fidm", "fidp") else self.config.samplingtype
                    #     )
                    #     belonging.append(i)
                    #     eval_samples.append(G_c)
                elif metric == "suff" or metric == "suff++" or metric == "suff_simple":
                    if ratio == 1.0:
                        eval_samples.extend([graphs[i]]*self.config.expval_budget)
                        belonging.extend([i]*self.config.expval_budget)
                    else:
                        z, c = -1, 0
                        idxs = np.random.permutation(np.arange(len(labels))) #pick random from every class
                        budget = self.config.expval_budget
                        
                        if metric == "suff++":
                            budget = budget // 2
                        if metric == "suff_simple":
                            budget = 0 # skip interventions and just pick subsamples

                        while c < budget:
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
                                causal_subgraphs_r[ratio][j],
                                spu_subgraphs_r[ratio][j],
                                G_filt,
                                debug,
                                (i, j, c),
                                feature_intervention=False,
                                feature_bank=None
                            )
                            if G_union is None:
                                continue
                            eval_samples.append(G_union)
                            belonging.append(i)
                            c += 1

                        # for k in range(c, self.config.expval_budget): # if not enough interventions, pad with sub-sampling
                        #     G_c = xai_utils.sample_edges_tensorized(
                        #         graphs[i],
                        #         nec_number_samples=self.config.nec_number_samples,
                        #         nec_alpha_1=self.config.nec_alpha_1,
                        #         avg_graph_size=avg_graph_size,
                        #         edge_index_to_remove=~causal_masks_r[ratio][i],
                        #         sampling_type="bernoulli" if metric in ("fidm", "fidp") else self.config.samplingtype
                        #     )
                        #     belonging.append(i)
                        #     eval_samples.append(G_c)
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
                            expl_acc_ori.append(expl_accs_r[ratio][i])
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
            
            # # Inspect edge_scores of intervened edges
            # self.debug_edge_scores(int_dataset, reference, ratio)            
            # Compute new prediction and evaluate KL
            int_dataset = CustomDataset("", eval_samples, belonging)

            loader = DataLoader(int_dataset, batch_size=256, shuffle=False)
            if self.config.mask:
                print("Computing with masking")
                preds_eval, belonging = self.evaluate_graphs(loader, log=False if "fid" in metric else True, weight=ratio, is_ratio=is_ratio, eval_kl=True)
            else:
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
                    if not "fid" in metric:
                        preds_ori_ori = preds_ori_ori.exp() # undo the log
                        preds_eval = preds_eval.exp()
                    preds_ori_ori = preds_ori_ori.round().reshape(-1)
                    preds_eval = preds_eval.round().reshape(-1)
                acc = (labels_ori_ori == preds_ori_ori.argmax(-1)).sum() / (preds_ori_ori.shape[0])
                acc_int = (labels_ori == preds_eval.argmax(-1)).sum() / preds_eval.shape[0]

            acc_ints.append(acc_int.item())
            print(f"\nModel {dataset_metric} of binarized graphs for r={ratio} = ", round(acc.item(), 3))
            print(f"Model XAI F1 of binarized graphs for r={ratio} = ", np.mean([e[1] for e in expl_accs_r[ratio]]))
            print(f"Model XAI WIoU of binarized graphs for r={ratio} = ", np.mean([e[0] for e in expl_accs_r[ratio]]))
            print(f"len(reference) = {len(reference)}")
            print(f"Effective ratio: {np.mean(effective_ratio):.3f} +- {np.std(effective_ratio):.3f}")
            if preds_eval.shape[0] > 0:
                print(f"Model {dataset_metric} over intervened graphs for r={ratio} = ", round(acc_int.item(), 3))
                for c in labels_ori_ori.unique().numpy().tolist():
                    print(f"{metric.upper()} for r={ratio} class {c} = {scores[c][-1]} +- {aggr['KL'].std():.3f} (in-sample avg dev_std = {(aggr_std**2).mean().sqrt():.3f})")
                    del scores[c]
                for m in aggr.keys():
                    print(f"{metric.upper()} for r={ratio} all {m} = {scores[f'all_{m}'][-1]} +- {aggr[f'{m}'].std():.3f} (in-sample avg dev_std = {(aggr_std**2).mean().sqrt():.3f})")
        return scores, acc_ints, results


    def normalize_belonging(self, belonging):
        #TODO: make more efficient
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

        if metric in ("suff", "suff++", "nec", "nec++", "suff_simple") and preds_eval.shape[0] > 0:
            div_kl = torch.nn.KLDivLoss(reduction="none", log_target=True)(preds_ori, preds_eval).sum(-1)
            div_l1 = torch.abs(preds_ori.exp() - preds_eval.exp()).sum(-1)

            # results[ratio] = div_l1.numpy().tolist()
            if metric in ("suff", "suff++", "suff_simple"):
                ret["KL"] = torch.exp(-scatter_mean(div_kl, belonging, dim=0)) # on paper
                ret["L1"] = torch.exp(-scatter_mean(div_l1, belonging, dim=0))
            elif metric in ("nec", "nec++"):
                ret["KL"] = 1 - torch.exp(-scatter_mean(div_kl, belonging, dim=0)) # on paper
                ret["L1"] = 1 - torch.exp(-scatter_mean(div_l1, belonging, dim=0))
            aggr_std = scatter_std(div_l1, belonging, dim=0)
        elif "fid" in metric and preds_eval.shape[0] > 0:
            if preds_ori_ori.shape[1] == 1:
                l1 = torch.abs(preds_eval.reshape(-1) - preds_ori.reshape(-1))
            else:
                l1 = torch.abs(preds_eval.gather(1, labels_ori.unsqueeze(1)) - preds_ori.gather(1, labels_ori.unsqueeze(1)))
            # results[ratio] = l1.numpy().tolist()
            ret["L1"] = scatter_mean(l1, belonging, dim=0)
            aggr_std = scatter_std(l1, belonging, dim=0)                    
        else:
            raise ValueError(metric)
        return ret, aggr_std

    @torch.no_grad()
    def compute_accuracy_binarizing(
        self,
        split: str,
        givenR,
        debug=False,
        metric_collector=None
    ):
        """
            Either computes the Accuracy of P(Y|R) or P(Y|G) under different weight/ratio binarizations
        """
        print(self.config.device)
        dataset = self.get_local_dataset(split)
        print(dataset)

        if "CIGA" in self.config.model.model_name:
            is_ratio = True
            weights = [self.model.att_net.ratio]
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
        if self.config.numsamples_budget == "all" or self.config.numsamples_budget >= len(dataset):
            idx = np.arange(len(dataset))        
        elif self.config.numsamples_budget < len(dataset):
            idx, _ = train_test_split(
                np.arange(len(dataset)),
                train_size=min(self.config.numsamples_budget, len(dataset)) / len(dataset),
                random_state=42,
                shuffle=True,
                stratify=dataset.y if torch_geometric.__version__ == "2.4.0" else dataset.data.y
            )

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


        # loader = DataLoader(dataset[idx], batch_size=256, shuffle=False)
        # pbar = tqdm(loader, desc=f'Extracting edge_scores {split.capitalize()} batched', total=len(loader), **pbar_setting)
        # graphs = []
        # edge_scores = []
        # labels = []
        # for data in pbar:
        #     data: Batch = data.to(self.config.device)            
        #     edge_score = self.model.get_subgraph(
        #                     data=data,
        #                     edge_weight=None,
        #                     ood_algorithm=self.ood_algorithm,
        #                     do_relabel=False,
        #                     return_attn=False,
        #                     ratio=None
        #             )   
        #     labels.extend(data.y.detach().cpu().numpy().tolist())
        #     for j, g in enumerate(data.to_data_list()):
        #         g.ori_x = data.ori_x[data.batch == j]
        #         g.ori_edge_index = data.ori_edge_index[:, data.batch[data.ori_edge_index[0]] == j]
        #         graphs.append(g.detach().cpu())
        #         edge_scores.append(edge_score[data.batch[data.edge_index[0]] == j].detach().cpu())
        # labels = torch.tensor(labels)
        # graphs_nx = [
        #     to_networkx(g, node_attrs=["ori_x"], edge_attrs=["edge_attr"] if not g.edge_attr is None else None) for g in graphs
        # ]
        # self.plot_attn_distrib([[]], edge_scores)

        (edge_scores, graphs, graphs_nx, labels, avg_graph_size, \
            causal_subgraphs_r, spu_subgraphs_r, expl_accs_r, causal_masks_r)  = self.compute_scores_and_graphs(
            weights,
            [split],
            convert_to_nx=True,
            log=False,
            extract_all=False
        )
        graphs_nx = graphs_nx[split]
        labels = labels[split]
        spu_subgraphs_r = spu_subgraphs_r[split]
        expl_accs_r = expl_accs_r[split]

        acc_scores, plaus_scores, wiou_scores = [], defaultdict(list), defaultdict(list)
        for weight in weights:
            print(f"\n\nr={weight}\n")
            eval_samples, labels_ori = [], []
            empty_graphs = 0
            
            # Select relevant subgraph based on ratio
            # if is_ratio:
            #     causal_subgraphs, spu_subgraphs, expl_accs = self.get_subragphs_ratio(graphs, weight, edge_scores)
            # else:
            #     causal_subgraphs, spu_subgraphs, expl_accs, causal_idxs, spu_idxs = self.get_subragphs_weight(graphs, weight, edge_scores)            
            effective_ratio = np.array([spu_subgraphs_r[weight][i].shape[1] / (spu_subgraphs_r[weight][i].shape[1] + spu_subgraphs_r[weight][i].shape[1] + 1e-5) for i in range(len(spu_subgraphs_r[weight]))])

            # Create interventional distribution     
            pbar = tqdm(range(len(idx)), desc=f'Int. distrib', total=len(idx), **pbar_setting)
            for i in pbar:                
                G = graphs_nx[i].copy()
                G_filt = G

                if len(G.edges()) == 0:
                    empty_graphs += 1
                    continue
                if givenR: # for P(Y|R)
                    G_filt = xai_utils.remove_from_graph(G, edge_index_to_remove=spu_subgraphs_r[weight][i])                    
                    if len(G_filt) == 0:
                        # xai_utils.mark_edges(G, causal_subgraphs[i], spu_subgraphs[i])
                        # xai_utils.draw(self.config, G, subfolder="plots_of_suff_scores", name=f"graph_{i}")
                        empty_graphs += 1
                        continue

                eval_samples.append(G_filt)
                labels_ori.append(labels[i])

            # Compute accuracy
            labels_ori = torch.tensor(labels_ori)
            if len(eval_samples) == 0:
                acc = 0.
            else:
                eval_set = CustomDataset("", eval_samples, torch.arange(len(eval_samples)))
                loader = DataLoader(eval_set, batch_size=256, shuffle=False, num_workers=2)
                if self.config.mask and weight <= 1.:
                    print("Computing with masking")
                    preds, _ = self.evaluate_graphs(loader, log=False, weight=None if givenR else weight, is_ratio=is_ratio)
                else:                    
                    preds, _ = self.evaluate_graphs(loader, log=False)

                if dataset.metric == "ROC-AUC":
                    acc = sk_roc_auc(labels_ori.long(), preds, multi_class='ovo')
                elif dataset.metric == "F1":
                    acc = f1_score(labels_ori.long(), preds.round().reshape(-1), average="binary", pos_label=dataset.minority_class)
                else:
                    if preds.shape[1] == 1:
                        preds = preds.round().reshape(-1)
                    else:
                        preds = preds.argmax(-1)     
                    acc = (labels_ori == preds).sum() / (preds.shape[0] + empty_graphs)
            acc_scores.append(acc.item())   
    
            print(f"\nModel Acc of binarized graphs for weight={weight} = {acc:.3f}")
            print("Num empty graphs = ", empty_graphs)
            print("Avg effective explanation ratio = ", np.mean(effective_ratio[effective_ratio > 0.01]))
            for c in labels_ori.unique():
                idx_class = np.arange(labels_ori.shape[0])[labels_ori == c]
                for q, (d, s) in enumerate(zip([wiou_scores, plaus_scores], ["WIoU", "F1"])):
                    d[c.item()].append(np.mean([e[q] for e in expl_accs_r[weight]]))
                    print(f"Model XAI {s} r={weight} class {c.item()} \t= {d[c.item()][-1]:.3f}")
            for q, (d, s) in enumerate(zip([wiou_scores, plaus_scores], ["WIoU", "F1"])):
                d["all"].append(np.mean([e[q] for e in expl_accs_r[weight]]))
                print(f"Model XAI {s} r={weight} for all classes \t= {d['all'][-1]:.3f}")
        metric_collector["acc"].append(acc_scores)
        metric_collector["plaus"].append(plaus_scores)
        metric_collector["wiou"].append(wiou_scores)
        return None

    def get_local_dataset(self, split, log=True):
        if torch_geometric.__version__ == "2.4.0" and log:
            print(self.loader[split].dataset)
            print(f"Data example from {split}: {self.loader[split].dataset.get(0)}")
            print(f"Label distribution from {split}: {self.loader[split].dataset.y.unique(return_counts=True)}")        

        dataset = self.loader[split].dataset
        
        if abs(dataset.y.unique(return_counts=True)[1].min() - dataset.y.unique(return_counts=True)[1].max()) > 1000:
            print(f"#D#Unbalanced warning for {self.config.dataset.dataset_name} ({split})")
        
        if "hiv" in self.config.dataset.dataset_name.lower() and str(self.config.numsamples_budget) != "all":
            balanced_idx, _ = RandomUnderSampler(random_state=42).fit_resample(np.arange(len(dataset)).reshape(-1,1), dataset.y)

            dataset = dataset[balanced_idx.reshape(-1)]
            print(f"Creating balanced dataset: {dataset.y.unique(return_counts=True)}")
        return dataset

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
    def evaluate(self, split: str, compute_suff=False, compute_wiou=False):
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

        # self.model.gnn.encoder.batch_norms.train()
        # for conv in self.model.gnn.encoder.convs:
        #     conv.mlp.train()
        # self.model.gnn.eval()

        loss_all = []
        mask_all = []
        pred_all = []
        target_all = []
        likelihoods_all = []
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

            # ------------- Likelihood data collection ------------------
            if raw_preds.shape[-1] > 1:
                probs = raw_preds.softmax(dim=1)
                likelihoods_all.append(probs.gather(1, targets.unsqueeze(1)))
            else:
                probs = raw_preds.sigmoid()
                likelihoods_all.append(torch.full_like(probs, fill_value=-1))            

            # ------------- Score data collection ------------------
            pred, target = eval_data_preprocess(data.y, raw_preds, mask, self.config)
            pred_all.append(pred)
            target_all.append(target)

            # ------------- WIOU ------------------
            if compute_wiou:
                wious_mask = torch.ones(data.batch.max() + 1, dtype=torch.bool)
                if self.config.dataset.dataset_name == "TopoFeature" or self.config.dataset.dataset_name == "SimpleMotif":
                    wious_mask[data.pattern == 0] = False # Mask out examples without the motif
                
                _, explanation = to_undirected(data.edge_index, self.ood_algorithm.edge_att.squeeze(-1), reduce="mean")
                wious_all.append(
                    xai_utils.expl_acc_super_fast(data, explanation, reference_intersection=data.edge_gt)[wious_mask].mean().item()
                )

        # ------- Loss calculate -------
        loss_all = torch.cat(loss_all)
        mask_all = torch.cat(mask_all)
        likelihoods_all = torch.cat(likelihoods_all)
        stat['loss'] = loss_all.sum() / mask_all.sum()
        stat['likelihood_avg'] = likelihoods_all.mean()
        stat['likelihood_prod'] = torch.prod(likelihoods_all)
        stat['likelihood_logprod'] = torch.sum(likelihoods_all.log())
        stat['wiou'] = np.mean(wious_all) if len(wious_all) > 0 else np.nan

        # --------------- Metric calculation including ROC_AUC, Accuracy, AP.  --------------------
        stat['score'] = eval_score(pred_all, target_all, self.config, self.loader[split].dataset.minority_class)

        print(
            f'{split.capitalize()} {self.config.metric.score_name}: {stat["score"]:.4f} \t' + 
            f'{split.capitalize()} Loss: {stat["loss"]:.4f} \t' + 
            (f'{split.capitalize()} WIoU: {stat["wiou"]:.3f} \t' if compute_wiou else '')
        )


        if was_training:
            self.model.train()

        return {
            'score': stat['score'],
            'loss': stat['loss'],
            'likelihood_avg': stat['likelihood_avg'],
            'likelihood_prod': stat['likelihood_prod'],
            'likelihood_logprod': stat['likelihood_logprod'],
            'wiou': stat['wiou']
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

    # @torch.no_grad()
    # def save_epoch(self, epoch: int, train_stat: dir, id_val_stat: dir, id_test_stat: dir, val_stat: dir,
    #                test_stat: dir, config: Union[CommonArgs, Munch]):
    #     r"""
    #     Training util for checkpoint saving.

    #     Args:
    #         epoch (int): epoch number
    #         train_stat (dir): train statistics
    #         id_val_stat (dir): in-domain validation statistics
    #         id_test_stat (dir): in-domain test statistics
    #         val_stat (dir): ood validation statistics
    #         test_stat (dir): ood test statistics
    #         config (Union[CommonArgs, Munch]): munchified dictionary of args (:obj:`config.ckpt_dir`, :obj:`config.dataset`, :obj:`config.train`, :obj:`config.model`, :obj:`config.metric`, :obj:`config.log_path`, :obj:`config.ood`)

    #     Returns:
    #         None

    #     """
    #     if epoch < config.train.pre_train:
    #         return

    #     if not (config.metric.best_stat['score'] is None or config.metric.lower_better * val_stat[
    #         'score'] < config.metric.lower_better *
    #             config.metric.best_stat['score']
    #             or (id_val_stat.get('score') and (
    #                     config.metric.id_best_stat['score'] is None or config.metric.lower_better * id_val_stat[
    #                 'score'] < config.metric.lower_better * config.metric.id_best_stat['score']))
    #             or epoch % config.train.save_gap == 0):
    #         return
        
    #     state_dict = self.model.state_dict() if config.ood.ood_alg != 'EERM' else self.model.gnn.state_dict()
    #     ckpt = {
    #         'state_dict': state_dict,
    #         'train_score': train_stat['score'],
    #         'train_loss': train_stat['loss'],
    #         'id_val_score': id_val_stat['score'],
    #         'id_val_loss': id_val_stat['loss'],
    #         'id_test_score': id_test_stat['score'],
    #         'id_test_loss': id_test_stat['loss'],
    #         'val_score': val_stat['score'],
    #         'val_loss': val_stat['loss'],
    #         'test_score': test_stat['score'],
    #         'test_loss': test_stat['loss'],
    #         'time': datetime.datetime.now().strftime('%b%d %Hh %M:%S'),
    #         'model': {
    #             'model name': f'{config.model.model_name} {config.model.model_level} layers',
    #             'dim_hidden': config.model.dim_hidden,
    #             'dim_ffn': config.model.dim_ffn,
    #             'global pooling': config.model.global_pool
    #         },
    #         'dataset': config.dataset.dataset_name,
    #         'train': {
    #             'weight_decay': config.train.weight_decay,
    #             'learning_rate': config.train.lr,
    #             'mile stone': config.train.mile_stones,
    #             'shift_type': config.dataset.shift_type,
    #             'Batch size': f'{config.train.train_bs}, {config.train.val_bs}, {config.train.test_bs}'
    #         },
    #         'OOD': {
    #             'OOD alg': config.ood.ood_alg,
    #             'OOD param': config.ood.ood_param,
    #             'number of environments': config.dataset.num_envs
    #         },
    #         'log file': config.log_path,
    #         'epoch': epoch,
    #         'max epoch': config.train.max_epoch
    #     }

    #     if not os.path.exists(config.ckpt_dir):
    #         os.makedirs(config.ckpt_dir)
    #         print(f'#W#Directory does not exists. Have built it automatically.\n'
    #               f'{os.path.abspath(config.ckpt_dir)}')

    #     saved_file = os.path.join(config.ckpt_dir, f'last.ckpt')
    #     torch.save(ckpt, saved_file)

    #     if not config.clean_save:
    #         # WARNING: Original code was saving every epoch and then if 'clean_save' delete checkpoint
    #         shutil.copy(saved_file, os.path.join(config.ckpt_dir, f'{epoch}.ckpt'))

    #     # --- In-Domain checkpoint ---
    #     # WARNING: Original code saves if 'score' is not None AND if valiation score is grater than best validation score
    #     # if id_val_stat.get('score') and (
    #     #         config.metric.id_best_stat['score'] is None or config.metric.lower_better * id_val_stat[
    #     #     'score'] < config.metric.lower_better * config.metric.id_best_stat['score']):
    #     if id_val_stat.get('loss') and (
    #             config.metric.id_best_stat['loss'] is None or id_val_stat['loss'] < config.metric.id_best_stat['loss']):            
    #         config.metric.id_best_stat['score'] = id_val_stat['score']
    #         config.metric.id_best_stat['loss'] = id_val_stat['loss']
    #         shutil.copy(saved_file, os.path.join(config.ckpt_dir, f'id_best.ckpt'))
    #         print('#IM#Saved a new best In-Domain checkpoint based on validation loss.')

    #     # --- Out-Of-Domain checkpoint ---
    #     # if id_val_stat.get('score'):
    #     #     if not (config.metric.lower_better * id_val_stat['score'] < config.metric.lower_better * val_stat['score']):
    #     #         return
    #     if val_stat.get('loss') and (
    #             config.metric.best_stat['loss'] is None or val_stat['loss'] < config.metric.best_stat['loss']):
            
    #         config.metric.best_stat['score'] = val_stat['score']
    #         config.metric.best_stat['loss'] = val_stat['loss']
    #         shutil.copy(saved_file, os.path.join(config.ckpt_dir, f'best.ckpt'))
    #         print('#IM#Saved a new best OOD checkpoint based on validation loss.')

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

        # --- Out-Of-Domain checkpoint ---
        # if id_val_stat.get('score'):
        #     if not (config.metric.lower_better * id_val_stat['score'] < config.metric.lower_better * val_stat['score']):
        #         return
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
        
        # self.model.gnn.encoder.batch_norms.train()
        # for conv in self.model.gnn.encoder.convs:
        #     conv.mlp.train() # Make BN of ConvLayer in train mode

        splits = ["train", "id_val", "id_test"] #, "test"
        n_row = 1
        fig, axs = plt.subplots(n_row, len(splits), figsize=(9,4))
        # plt.suptitle(f"{self.config.model.model_name[:4]}") # - {self.config.dataset.dataset_name} {self.config.dataset.domain}
        
        for i, split in enumerate(splits):            
            # acc = self.evaluate(split, compute_suff=False)["score"]
            # print(f"Acc ({split}) =  ({acc:.3f}%)")
            dataset = self.get_local_dataset(split)

            loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
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
            if "CIGA" in self.config.model.model_name:
                edge_scores = [np.abs(np.array(e)) for e in edge_scores]
                edge_scores = [(e - e.min()) / (e.max() - e.min() + 1e-7) for e in edge_scores if len(e) > 0]

            return edge_scores

            print(min(np.concatenate(edge_scores)), max(np.concatenate(edge_scores)), np.mean(np.concatenate(edge_scores)), np.std(np.concatenate(edge_scores)))
            axs[int(i/n_row)].hist(np.concatenate(edge_scores) + np.random.normal(0, 0.001, np.concatenate(edge_scores).shape), density=True, log=False, bins=100) #100 or np.linspace(-1, 1, 100)
            # a,b = np.unique(np.concatenate(edge_scores), return_counts=True)
            # print(a)
            # print(b)
            # axs[int(i/n_row)].bar(a,b)
            # axs[int(i/n_row)].set_title(f"{split}")
            # axs[int(i/n_row)].set_xlabel(f"explanation scores")
            # axs[int(i/n_row)].set_ylabel(f"density")
            axs[int(i/n_row)].set_xlim(-0.1, 1.1)  #axs[int(i/n_row)]
            axs[int(i/n_row)].set_ylim(0.0, 100)

            fig.supxlabel('explanation relevance scores', fontsize=13)
            fig.supylabel('density', fontsize=13)

            # means, stds = zip(*[(np.mean(e), np.std(e)) for e in edge_scores])
            # means, stds = self.smooth(np.array(means), k=5), np.array(stds)
            # axs[n_row - int(i/n_row) -1, n_row - int(i%n_row) -1].plot(np.arange(len(means)), means)
            # axs[n_row - int(i/n_row) -1, n_row - int(i%n_row) -1].plot(np.arange(len(effective_ratios)), self.smooth(effective_ratios, k=7), 'r', alpha=0.7)
            # axs[n_row - int(i/n_row) -1, n_row - int(i%n_row) -1].fill_between(np.arange(len(means)), means - stds, means + stds, alpha=0.5)
            # axs[n_row - int(i/n_row) -1, n_row - int(i%n_row) -1].set_title(f"Per sample attn. variability - {split}")
            # axs[n_row - int(i/n_row) -1, n_row - int(i%n_row) -1].set_ylim(0, 1.1)

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
            ax.tick_params(axis='y', labelsize=12)
            ax.tick_params(axis='x', labelsize=12)
            ax.set_xlim((0.0,1.0))
        fig.supxlabel('explanation relevance scores', fontsize=18)
        fig.supylabel('density', fontsize=18)
        
        path = f'GOOD/kernel/pipelines/plots/panels/{self.config.ood_dirname}/'
        if not os.path.exists(path):
            os.makedirs(path)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        path += f"{self.config.load_split}_{self.config.dataset.dataset_name}_{self.config.dataset.domain}_{self.config.util_model_dirname}_allseeds"
        plt.savefig(path + ".png")
        plt.savefig(f'GOOD/kernel/pipelines/plots/panels/pdfs/{self.config.load_split}_{self.config.dataset.dataset_name}_{self.config.dataset.domain}_{self.config.util_model_dirname}_allseeds.pdf')
        print("\n Saved plot ", path, "\n")
        plt.close()

    @torch.no_grad()
    def generate_global_explanation(self):
        self.model.eval()
        splits = ["train", "id_val", "id_test"] #, "test"
        n_row = 1
        fig, axs = plt.subplots(n_row, len(splits), figsize=(9,4))
        
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

            axs[int(i/n_row)].scatter(samples[preds_global_only == 0, 0], samples[preds_global_only == 0, 1], c="yellow", alpha=0.2, label="pred. class 0")
            axs[int(i/n_row)].scatter(samples[preds_global_only == 1, 0], samples[preds_global_only == 1, 1], c="violet", alpha=0.2, label="pred. class 1")
            axs[int(i/n_row)].scatter(samples[preds_global_only != labels, 0], samples[preds_global_only != labels, 1], c="red", alpha=0.8, marker="x")
            
            acc = self.evaluate(split, compute_suff=False)["score"]
            print(f"Score overall model ({split}) =  {acc:.3f}%")
            acc_global = self.config.metric.score_func(labels, preds_global_only, pos_class=1)
            print(f"Score global channel only ({split}) =  {acc_global:.3f}%")
            acc_gnn = self.config.metric.score_func(labels, preds_gnn_only, pos_class=1)
            print(f"Score GNN only ({split}) =  {acc_gnn:.3f}%\n")


            # TODO: Print general info also in plot_panel function
            # Add a simple rule and train a DT
            # Try BA2Color with DT and see if it uses just local explanations

            # Plot classifier decision boundary
            x_min, x_max = samples[:, 0].min() - 0.5, samples[:, 0].max() + 0.5
            x_vals = np.linspace(x_min, x_max, 100)
            for c in range(w.shape[0]):
                w_c = w[c]
                b_c = b[c]
                y_vals = -(w_c[0] * x_vals + b_c) / w_c[1]
                axs[int(i/n_row)].plot(x_vals, y_vals, color='black', label=f'Boundary {c}')

            fig.supxlabel('global side channel decision boundary', fontsize=13)

            # axs[n_row - int(i/n_row) -1, n_row - int(i%n_row) -1].fill_between(np.arange(len(means)), means - stds, means + stds, alpha=0.5)
            # axs[n_row - int(i/n_row) -1, n_row - int(i%n_row) -1].set_title(f"Per sample attn. variability - {split}")
            # axs[n_row - int(i/n_row) -1, n_row - int(i%n_row) -1].set_ylim(0, 1.1)

        plt.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        path = f'GOOD/kernel/pipelines/plots/global_explanations/{self.config.dataset.dataset_name}_{self.config.dataset.domain}/{self.config.ood_dirname}/'
        if not os.path.exists(path):
            os.makedirs(path)

        path += f"{self.config.load_split}_{self.config.util_model_dirname}_{self.config.random_seed}"
        plt.savefig(path + ".png")
        # plt.savefig(f'GOOD/kernel/pipelines/plots/panels/pdfs/{self.config.load_split}_{self.config.dataset.dataset_name}_{self.config.dataset.domain}_{self.config.util_model_dirname}_{self.config.random_seed}.pdf')
        print("\n Saved plot ", path, "\n")
        plt.close()

    def smooth(self, y, k):
        box = np.ones(k) / k
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    @torch.no_grad()
    def test_motif(self):
        from GOOD.utils.synthetic_data import synthetic_structsim
        
        self.model.eval()
        print()
        print(self.loader['id_val'].dataset.all_motifs)
        print(self.loader['id_val'].dataset[0])

        house, _ = synthetic_structsim.house(start=0)
        crane, _ = synthetic_structsim.crane(start=0)
        dircycle, _ = synthetic_structsim.dircycle(start=0)
        path, _ = synthetic_structsim.path(start=0, width=8)        

        house_pyg = from_networkx(house)
        house_pyg.x = torch.ones((house_pyg.num_nodes, 1), dtype=torch.float32)
        print(house_pyg)

        crane_pyg = from_networkx(crane)
        crane_pyg.x = torch.ones((crane_pyg.num_nodes, 1), dtype=torch.float32)

        dircycle_pyg = from_networkx(dircycle)
        dircycle_pyg.x = torch.ones((dircycle_pyg.num_nodes, 1), dtype=torch.float32)

        path_pyg = from_networkx(path)
        path_pyg.x = torch.ones((path_pyg.num_nodes, 1), dtype=torch.float32)

        data = Batch().from_data_list([house_pyg, dircycle_pyg, crane_pyg, path_pyg]).to(self.config.device)
        preds = self.model.probs(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm)

        print("Predictions of entire model")
        print(preds)

        print("Predictions of classifier")
        if "LECI" in self.config.model.model_name:
            lc_logits = self.model.lc_classifier(self.model.lc_gnn(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm))
        else:
            lc_logits = self.model(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm)[0]
        print(lc_logits.softmax(-1))

        for split in ["train", "id_val", "test"]:
            dataset = self.get_local_dataset(split, log=False)
            loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
            wious = []
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
                    score = (edge_score[data.batch[data.edge_index[0]] == j]).detach().cpu().numpy().tolist()

                    edge_gt = {(u.item(),v.item()): g.edge_gt[i] for i, (u,v) in enumerate(g.edge_index.T)} 
                    wiou, den = 0, 0
                    for i, (u,v) in enumerate((g.edge_index.T)):
                        u, v = u.item(), v.item()
                        if edge_gt[(u,v)]:
                            wiou += score[i]
                        den += score[i]
                    wious.append(round(wiou / den, 3))
            print(f"WIoU {split} = {np.mean(wious):.2f}")

            self.permute_attention_scores("id_val")

        print("\n"*3)

    @torch.no_grad()
    def permute_attention_scores(self, split):
        self.config.numsamples_budget = "all"

        self.model.eval()
        print(f"Trying to replace attention weigths for {split}:")
        dataset = self.get_local_dataset(split, log=False)

        if self.config.numsamples_budget != "all" and self.config.numsamples_budget < len(dataset):
            assert False
            idx, _ = train_test_split(
                    np.arange(len(dataset)),
                    train_size=min(self.config.numsamples_budget, len(dataset)) / len(dataset),
                    random_state=42,
                    shuffle=True,
                    stratify=dataset.y if torch_geometric.__version__ == "2.4.0" else dataset.data.y
            )
        else:
            idx = np.arange(len(dataset))

        dataset = dataset[idx]
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
        preds, ori_preds = [], []
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
            ori_out = self.model.predict_from_subgraph(
                edge_att=edge_score,
                data=data,
                edge_weight=None,
                ood_algorithm=self.ood_algorithm
            )
            ori_preds.extend(ori_out.cpu().numpy().tolist())
            
            # permute edge scores
            edge_score = edge_score[
                shuffle_node(torch.arange(edge_score.shape[0], device=edge_score.device), batch=data.batch[data.edge_index[0]])[1]
            ]
            # data.edge_index, edge_score = to_undirected(data.edge_index, edge_score, reduce="mean")

            # max_val = scatter_max(edge_score.cpu(), index=data.batch[data.edge_index[0]].cpu())[0]
            # edge_score = -edge_score.cpu()
            # edge_score = edge_score + max_val[data.batch[data.edge_index[0]].cpu()]
            # edge_score = edge_score.to(self.config.device)

            out = self.model.predict_from_subgraph(
                edge_att=edge_score,
                data=data,
                edge_weight=None,
                ood_algorithm=self.ood_algorithm
            )
            preds.extend(out.cpu().numpy().tolist())
        
        preds = torch.tensor(preds)
        ori_preds = torch.tensor(ori_preds)
        print(preds.shape, ori_preds.shape, dataset.y.shape)
        if dataset.metric == "ROC-AUC":
            acc_ori = sk_roc_auc(dataset.y.long().numpy(), ori_preds, multi_class='ovo')
            acc = sk_roc_auc(dataset.y.long().numpy(), preds, multi_class='ovo')
        elif dataset.metric == "F1":
            acc_ori = f1_score(dataset.y.long().numpy(), ori_preds.round().reshape(-1), average="binary", pos_label=dataset.minority_class)
            acc = f1_score(dataset.y.long().numpy(), preds.round().reshape(-1), average="binary", pos_label=dataset.minority_class)
        else:
            if preds.dtype == torch.float or preds.dtype == torch.double:
                preds = preds.round()
                ori_preds = ori_preds.round()
            preds = preds.reshape(-1)
            ori_preds = ori_preds.reshape(-1)
            acc_ori = accuracy_score(dataset.y.numpy(), ori_preds)
            acc = accuracy_score(dataset.y.numpy(), preds)

        print(f"{dataset.metric.upper()} original: {acc_ori:.3f}")
        print(f"{dataset.metric.upper()} permuted: {acc:.3f}")
        return acc_ori, acc

    def generate_plot_sampling_type(self, splits, ratios, sampling_alphas, graphs, graphs_nx, causal_subgraphs_r, causal_masks, avg_graph_size):
        def nec_kl(ori_pred, pred, belonging, labels):
            return 1 - torch.exp(-scatter_mean(torch.nn.KLDivLoss(reduction="none", log_target=True)(ori_pred, pred).sum(-1), belonging, dim=0))
        def nec_l1(ori_pred, pred, belonging, labels):
            return 1 - torch.exp(-scatter_mean(torch.abs(ori_pred.exp() - pred.exp()).sum(-1), belonging, dim=0))
        def fid_l1_div(ori_pred, pred, belonging, labels):
            return torch.abs(ori_pred.exp() - pred.exp()).sum(-1).mean()
        def fid_model(ori_pred, pred, belonging, labels):
            return torch.abs(ori_pred.exp().gather(1, ori_pred.argmax(-1).unsqueeze(1)) - pred.exp().gather(1, ori_pred.argmax(-1).unsqueeze(1))).mean()
        def fid_phenom(ori_pred, pred, belonging, labels):
            return torch.abs(ori_pred.exp().gather(1, labels.long().unsqueeze(1)) - pred.exp().gather(1, labels.long().unsqueeze(1))).mean()
        def ratio_pred_change(ori_pred, pred, belonging, labels):
            return sum(ori_pred.argmax(-1) != pred.argmax(-1)) / pred.shape[0]
        def predict_sample(graphs, ratio):
            eval_set = CustomDataset("", graphs, torch.arange(len(graphs)))
            loader = DataLoader(eval_set, batch_size=256, shuffle=False, num_workers=0)
            preds, belonging = self.evaluate_graphs(loader, log=True, weight=ratio, is_ratio=True, eval_kl=True)
            return preds
    
        self.model.eval()

        metrics, accs = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
        for SPLIT in splits:
            for ratio in ratios:
                ori_graphs = []
                G_sampled_rfid, G_sampled_deconf, G_sampled_fixed, G_sampled_deconf_R = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
                belonging, labels = [], torch.tensor([])
                labels_norep = torch.tensor([])
                for idx in range(len(graphs[SPLIT])):
                    # reset_random_seed(self.config)
                    # if graphs[SPLIT][idx].edge_index.shape[1] <= 3 or graphs[SPLIT][idx].num_nodes <= 3:
                    #     continue
                    labels_norep = torch.cat((labels_norep, graphs[SPLIT][idx].y), dim=0)
                    for j in range(self.config.expval_budget):
                        for alpha in sampling_alphas:
                            sample = xai_utils.sample_edges_tensorized(graphs[SPLIT][idx], nec_number_samples=None, nec_alpha_1=alpha, avg_graph_size=None, sampling_type="bernoulli", edge_index_to_remove=causal_masks[SPLIT][ratio][idx], force_undirected=True)
                            G_sampled_rfid[alpha].append(sample)
                        for alpha in sampling_alphas:
                            sample = xai_utils.sample_edges_tensorized(graphs[SPLIT][idx], nec_number_samples="prop_G_dataset", nec_alpha_1=alpha, avg_graph_size=avg_graph_size[SPLIT], sampling_type="deconfounded", edge_index_to_remove=causal_masks[SPLIT][ratio][idx], force_undirected=True)
                            G_sampled_deconf[alpha].append(sample)
                        for alpha in sampling_alphas:
                            sample = xai_utils.sample_edges_tensorized(graphs[SPLIT][idx], nec_number_samples="prop_R", nec_alpha_1=alpha, avg_graph_size=avg_graph_size[SPLIT], sampling_type="deconfounded", edge_index_to_remove=causal_masks[SPLIT][ratio][idx], force_undirected=True)
                            G_sampled_deconf_R[alpha].append(sample)
                        # for k in range(1, len(sampling_alphas)+1):
                        #     sample = xai_utils.sample_edges_tensorized(graphs[SPLIT][idx], nec_number_samples="alwaysK", nec_alpha_1=k, avg_graph_size=None, sampling_type="deconfounded", edge_index_to_remove=causal_masks[SPLIT][ratio][idx], force_undirected=True)
                        #     G_sampled_fixed[k].append(sample)

                        ori_graphs.append(graphs[SPLIT][idx])
                        belonging.append(idx)
                        labels = torch.cat((labels, graphs[SPLIT][idx].y), dim=0)

                labels = labels.view(-1)                
                belonging = torch.tensor(self.normalize_belonging(belonging), dtype=torch.long, device=labels.device)
                ori_pred = predict_sample(ori_graphs, ratio)
                divergences = {}                

                for k, alpha in enumerate(sampling_alphas):
                    k = k + 1 #for G_sampled_fixed
                    divergences[f"RFID_{alpha}"] = {}
                    divergences[f"DECONF_{alpha}"] = {}
                    divergences[f"DECONF_R_{alpha}"] = {}
                    # divergences[f"FIXED_{k}"] = {}
                    preds = predict_sample(G_sampled_rfid[alpha] + G_sampled_deconf[alpha] + G_sampled_deconf_R[alpha], ratio)
                    pred_rfid   = preds[:len(G_sampled_rfid[alpha])]
                    pred_deconf = preds[len(G_sampled_rfid[alpha]): len(G_sampled_rfid[alpha]) + len(G_sampled_deconf[alpha])]
                    pred_deconf_R = preds[len(G_sampled_rfid[alpha]) + len(G_sampled_deconf[alpha]):]
                    

                    # pred_fixed  = preds[len(G_sampled_rfid[alpha]) + len(G_sampled_deconf[alpha]):]
                    for metric_name, div_f in zip(["NEC L1"], [nec_kl, nec_l1]): #"FID L1 div", fid_l1_div
                        divergences[f"RFID_{alpha}"][metric_name]   = div_f(ori_pred, pred_rfid, belonging, labels).mean().item()
                        divergences[f"DECONF_{alpha}"][metric_name] = div_f(ori_pred, pred_deconf, belonging, labels).mean().item()
                        divergences[f"DECONF_R_{alpha}"][metric_name] = div_f(ori_pred, pred_deconf_R, belonging, labels).mean().item()
                        # divergences[f"FIXED_{k}"][metric_name]      = div_f(ori_pred, pred_fixed, belonging, labels).mean().item()
                        
                metrics[SPLIT][ratio] = divergences
                accs[SPLIT][ratio] = sum(labels == ori_pred.argmax(-1)) / ori_pred.shape[0]

                # pred_all = self.predict_sample(ori_graphs + [s[0] for s in sampled] + [s[1] for s in sampled] + [s[2] for s in sampled] + [s[3] for s in sampled] + [s[4] for s in sampled], tested_pipeline, ratio=ratio)
                # pred = pred_all[len(ori_graphs):len(ori_graphs) + len(sampled)]
                # pred_rfid = pred_all[len(ori_graphs) + len(sampled):len(ori_graphs) + 2*len(sampled)]
                # ori_preds[SPLIT][ratio] = ori_pred
                # preds[SPLIT][ratio] = pred
                # for metric_name, div_f in zip(["NEC KL", "NEC L1", "FID L1 div", "Model FID", "Phen. FID", "Change pred"], [nec_kl, nec_l1, fid_l1_div, fid_model, fid_phenom, ratio_pred_change]):
                #     divergences["alphaAvgG"][metric_name]        = div_f(ori_pred, pred, belonging, labels).mean().item()
        return metrics, accs


    @torch.no_grad()
    def compute_stability_detector_extended(
        self,
        seed,
        ratios,
        split: str,
        metric: str,
        edge_scores,
        graphs,
        graphs_nx,
        labels,
        avg_graph_size,
        causal_subgraphs_r,
        spu_subgraphs_r,
        expl_accs_r,
        causal_masks_r,
        intervention_bank,
        intervention_distrib:str = "model_dependent",
        debug=False,
    ):
        assert metric in ["suff", "suff++", "suff_simple"]
        assert intervention_distrib == "model_dependent"
        assert self.config.mask

        print(f"\n\n", "-"*50, f"\n\n#D#Computing stability detector extended under {metric.upper()} over {split} across ratios (random_expl={self.config.random_expl})")
        self.model.eval()   

        scores, results, acc_ints = defaultdict(list), {}, []
        for ratio in ratios:
            if ratio == 1.0:
                continue
            reset_random_seed(self.config)
            print(f"\n\nratio={ratio}\n\n")            

            eval_samples, belonging, reference = [], [], []
            preds_ori, labels_ori, expl_acc_ori = [], [], []

            pbar = tqdm(range(len(edge_scores)), desc=f'Creating Intervent. distrib.', total=len(edge_scores), **pbar_setting)
            startInterventionTime = datetime.datetime.now()
            for i in pbar:
                if graphs[i].edge_index.shape[1] <= 6:
                    continue                
                
                graphs[i].causal_mask = causal_masks_r[ratio][i]

                if hasattr(graphs[i], "edge_gt"):
                    edge_index_mask = ~graphs[i].causal_mask & ~graphs[i].edge_gt # intervene outside of explanation and outside of GT
                else:
                    edge_index_mask = ~graphs[i].causal_mask # intervene only outside of explanation

                intervened_graphs = xai_utils.sample_edges_tensorized_batched(
                    graphs[i],
                    nec_number_samples=self.config.nec_number_samples,
                    nec_alpha_1=self.config.nec_alpha_1,
                    avg_graph_size=avg_graph_size,
                    edge_index_to_remove=edge_index_mask,
                    sampling_type=self.config.samplingtype,
                    budget=self.config.expval_budget
                )

                ## This different way to obtain 'intervened_graphs' is calling a different function, which is removing isolated_nodes
                ## I found using remove_isolated_nodes cumbersome, as it creates problem with the correct assignment of node_gt
                ## Therefore I preferred to keep isolated nodes, as the goal of the analysis is to check the different in edge removals.
                # intervened_graphs = []
                # for k in range(0, 2): # if not enough interventions, pad with sub-sampling
                #     G_c = xai_utils.sample_edges_tensorized(
                #         graphs[i],
                #         nec_number_samples=self.config.nec_number_samples,
                #         nec_alpha_1=self.config.nec_alpha_1,
                #         avg_graph_size=avg_graph_size,
                #         edge_index_to_remove=~graphs[i].causal_mask & ~graphs[i].edge_gt, # intervene outside of explanation and outside of GT
                #         sampling_type=self.config.samplingtype,
                #     )
                #     intervened_graphs.append(G_c)

                # if i == 1:
                #     print(len(intervened_graphs))
                #     g0 = to_networkx(graphs[i], edge_attrs=["edge_gt", "causal_mask"], node_attrs=["node_gt"], to_undirected=True)
                #     g1 = to_networkx(intervened_graphs[0], edge_attrs=["edge_gt", "causal_mask"], node_attrs=["node_gt"], to_undirected=True)
                #     g2 = to_networkx(intervened_graphs[1], edge_attrs=["edge_gt", "causal_mask"], node_attrs=["node_gt"], to_undirected=True)
                #     print(intervened_graphs[0].num_nodes)
                #     print(intervened_graphs[0].edge_index.shape)
                #     print(graphs[i].causal_mask)
                #     print(intervened_graphs[0].causal_mask)
                #     print(intervened_graphs[1].causal_mask)
                #     print(graphs[i].edge_index[:, graphs[i].causal_mask == True])
                #     print(intervened_graphs[0].edge_index[:, intervened_graphs[0].causal_mask == True])
                #     print(intervened_graphs[1].edge_index[:, intervened_graphs[1].causal_mask == True])
                #     print(graphs[i].node_gt)
                #     print(intervened_graphs[1].node_gt)
                #     print(g2.nodes(data=True))
                #     # nx.set_edge_attributes(g0, "spu", "origin")
                #     # nx.set_edge_attributes(g1, "spu", "origin")
                #     # nx.set_edge_attributes(g2, "spu", "origin")
                #     nx.set_edge_attributes(g0, {(u,v): "inv" if val["causal_mask"] else "spu" for u,v,val in g0.edges(data=True)}, "origin")
                #     nx.set_edge_attributes(g1, {(u,v): "inv" if val["causal_mask"] else "spu" for u,v,val in g1.edges(data=True)}, "origin")
                #     nx.set_edge_attributes(g2, {(u,v): "inv" if val["causal_mask"] else "spu" for u,v,val in g2.edges(data=True)}, "origin")
                #     xai_utils.draw(self.config, g0, subfolder="plots_of_gt_stability_det", name=f"int_ori_{i}", title=50)
                #     xai_utils.draw(self.config, g1, subfolder="plots_of_gt_stability_det", name=f"int_{i}_1", title=50)
                #     xai_utils.draw(self.config, g2, subfolder="plots_of_gt_stability_det", name=f"int_{i}_2", title=50)
                #     exit()

                if intervened_graphs is not None:
                    eval_samples.append(graphs[i])
                    reference.append(len(eval_samples) - 1)
                    belonging.append(-1)
                    labels_ori.append(labels[i])

                    belonging.extend([i]*(self.config.expval_budget))
                    eval_samples.extend(intervened_graphs)
            
            # Compute new prediction and evaluate KL
            endInterventionTime = datetime.datetime.now()
            startPlausTime = datetime.datetime.now()

            int_dataset = CustomDataset("", eval_samples, belonging, add_fake_edge_gt=True, dataset_name=self.config.dataset.dataset_name)
            loader = DataLoader(int_dataset, batch_size=256, shuffle=False)

            # return explanations and make predictions
            plausibility_wious = torch.tensor([], device=self.config.device)
            stability_wious = torch.tensor([], device=self.config.device)
            stability_mcc = torch.tensor([], device=self.config.device)
            stability_f1 = torch.tensor([], device=self.config.device)
            belonging = torch.tensor([], device=self.config.device, dtype=int)

            for i, data in enumerate(loader):
                data: Batch = data.to(self.config.device)
                explanations = self.model.get_subgraph(
                    data=data,
                    edge_weight=None,
                    ood_algorithm=self.ood_algorithm,
                    do_relabel=False,
                    return_attn=False,
                    ratio=None
                )
                
                if "CIGA" in self.config.model.model_name:
                    explanations = explanations.sigmoid() # normalize scores for computing WIOU

                # wrt GT
                plausibility_wious = torch.cat(
                    (plausibility_wious, xai_utils.expl_acc_super_fast(data, explanations, reference_intersection=data.edge_gt)),
                    dim=0
                )
                # wrt predicted explanation
                stability_wious = torch.cat(
                    (stability_wious, xai_utils.expl_acc_super_fast(data, explanations, reference_intersection=data.causal_mask)),
                    dim=0
                )
                stability_hard = xai_utils.explanation_stability_hard(data, explanations, ratio)
                stability_mcc = torch.cat(
                    (stability_mcc, stability_hard[0]),
                    dim=0
                )
                stability_f1 = torch.cat(
                    (stability_f1, stability_hard[1]),
                    dim=0
                )

                
                ## DEBUG: Check that F1 and MCC scores are meaningful by plotting some examples
                (causal_edge_index, _, _, causal_batch), \
                    (spu_edge_index, _, _), mask_batch = split_graph(
                        data,
                        explanations,
                        ratio,
                        return_batch=True,
                        compensate_edge_removal=data.num_edge_removed
                )
                pos = None
                for j, d_ori in enumerate(data.to_data_list()):
                    d = d_ori.clone()
                    if j > 8:
                        exit("SUUU")

                    # Make mask and causal_mask both undirected. Othewrise there could be a mismatch since split_graph()
                    # does not always return both directionalities for a certain edge, while for intervened graphs 
                    # causal_mask is forced to do so.
                    mask = mask_batch[data.batch[data.edge_index[0]] == j]                    
                    row, col = d.edge_index
                    undirected = d.edge_index[:, row < col]          
                    undirected_mask = mask[row < col]
                    mask = torch.cat((undirected_mask, undirected_mask), dim=0)
                    d.edge_index = torch.cat((undirected, undirected.flip(0)), dim=1)
                    d.causal_mask = torch.cat((d.causal_mask[row < col], d.causal_mask[row < col]), dim=0)

                    cumnum = torch.tensor([g.num_nodes for g in data.to_data_list()]).cumsum(0)
                    cumnum[-1] = 0
                    causal_subgraph = causal_edge_index[:, data.batch[causal_edge_index[0]] == j] - cumnum[j-1]
                    spu_subgraph = spu_edge_index[:, data.batch[spu_edge_index[0]] == j] - cumnum[j-1]

                    # g = to_networkx(d_ori, node_attrs=["node_gt"], to_undirected=True)
                    # xai_utils.mark_edges(g, d_ori.edge_index, d_ori.edge_index[:, d_ori.edge_gt == 1], inv_edge_w=explanations[data.batch[data.edge_index[0]] == j])
                    # pos_here = xai_utils.draw(self.config, g, subfolder="plots_of_gt_stability_det", name=f"expl_gt_{i}_{j}", pos=pos)
                    
                    g = to_networkx(d, to_undirected=True) # node_attrs=["node_gt"],
                    xai_utils.mark_edges(g, causal_subgraph, spu_subgraph)
                    pos_here = xai_utils.draw(self.config, g, subfolder="plots_of_gt_stability_det", name=f"expl_{i}_{j}", pos=pos)

                    d.mask = mask
                    pos_here = xai_utils.draw_antonio(self.config, d, pos=pos, subfolder="plots_of_gt_stability_det_antonio", name=f"expl_{i}_{j}")

                    # print(causal_subgraph.shape, d.causal_mask.sum(), mask.sum())
                    # print(torch.cat((d.edge_index.T, mask.unsqueeze(1)), dim=1))
                    # print()
                    # print(torch.cat((d.edge_index.T, d.causal_mask.unsqueeze(1)), dim=1))
                    # print("\n\n")

                    # Compute manually MCC on single instance
                    eps = 1e-6
                    input = d.mask
                    target = d.causal_mask
                    TP = sum((input & target).to(int))
                    TN = sum(((input == False) & (target == False)).to(int))
                    FP = sum(((input == True) & (target == False)).to(int))
                    FN = sum(((input == False) & (target == True)).to(int))
                    numerator = (TP * TN) - (FP * FN)
                    denominator = torch.sqrt(((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)))
                    mcc_batched = numerator / (denominator + eps)

                    # print("causal_mask graphs[0] = ", torch.argwhere(graphs[0].causal_mask==True).flatten())
                    # print(d.edge_index[:, torch.argwhere(graphs[0].causal_mask==True).flatten()])

                    # print("causal_mask = ", torch.argwhere(d.causal_mask==True).flatten())
                    # print(d.edge_index[:, torch.argwhere(d.causal_mask==True).flatten()])

                    # print("mask = ", torch.argwhere(mask==True).flatten())
                    # print(d.edge_index[:, torch.argwhere(mask==True).flatten()])

                    print(f"{j}: {plausibility_wious[j]:.3f} - {stability_wious[j]:.3f} - {stability_mcc[j]:.3f} ({mcc_batched:.3f}) - {stability_f1[j]:.3f}")
                    print("\n\n")

                    if j == 3:
                        exit("ANTO")


                    if j % self.config.expval_budget == 0:
                        pos = pos_here

                # # threshold explanation before feeding to classifier
                # (causal_edge_index, causal_edge_attr, edge_att), _ = split_graph(data, edge_score, ratio)
                # data.x, data.edge_index, data.batch, _ = relabel(data.x, causal_edge_index, data.batch)
                # if not data.edge_attr is None:
                #     data.edge_attr = causal_edge_attr

                # ori_out = self.model.predict_from_subgraph(
                #     edge_att=edge_att,
                #     data=data,
                #     edge_weight=None,
                #     ood_algorithm=self.ood_algorithm,
                #     log=True,
                #     eval_kl=True
                # )
                # preds_eval.extend(ori_out.detach().cpu().numpy().tolist())
                # belonging.extend(data.belonging.detach().cpu().numpy().tolist())
                belonging = torch.cat((belonging, data.belonging), dim=0)
            # preds_eval = torch.tensor(preds_eval)
            # belonging = torch.tensor(belonging, dtype=int)
            # just make predictions
            # preds_eval, belonging = self.evaluate_graphs(loader, log=False if "fid" in metric else True, weight=ratio, is_ratio=is_ratio, eval_kl=True)
            
            print("Interventions completed in ", endInterventionTime - startInterventionTime)
            print("Plausibility completed in ", datetime.datetime.now() - startPlausTime)
            
            mask = torch.ones(belonging.shape[0], dtype=bool, device=belonging.device)
            mask[reference] = False
            belonging = belonging[mask]            
            assert torch.all(belonging >= 0), f"{torch.all(belonging >= 0)}"

            # divide predictions and labels into original and intervened
            # preds_ori = preds_eval[reference]
            # preds_eval = preds_eval[mask]
            # labels_ori_ori = torch.tensor(labels_ori)
            # preds_ori_ori = preds_ori
            # preds_ori = preds_ori_ori.repeat_interleave(self.config.expval_budget, dim=0)
            # labels_ori = labels_ori_ori.repeat_interleave(self.config.expval_budget, dim=0)

            # divide metrics into original and intervened
            plausibility_wiou_ori = plausibility_wious[reference]
            plausibility_wiou_int = plausibility_wious[mask]
            stability_wiou_ori = stability_wious[reference]
            stability_wiou_int = stability_wious[mask]
            stability_mcc_ori = stability_mcc[reference]
            stability_mcc_int = stability_mcc[mask]
            stability_f1_ori  = stability_f1[reference]
            stability_f1_int  = stability_f1[mask]
           
            # compute and store metrics
            # aggr, aggr_std = self.get_aggregated_metric(metric, preds_ori, preds_eval, preds_ori_ori, labels_ori, belonging, results, ratio)            
            # scores[f"all_L1"].append(round(aggr["L1"].mean().item(), 3))
            scores[f"plausibility_wiou_original"].append(plausibility_wiou_ori.mean().item())
            scores[f"plausibility_wiou_perturbed"].append(scatter_mean(plausibility_wiou_int, belonging, dim=0).mean().item())
            scores[f"stability_wiou_original"].append(stability_wiou_ori.mean().item())
            scores[f"stability_wiou_perturbed"].append(scatter_mean(stability_wiou_int, belonging, dim=0).mean().item())
            scores[f"stability_mcc_original"].append(stability_mcc_ori.mean().item())
            scores[f"stability_mcc_perturbed"].append(stability_mcc_int.mean().item())
            scores[f"stability_f1_original"].append(stability_f1_ori.mean().item())
            scores[f"stability_f1_perturbed"].append(stability_f1_int.mean().item())
            # scores[f"wiou_original_std"].append(wiou_ori.std().item())
            # scores[f"wiou_perturbed_std"].append(scatter_mean(wiou_int, belonging, dim=0).std().item())

            # assert preds_ori_ori.shape[1] > 1, preds_ori_ori.shape
            # dataset_metric = self.loader["id_val"].dataset.metric
            # if dataset_metric == "ROC-AUC":
            #     if not "fid" in metric:
            #         preds_ori_ori = preds_ori_ori.exp() # undo the log
            #         preds_eval = preds_eval.exp()
            #     acc = sk_roc_auc(labels_ori_ori.long(), preds_ori_ori[:, 1], multi_class='ovo')
            #     acc_int = sk_roc_auc(labels_ori.long(), preds_eval[:, 1], multi_class='ovo')
            # elif dataset_metric == "F1":
            #     acc = f1_score(labels_ori_ori.long(), preds_ori_ori.argmax(-1), average="binary", pos_label=dataset.minority_class)
            #     acc_int = f1_score(labels_ori.long(), preds_eval.argmax(-1), average="binary", pos_label=dataset.minority_class)
            # else:
            #     if preds_ori_ori.shape[1] == 1:
            #         assert False
            #         if not "fid" in metric:
            #             preds_ori_ori = preds_ori_ori.exp() # undo the log
            #             preds_eval = preds_eval.exp()
            #         preds_ori_ori = preds_ori_ori.round().reshape(-1)
            #         preds_eval = preds_eval.round().reshape(-1)
            #     acc = (labels_ori_ori == preds_ori_ori.argmax(-1)).sum() / (preds_ori_ori.shape[0])
            #     acc_int = (labels_ori == preds_eval.argmax(-1)).sum() / preds_eval.shape[0]

            # acc_ints.append(acc_int)
            print(f"Original Plaus. XAI WIoU for r={ratio} = ", plausibility_wiou_ori.mean(), " +- ", plausibility_wiou_ori.std())
            print(f"Intervened Plaus. XAI WIoU for r={ratio} = ", scatter_mean(plausibility_wiou_int, belonging, dim=0).mean(), " +- ", scatter_mean(plausibility_wiou_int, belonging, dim=0).std(), f"({scatter_std(plausibility_wiou_int, belonging, dim=0).mean():.3f})")
            print(f"Original Plaus. XAI WIoU for r={ratio} = ", plausibility_wiou_ori.mean(), " +- ", plausibility_wiou_ori.std())
            print(f"Intervened Plaus. XAI WIoU for r={ratio} = ", scatter_mean(stability_wiou_int, belonging, dim=0).mean(), " +- ", scatter_mean(stability_wiou_int, belonging, dim=0).std(), f"({scatter_std(stability_wiou_int, belonging, dim=0).mean():.3f})")
            print(f"Original Stability MCC for r={ratio} = {stability_mcc_ori.mean().item():.3f} +- {stability_mcc_ori.std().item():.3f}")
            print(f"Intervened Stability MCC for r={ratio} = {stability_mcc_int.mean().item():.3f} +- {stability_mcc_int.std().item():.3f}")
            print(f"Original Stability F1 for r={ratio} = {stability_f1_ori.mean().item():.3f} +- {stability_f1_ori.std().item():.3f}")
            print(f"Intervened Stability F1 for r={ratio} = {stability_f1_int.mean().item():.3f} +- {stability_f1_int.std().item():.3f}")
            print(f"Num considered samples: {len(reference)}")
        return scores, acc_ints, results


    @torch.no_grad()
    def compute_stability_detector_rebuttal(
        self,
        seed,
        ratios,
        split: str,
        metric: str,
        edge_scores,
        graphs,
        graphs_nx,
        labels,
        avg_graph_size,
        causal_subgraphs_r,
        spu_subgraphs_r,
        expl_accs_r,
        causal_masks_r,
        intervention_bank,
        intervention_distrib:str = "model_dependent",
        debug=False,
    ):
        assert metric in ["suff", "nec", "nec++", "suff++", "suff_simple"]
        assert intervention_distrib == "model_dependent"
        assert self.config.mask

        print(f"\n\n")
        print("-"*50)
        print(f"\n\n#D#Computing stability detector under {metric.upper()} over {split} across ratios (random_expl={self.config.random_expl})")
        reset_random_seed(self.config)
        self.model.eval()   

        scores, results, acc_ints = defaultdict(list), {}, []
        for ratio in ratios:
            if ratio == 1.0:
                continue
            reset_random_seed(self.config)
            print(f"\n\nratio={ratio}\n\n")            

            eval_samples, belonging, reference = [], [], []
            preds_ori, labels_ori, expl_acc_ori = [], [], []
            empty_idx = set()

            pbar = tqdm(range(len(edge_scores)), desc=f'Creating Intervent. distrib.', total=len(edge_scores), **pbar_setting)
            for i in pbar:
                if graphs[i].edge_index.shape[1] <= 6:
                    continue                
                if metric in ("suff", "suff++") and intervention_distrib == "model_dependent":
                    G = graphs_nx[i].copy()
                    G_filt = xai_utils.remove_from_graph(G, edge_index_to_remove=spu_subgraphs_r[ratio][i])
                    num_elem = xai_utils.mark_frontier(G, G_filt)
                    if len(G_filt) == 0 or num_elem == 0:
                        continue
                
                eval_samples.append(graphs[i])
                reference.append(len(eval_samples) - 1)
                belonging.append(-1)
                labels_ori.append(labels[i])
                # expl_acc_ori.append(expl_accs_r[ratio][i])

                if metric in ("nec", "nec++") or len(empty_idx) == len(graphs) or intervention_distrib in ("fixed", "bank"):
                    assert False, "Computing NEC interventions"
                    if metric in ("suff", "suff++", "suff_simple") and intervention_distrib in ("fixed", "bank") and i == 0:
                        print(f"Using {intervention_distrib} interventional distribution")
                    elif metric in ("suff", "suff++", "suff_simple") and intervention_distrib == "model_dependent":
                        pass

                    for m in range(self.config.expval_budget):                        
                        G_c = xai_utils.sample_edges_tensorized(
                            graphs[i],
                            nec_number_samples=self.config.nec_number_samples,
                            nec_alpha_1=self.config.nec_alpha_1,
                            avg_graph_size=avg_graph_size,
                            edge_index_to_remove=causal_masks_r[ratio][i],
                            sampling_type="bernoulli" if metric in ("fidm", "fidp") else self.config.samplingtype
                        )
                        belonging.append(i)
                        eval_samples.append(G_c)
                elif metric == "suff" or metric == "suff++" or metric == "suff_simple":
                    if ratio == 1.0:
                        eval_samples.extend([graphs[i]]*self.config.expval_budget)
                        belonging.extend([i]*self.config.expval_budget)
                    else:
                        z, c = -1, 0
                        idxs = np.random.permutation(np.arange(len(labels))) #pick random from every class
                        budget = self.config.expval_budget
                        if metric == "suff++":
                            budget = budget // 2
                        if metric == "suff_simple":
                            budget = 0 # just pick subsamples
                        while c < budget:
                            assert False, "I'm using suff-Simple at the moment"
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
                                causal_subgraphs_r[ratio][j],
                                spu_subgraphs_r[ratio][j],
                                G_filt,
                                debug,
                                (i, j, c),
                                feature_intervention=False,
                                feature_bank=None
                            )
                            if G_union is None:
                                continue
                            eval_samples.append(G_union)
                            belonging.append(i)
                            c += 1
                        for k in range(c, self.config.expval_budget): # if not enough interventions, pad with sub-sampling
                            G_c = xai_utils.sample_edges_tensorized(
                                graphs[i],
                                nec_number_samples=self.config.nec_number_samples,
                                nec_alpha_1=self.config.nec_alpha_1,
                                avg_graph_size=avg_graph_size,
                                edge_index_to_remove=~graphs[i].edge_gt, #~causal_masks_r[ratio][i]&
                                sampling_type="bernoulli" if metric in ("fidm", "fidp") else self.config.samplingtype
                            )
                            belonging.append(i)
                            eval_samples.append(G_c)
            
            # Compute new prediction and evaluate KL
            int_dataset = CustomDataset("", eval_samples, belonging)
            loader = DataLoader(int_dataset, batch_size=512, shuffle=False)

            # PLOT EXAMPLES OF EXPLANATIONS
            # print(len(edge_scores), len(graphs), len(int_dataset), self.config.expval_budget, avg_graph_size)
            # for i in range(3):
            #     if i > 3:
            #         break
            #     data = int_dataset[reference[i]]
            #     g = to_networkx(data)
            #     xai_utils.mark_edges(g, data.edge_index, data.edge_index[:, data.edge_gt == 1], inv_edge_w=edge_scores[i])
            #     xai_utils.draw(self.config, g, subfolder="plots_of_gt_stability_det", name=f"graph_{i}")

            # for i, data in enumerate(loader):
            #     if i > 15:
            #         break
            #     data: Batch = data.to(self.config.device)
            #     edge_score = self.model.get_subgraph(
            #         data=data,
            #         edge_weight=None,
            #         ood_algorithm=self.ood_algorithm,
            #         do_relabel=False,
            #         return_attn=False,
            #         ratio=None
            #     )
            #     wiou = xai_utils.expl_acc_super_fast(None, data, edge_score)[0]
            #     g = to_networkx(data, node_attrs=["node_gt"], to_undirected=True) #to_undirected=True
            #     xai_utils.mark_edges(g, data.edge_index, data.edge_index[:, data.edge_gt == 1], inv_edge_w=edge_score)
            #     xai_utils.draw(self.config, g, subfolder="plots_of_gt_stability_det", name=f"expl_{i}", title=wiou)
            # exit("sare")
            # END OF PLOT EXAMPLES OF EXPLANATIONS

            
            print("Computing with masking")
            startTime = datetime.datetime.now()

            # return explanations and make predictions
            preds_eval, belonging, wious = [], [], []
            wious = torch.tensor([], device=self.config.device)
            belonging = torch.tensor([], device=self.config.device, dtype=int)
            for i, data in enumerate(loader):
                data: Batch = data.to(self.config.device)
                edge_score = self.model.get_subgraph(
                    data=data,
                    edge_weight=None,
                    ood_algorithm=self.ood_algorithm,
                    do_relabel=False,
                    return_attn=False,
                    ratio=None
                )
                
                # compute Plausibility of expl. after intervention
                if "CIGA" in self.config.model.model_name:
                    edge_score = edge_score.sigmoid() # normalize scores for computing WIOU

                wious = torch.cat((wious, xai_utils.expl_acc_super_fast(None, data, edge_score)[0]), dim=0)
                # for j, g in enumerate(data.to_data_list()):
                #     wious.append(xai_utils.expl_acc_super_fast(g.edge_index, g, edge_score[data.batch[data.edge_index[0]] == j])[0])
                # wious.append(xai_utils.expl_acc(data.edge_index, data, edge_score)[0])

                # threshold explanation before feeding to classifier
            #     (causal_edge_index, causal_edge_attr, edge_att), _ = split_graph(data, edge_score, ratio)
            #     data.x, data.edge_index, data.batch, _ = relabel(data.x, causal_edge_index, data.batch)
            #     if not data.edge_attr is None:
            #         data.edge_attr = causal_edge_attr

            #     ori_out = self.model.predict_from_subgraph(
            #         edge_att=edge_att,
            #         data=data,
            #         edge_weight=None,
            #         ood_algorithm=self.ood_algorithm,
            #         log=True,
            #         eval_kl=True
            #     )
            #     preds_eval.extend(ori_out.detach().cpu().numpy().tolist())
                # belonging.extend(data.belonging.detach().cpu().numpy().tolist())
                belonging = torch.cat((belonging, data.belonging), dim=0)
            # preds_eval = torch.tensor(preds_eval)
            # belonging = torch.tensor(belonging, dtype=int)
            # just make predictions
            # preds_eval, belonging = self.evaluate_graphs(loader, log=False if "fid" in metric else True, weight=ratio, is_ratio=is_ratio, eval_kl=True)
            
            print("Completed in ", datetime.datetime.now() - startTime)
            
            # divide predictions and labels into original and intervened
            # preds_ori = preds_eval[reference]
            
            mask = torch.ones(belonging.shape[0], dtype=bool, device=belonging.device)
            mask[reference] = False
            # preds_eval = preds_eval[mask]
            belonging = belonging[mask]            
            assert torch.all(belonging >= 0), f"{torch.all(belonging >= 0)}"

            # labels_ori_ori = torch.tensor(labels_ori)
            # preds_ori_ori = preds_ori
            # preds_ori = preds_ori_ori.repeat_interleave(self.config.expval_budget, dim=0)
            # labels_ori = labels_ori_ori.repeat_interleave(self.config.expval_budget, dim=0)

            # divide Plausibility into original and intervened
            wiou_ori = wious[reference]
            wiou_int = wious[mask]

            # plot Plausibility
            # idx_order = torch.argsort(wiou_ori)
            # plt.plot(wiou_ori[idx_order], label="original samples")
            # plt.plot(scatter_mean(wiou_int, belonging, dim=0)[idx_order], label="perturbed samples")
            # plt.xlabel("ordered sample idx", fontsize=13)
            # plt.ylabel("plausibility", fontsize=13)
            # plt.legend()
            # path =  f'GOOD/kernel/pipelines/plots/plots_of_gt_stability_det/' \
            #         f'{self.config.load_split}_{self.config.util_model_dirname}/'
            # if not os.path.exists(path):
            #     try:
            #         os.makedirs(path)
            #     except Exception as e:
            #         exit(e)
            # plt.savefig(path + f'wious_{split}_{metric}_r{ratio}_b{self.config.nec_alpha_1}_seed{seed}.png')
            # plt.close()
           
            # compute and store metrics
            # aggr, aggr_std = self.get_aggregated_metric(metric, preds_ori, preds_eval, preds_ori_ori, labels_ori, belonging, results, ratio)            
            # scores[f"all_L1"].append(round(aggr["L1"].mean().item(), 3))
            scores[f"wiou_original"].append(wiou_ori.mean().item())
            scores[f"wiou_perturbed"].append(scatter_mean(wiou_int, belonging, dim=0).mean().item())
            # scores[f"wiou_original_std"].append(wiou_ori.std().item())
            # scores[f"wiou_perturbed_std"].append(scatter_mean(wiou_int, belonging, dim=0).std().item())

            # assert preds_ori_ori.shape[1] > 1, preds_ori_ori.shape
            # dataset_metric = self.loader["id_val"].dataset.metric
            # if dataset_metric == "ROC-AUC":
            #     if not "fid" in metric:
            #         preds_ori_ori = preds_ori_ori.exp() # undo the log
            #         preds_eval = preds_eval.exp()
            #     acc = sk_roc_auc(labels_ori_ori.long(), preds_ori_ori[:, 1], multi_class='ovo')
            #     acc_int = sk_roc_auc(labels_ori.long(), preds_eval[:, 1], multi_class='ovo')
            # elif dataset_metric == "F1":
            #     acc = f1_score(labels_ori_ori.long(), preds_ori_ori.argmax(-1), average="binary", pos_label=dataset.minority_class)
            #     acc_int = f1_score(labels_ori.long(), preds_eval.argmax(-1), average="binary", pos_label=dataset.minority_class)
            # else:
            #     if preds_ori_ori.shape[1] == 1:
            #         assert False
            #         if not "fid" in metric:
            #             preds_ori_ori = preds_ori_ori.exp() # undo the log
            #             preds_eval = preds_eval.exp()
            #         preds_ori_ori = preds_ori_ori.round().reshape(-1)
            #         preds_eval = preds_eval.round().reshape(-1)
            #     acc = (labels_ori_ori == preds_ori_ori.argmax(-1)).sum() / (preds_ori_ori.shape[0])
            #     acc_int = (labels_ori == preds_eval.argmax(-1)).sum() / preds_eval.shape[0]

            # acc_ints.append(acc_int)
            # print(f"\nModel {dataset_metric} of binarized graphs for r={ratio} = ", round(acc.item(), 3))
            # print(f"Original XAI F1 for r={ratio} = ", np.mean([e[1] for e in expl_accs_r[ratio]]))
            # print(f"Intervened XAI F1 for r={ratio} = ", np.mean([e[1] for e in int_expl_accs_r[ratio]]))
            print(f"Original XAI WIoU for r={ratio} = ", wiou_ori.mean(), " +- ", wiou_ori.std())
            print(f"Intervened XAI WIoU for r={ratio} = ", scatter_mean(wiou_int, belonging, dim=0).mean(), " +- ", scatter_mean(wiou_int, belonging, dim=0).std(), f"({scatter_std(wiou_int, belonging, dim=0).mean()})")
            print(f"len(reference) = {len(reference)}")
            # if preds_eval.shape[0] > 0:
                # print(f"Model {dataset_metric} over intervened graphs for r={ratio} = ", round(acc_int.item(), 3))
                # print(f"{metric.upper()} for r={ratio} all L1 = {scores[f'all_L1'][-1]} +- {aggr['L1'].std():.3f} (in-sample avg dev_std = {(aggr_std**2).mean().sqrt():.3f})")
        return scores, acc_ints, results


    @torch.no_grad()
    def generate_explanation_examples(
        self,
        seed,
        ratios,
        split: str,
        metric: str,
        edge_scores,
        graphs,
        graphs_nx,
        labels,
        avg_graph_size,
        causal_subgraphs_r,
        spu_subgraphs_r,
        expl_accs_r,
        causal_masks_r,
        intervention_bank,
        intervention_distrib:str = "model_dependent",
        debug=False,
    ):
        print(f"\n\n")
        print("-"*50)
        print(f"\n\n#D#Plotting examples of explanations for {metric.upper()} over {split} across ratios (random_expl={self.config.random_expl})")
        reset_random_seed(self.config)
        self.model.eval()   

        scores, results, acc_ints = defaultdict(list), {}, []
        ratios = [1.0]
        for ratio in ratios:
            reset_random_seed(self.config)
            print(f"\n\nratio={ratio}\n\n")            

            eval_samples, belonging, reference = [], [], []
            preds_ori, labels_ori, expl_acc_ori = [], [], []
            empty_idx = set()

            pbar = tqdm(range(len(edge_scores)), desc=f'Creating Intervent. distrib.', total=len(edge_scores), **pbar_setting)
            for i in pbar:
                if graphs[i].edge_index.shape[1] <= 6:
                    continue                
                if metric in ("suff", "suff++") and intervention_distrib == "model_dependent":
                    G = graphs_nx[i].copy()
                    G_filt = xai_utils.remove_from_graph(G, edge_index_to_remove=spu_subgraphs_r[ratio][i])
                    num_elem = xai_utils.mark_frontier(G, G_filt)
                    if len(G_filt) == 0 or num_elem == 0:
                        continue
                
                eval_samples.append(graphs[i])
                reference.append(len(eval_samples) - 1)
                belonging.append(-1)
                labels_ori.append(labels[i])
                # expl_acc_ori.append(expl_accs_r[ratio][i])

                if metric in ("nec", "nec++") or len(empty_idx) == len(graphs) or intervention_distrib in ("fixed", "bank"):
                    assert False, "Computing NEC interventions"
                    if metric in ("suff", "suff++", "suff_simple") and intervention_distrib in ("fixed", "bank") and i == 0:
                        print(f"Using {intervention_distrib} interventional distribution")
                    elif metric in ("suff", "suff++", "suff_simple") and intervention_distrib == "model_dependent":
                        pass

                    for m in range(self.config.expval_budget):                        
                        G_c = xai_utils.sample_edges_tensorized(
                            graphs[i],
                            nec_number_samples=self.config.nec_number_samples,
                            nec_alpha_1=self.config.nec_alpha_1,
                            avg_graph_size=avg_graph_size,
                            edge_index_to_remove=causal_masks_r[ratio][i],
                            sampling_type="bernoulli" if metric in ("fidm", "fidp") else self.config.samplingtype
                        )
                        belonging.append(i)
                        eval_samples.append(G_c)
                elif metric == "suff" or metric == "suff++" or metric == "suff_simple":
                    if ratio == 1.0:
                        eval_samples.extend([graphs[i]]*self.config.expval_budget)
                        belonging.extend([i]*self.config.expval_budget)
                    else:
                        z, c = -1, 0
                        idxs = np.random.permutation(np.arange(len(labels))) #pick random from every class
                        budget = self.config.expval_budget
                        if metric == "suff++":
                            budget = budget // 2
                        if metric == "suff_simple":
                            budget = 0 # just pick subsamples
                        while c < budget:
                            assert False, "I'm using suff-Simple at the moment"
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
                                causal_subgraphs_r[ratio][j],
                                spu_subgraphs_r[ratio][j],
                                G_filt,
                                debug,
                                (i, j, c),
                                feature_intervention=False,
                                feature_bank=None
                            )
                            if G_union is None:
                                continue
                            eval_samples.append(G_union)
                            belonging.append(i)
                            c += 1
                        for k in range(c, self.config.expval_budget): # if not enough interventions, pad with sub-sampling
                            G_c = xai_utils.sample_edges_tensorized(
                                graphs[i],
                                nec_number_samples=self.config.nec_number_samples,
                                nec_alpha_1=self.config.nec_alpha_1,
                                avg_graph_size=avg_graph_size,
                                edge_index_to_remove=~graphs[i].edge_gt, #~causal_masks_r[ratio][i]&
                                sampling_type="bernoulli" if metric in ("fidm", "fidp") else self.config.samplingtype
                            )
                            belonging.append(i)
                            eval_samples.append(G_c)
            
            int_dataset = CustomDataset("", eval_samples, belonging)
            loader = DataLoader(int_dataset, batch_size=512, shuffle=False)

            # PLOT EXAMPLES OF EXPLANATIONS (ORIGINAL SAMPLES)
            print(len(edge_scores), len(graphs), len(int_dataset), self.config.expval_budget, avg_graph_size)
            for i in range(15):
                if i > 15:
                    break
                data = int_dataset[reference[i]]
                g = to_networkx(data, node_attrs=["node_gt", "x"], to_undirected=True)
                xai_utils.mark_edges(g, data.edge_index, data.edge_index[:, data.edge_gt == 1], inv_edge_w=edge_scores[i])
                xai_utils.draw_colored(
                    self.config,
                    g,
                    subfolder=f"plots_of_explanation_examples/{self.config.ood_dirname}/{self.config.dataset.dataset_name}_{self.config.dataset.domain}",
                    name=f"graph_{reference[i]}",
                    thrs=0.05,
                    title=f"Class {data.y}"
                )
                print(f"graph_{reference[i]} is of class {labels_ori[reference[i]]}")

            # PLOT EXAMPLES OF EXPLANATIONS (INTERVENED SAMPLES)
            # for i, data in enumerate(loader):
            #     if i > 15:
            #         break
            #     data: Batch = data.to(self.config.device)
            #     edge_score = self.model.get_subgraph(
            #         data=data,
            #         edge_weight=None,
            #         ood_algorithm=self.ood_algorithm,
            #         do_relabel=False,
            #         return_attn=False,
            #         ratio=None
            #     )
            #     wiou = xai_utils.expl_acc_super_fast(None, data, edge_score)[0]
            #     g = to_networkx(data, node_attrs=["node_gt"], to_undirected=True) #to_undirected=True
            #     xai_utils.mark_edges(g, data.edge_index, data.edge_index[:, data.edge_gt == 1], inv_edge_w=edge_score)
            #     xai_utils.draw(self.config, g, subfolder="plots_of_gt_stability_det", name=f"expl_{i}", title=wiou)
        return 
    

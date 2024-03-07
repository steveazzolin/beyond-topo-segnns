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
from torch_scatter import scatter_mean, scatter_std, scatter_min, scatter_max
from munch import Munch
# from torch.utils.data import DataLoader
import torch_geometric
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data, InMemoryDataset
from torch_geometric.utils import to_networkx, from_networkx, to_undirected, sort_edge_index, shuffle_node
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
from GOOD.utils.splitting import split_graph, sparse_sort

pbar_setting["disable"] = True

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
            if not hasattr(data, "ori_x"):
                print(i)
                print(G.nodes(data=True))
                print(data)
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
            expl_accs.append(xai_utils.expl_acc(causal_subgraphs[-1], graphs[j], norm_edge_scores[j]) if hasattr(graphs[j], "edge_gt") else (np.nan,np.nan))


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
        if metric == "fidm" or (metric == "suff" and intervention_distrib == "model_dependent" and causal is None):
            return xai_utils.sample_edges(graph, "spu", self.config.fidelity_alpha_2, spu)
        elif metric in ("nec", "nec++", "fidp"):
            if metric == "nec++":
                alpha = max(self.config.nec_alpha_1 - 0.1 * (j // 3), 0.1)
            else:
                alpha = self.config.nec_alpha_1
            return xai_utils.sample_edges(graph, "inv", alpha, causal)
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
            G_t = graph.copy()
            # xai_utils.mark_edges(G_t, causal, spu)
            G_t_filt = xai_utils.remove_from_graph(G_t, "inv", causal)
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


    @torch.no_grad()
    def compute_metric_ratio(self, split: str, metric: str, intervention_distrib:str = "model_dependent", debug=False, edge_scores=None, graphs=None):
        assert metric in ["suff", "fidm", "nec", "nec++", "fidp"]

        do_feature_intervention = False
        if "CIGA" in self.config.model.model_name:
            is_ratio = True
            weights = [self.model.att_net.ratio]
        else:
            is_ratio = True
            if "sst2" in self.config.dataset.dataset_name.lower() and split in ("id_val", "train"):
                weights = [0.6, 0.9, 1.0]
            else:
                weights = [0.3, 0.6, 0.9, 1.0]

        print(f"\n\n")
        print("-"*50)
        print(f"\n\n#D#Computing {metric.upper()} over {split} across ratios (random_expl={self.config.random_expl})")
        reset_random_seed(self.config)
        self.model.eval()   

        dataset = self.get_local_dataset(split)
        
        intervent_bank = None
        features_bank = None
        if intervention_distrib == "bank":
            if torch_geometric.__version__ == "2.4.0": 
                features_bank = dataset.x.unique(dim=0).cpu()
            else:
                features_bank = dataset.data.x.unique(dim=0).cpu()
            print(f"Shape of feature bank = {features_bank.shape}")
            print(f"Creating interventional bank with {self.config.expval_budget} elements")
            intervent_bank = []
            max_g_size = max([d.num_nodes for d in dataset])
            for i in range(self.config.expval_budget):
                I = nx.DiGraph(nx.barabasi_albert_graph(random.randint(5, max(int(max_g_size/2), 8)), 1), seed=42) #BA1 -> nx.barabasi_albert_graph(randint(5, max(len(G), 8)), randint(1, 3))
                nx.set_edge_attributes(I, name="origin", values="BA")
                if "motif" in self.config.dataset.dataset_name.lower():
                    nx.set_node_attributes(I, name="ori_x", values=1.0)
                else:
                    nx.set_node_attributes(I, name="ori_x", values=features_bank[random.randint(0, features_bank.shape[0]-1)].tolist())
                intervent_bank.append(I)  
        
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

        loader = DataLoader(dataset[idx], batch_size=256, shuffle=False)
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
                    return_attn=False,
                    ratio=None
                )
                if self.config.random_expl:
                    edge_score = edge_score[
                        shuffle_node(torch.arange(edge_score.shape[0], device=edge_score.device), batch=data.batch[data.edge_index[0]])[1]
                    ] # maybe biased?
                    data.edge_index, edge_score = to_undirected(data.edge_index, edge_score, reduce="mean")

                    # max_val = scatter_max(edge_score.cpu(), index=data.batch[data.edge_index[0]].cpu())[0]
                    # edge_score = -edge_score.cpu()
                    # edge_score = edge_score + max_val[data.batch[data.edge_index[0]].cpu()]
                    # edge_score = edge_score.to(self.config.device)
                # attn_distrib.append(self.model.attn_distrib)
                for j, g in enumerate(data.to_data_list()):
                    g.ori_x = data.ori_x[data.batch == j]
                    g.ori_edge_index = data.ori_edge_index[:, data.batch[data.ori_edge_index[0]] == j]
                    edge_scores.append(edge_score[data.batch[data.edge_index[0]] == j].detach().cpu())
                    graphs.append(g.detach().cpu())
            labels.extend(data.y.detach().cpu().numpy().tolist())
        labels = torch.tensor(labels)
        graphs_nx = [to_networkx(g, node_attrs=["ori_x"], edge_attrs=["edge_attr"] if not g.edge_attr is None else None) for g in graphs]

        # plot attn_distrib and compute the ratio between gt edges and all edges (gold cut ratio)
        # self.plot_attn_distrib(attn_distrib, edge_scores)

        if hasattr(graphs[0], "edge_gt"):
            num_gt_edges = torch.tensor([data.edge_gt.sum() for data in graphs])
            num_all_edges = torch.tensor([data.edge_index.shape[1] for data in graphs])
            print("\nGold ratio = ", torch.mean(num_gt_edges / num_all_edges), "+-", torch.std(num_gt_edges / num_all_edges))

        scores, results, acc_ints = defaultdict(list), {}, []
        for ratio in weights:
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
            effective_ratio = [causal_subgraphs[i].shape[1] / (causal_subgraphs[i].shape[1] + spu_subgraphs[i].shape[1] + 1e-5) for i in range(len(spu_subgraphs))]            

            pbar = tqdm(range(len(idx)), desc=f'Creating Intervent. distrib.', total=len(idx), **pbar_setting)
            for i in pbar:
                G = graphs_nx[i].copy()

                if len(G.edges()) == 0:
                    continue
                
                if metric == "suff" and intervention_distrib == "model_dependent":
                    G_filt = xai_utils.remove_from_graph(G, "spu", spu_subgraphs[i])
                    num_elem = xai_utils.mark_frontier(G, G_filt)
                    if len(G_filt) == 0 or num_elem == 0:
                        continue
                    # G = G_filt # P(Y|G) vs P(Y|R)
                
                eval_samples.append(G)
                reference.append(len(eval_samples) - 1)
                belonging.append(-1)
                labels_ori.append(labels[i])
                expl_acc_ori.append(expl_accs[i])

                if metric in ("fidm", "fidp", "nec", "nec++") or len(empty_idx) == len(graphs) or intervention_distrib in ("fixed", "bank"):
                    if metric == "suff" and intervention_distrib in ("fixed", "bank") and i == 0:
                        print(f"Using {intervention_distrib} interventional distribution")
                    elif metric == "suff" and intervention_distrib == "model_dependent":
                        # print("Empty graphs for SUFF. Rolling-back to FIDM")
                        pass

                    for m in range(self.config.expval_budget):                        
                        G_c = self.get_intervened_graph(metric if metric != "suff" else "fidm", intervention_distrib, G, idx=(i,m,-1), bank=intervent_bank, causal=causal_subgraphs[i], spu=spu_subgraphs[i],)                        
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
                for c in labels_ori_ori.unique():
                    scores[c.item()].append(1.0)
                scores["all"].append(1.0)
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
            
            assert aggr.shape[0] == labels_ori_ori.shape[0]
            for c in labels_ori_ori.unique():
                idx_class = np.arange(labels_ori_ori.shape[0])[labels_ori_ori == c]
                scores[c.item()].append(round(aggr[idx_class].mean().item(), 3))
            score = round(aggr.mean().item(), 3)
            scores["all"].append(score)

            assert preds_ori_ori.shape[1] > 1, preds_ori_ori.shape
            if dataset.metric == "ROC-AUC":
                if not "fid" in metric:
                    preds_ori_ori = preds_ori_ori.exp() # undo the log
                    preds_eval = preds_eval.exp()
                acc = sk_roc_auc(labels_ori_ori.long(), preds_ori_ori[:, 1], multi_class='ovo')
                acc_int = sk_roc_auc(labels_ori.long(), preds_eval[:, 1], multi_class='ovo')
            elif dataset.metric == "F1":
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
                preds_ori_ori = preds_ori_ori.argmax(-1)  
                preds_eval = preds_eval.argmax(-1)  
                acc = (labels_ori_ori == preds_ori_ori).sum() / (preds_ori_ori.shape[0])
                acc_int = (labels_ori == preds_eval).sum() / preds_eval.shape[0]

            acc_ints.append(acc_int)
            print(f"\nModel {dataset.metric} of binarized graphs for r={ratio} = ", round(acc.item(), 3))
            print(f"Model XAI F1 of binarized graphs for r={ratio} = ", np.mean([e[1] for e in expl_accs]))
            print(f"Model XAI WIoU of binarized graphs for r={ratio} = ", np.mean([e[0] for e in expl_accs]))
            print(f"len(reference) = {len(reference)}")
            print(f"Effective ratio: {np.mean(effective_ratio):.3f} +- {np.std(effective_ratio):.3f}")
            if preds_eval.shape[0] > 0:
                print(f"Model {dataset.metric} over intervened graphs for r={ratio} = ", round(acc_int.item(), 3))
                for c in labels_ori_ori.unique().numpy().tolist() + ["all"]:
                    print(f"{metric.upper()} for r={ratio} class {c} = {scores[c][-1]} +- {aggr.std():.3f} (in-sample avg dev_std = {(aggr_std**2).mean().sqrt():.3f})")


        return scores, acc_ints, results, edge_scores, graphs


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
        belonging = torch.tensor(self.normalize_belonging(belonging))

        if metric in ("suff", "nec", "nec++") and preds_eval.shape[0] > 0:
            div = torch.nn.KLDivLoss(reduction="none", log_target=True)(preds_ori, preds_eval).sum(-1)
            
            # print(preds_ori[:5].exp())
            # print(preds_eval[:5].exp())
            # print(div[:5])
            # print(torch.exp(-div[:5]))
            # print(scatter_mean(div, belonging, dim=0).mean().item())
            # print(torch.exp(-scatter_mean(div, belonging, dim=0).mean()))
            # print(torch.exp(-scatter_mean(div, belonging, dim=0)).mean()) # on paper
            # print(scatter_mean(torch.exp(-div), belonging, dim=0).mean().item())
            # exit()

            results[ratio] = div.numpy().tolist()
            if metric == "suff":
                # div = torch.exp(-div) # used so far
                aggr = torch.exp(-scatter_mean(div, belonging, dim=0)) # on paper
            elif metric in ("nec", "nec++"):
                # div = 1 - torch.exp(-div)
                aggr = 1 - torch.exp(-scatter_mean(div, belonging, dim=0)) # on paper
            # aggr = scatter_mean(div, belonging, dim=0) # used so far
            aggr_std = scatter_std(div, belonging, dim=0)
        elif "fid" in metric and preds_eval.shape[0] > 0:
            if preds_ori_ori.shape[1] == 1:
                l1 = torch.abs(preds_eval.reshape(-1) - preds_ori.reshape(-1))
            else:
                l1 = torch.abs(preds_eval.gather(1, labels_ori.unsqueeze(1)) - preds_ori.gather(1, labels_ori.unsqueeze(1)))
            results[ratio] = l1.numpy().tolist()
            aggr = scatter_mean(l1, belonging, dim=0)
            aggr_std = scatter_std(l1, belonging, dim=0)                    
        else:
            raise ValueError(metric)
        return aggr, aggr_std


    @torch.no_grad()
    def compute_accuracy_binarizing(self, split: str, givenR, debug=False, metric_collector=None):
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


        loader = DataLoader(dataset[idx], batch_size=256, shuffle=False)
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
            for j, g in enumerate(data.to_data_list()):
                g.ori_x = data.ori_x[data.batch == j]
                g.ori_edge_index = data.ori_edge_index[:, data.batch[data.ori_edge_index[0]] == j]
                graphs.append(g.detach().cpu())
                edge_scores.append(edge_score[data.batch[data.edge_index[0]] == j].detach().cpu())
        labels = torch.tensor(labels)
        graphs_nx = [
            to_networkx(g, node_attrs=["ori_x"], edge_attrs=["edge_attr"] if not g.edge_attr is None else None) for g in graphs
        ]
        # self.plot_attn_distrib([[]], edge_scores)

        acc_scores, plaus_scores, wiou_scores = [], defaultdict(list), defaultdict(list)
        for weight in weights:
            print(f"\n\nr={weight}\n")
            eval_samples, labels_ori = [], []
            empty_graphs = 0
            
            # Select relevant subgraph based on ratio
            if is_ratio:
                causal_subgraphs, spu_subgraphs, expl_accs = self.get_subragphs_ratio(graphs, weight, edge_scores)
            else:
                causal_subgraphs, spu_subgraphs, expl_accs, causal_idxs, spu_idxs = self.get_subragphs_weight(graphs, weight, edge_scores)            
            effective_ratio = np.array([causal_subgraphs[i].shape[1] / (causal_subgraphs[i].shape[1] + spu_subgraphs[i].shape[1] + 1e-5) for i in range(len(spu_subgraphs))])

            # Create interventional distribution     
            pbar = tqdm(range(len(idx)), desc=f'Int. distrib', total=len(idx), **pbar_setting)
            for i in pbar:                
                G = graphs_nx[i].copy()
                G_filt = G

                if len(G.edges()) == 0:
                    empty_graphs += 1
                    continue
                if givenR: # for P(Y|R)
                    G_filt = xai_utils.remove_from_graph(G, "spu", spu_subgraphs[i])                    
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
                    d[c.item()].append(np.mean([e[q] for e in expl_accs]))
                    print(f"Model XAI {s} r={weight} class {c.item()} \t= {d[c.item()][-1]:.3f}")
            for q, (d, s) in enumerate(zip([wiou_scores, plaus_scores], ["WIoU", "F1"])):
                d["all"].append(np.mean([e[q] for e in expl_accs]))
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
        
        was_training = self.model.training
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
        stat['score'] = eval_score(pred_all, target_all, self.config, self.loader[split].dataset.minority_class)

        print(f'{split.capitalize()} {self.config.metric.score_name}: {stat["score"]:.4f}'
              f'{split.capitalize()} Loss: {stat["loss"]:.4f}')

        if was_training:
            self.model.train()

        return {
            'score': stat['score'],
            'loss': stat['loss']
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

    def generate_panel(self):
        self.model.eval()
        splits = ["train", "id_val", "val", "test"]
        n_row = 3
        fig, axs = plt.subplots(n_row, n_row, figsize=(14,14))
        plt.suptitle(f"{self.config.model.model_name} {self.config.dataset.dataset_name}")
        
        for i, split in enumerate(splits):            
            acc = self.evaluate(split, compute_suff=False)["score"]
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

            axs[int(i/n_row), int(i%n_row)].hist(np.concatenate(edge_scores), bins=100, density=False, log=True)
            axs[int(i/n_row), int(i%n_row)].set_title(f"Attn. distribution {split} ({acc:.3f}%)")
            axs[int(i/n_row), int(i%n_row)].set_xlabel(f"attention scores")
            axs[int(i/n_row), int(i%n_row)].set_ylabel(f"")
            axs[int(i/n_row), int(i%n_row)].set_xlim(0, 1.1)

            means, stds = zip(*[(np.mean(e), np.std(e)) for e in edge_scores])
            means, stds = self.smooth(np.array(means), k=5), np.array(stds)
            axs[n_row - int(i/n_row) -1, n_row - int(i%n_row) -1].plot(np.arange(len(means)), means)
            axs[n_row - int(i/n_row) -1, n_row - int(i%n_row) -1].plot(np.arange(len(effective_ratios)), self.smooth(effective_ratios, k=7), 'r', alpha=0.7)
            axs[n_row - int(i/n_row) -1, n_row - int(i%n_row) -1].fill_between(np.arange(len(means)), means - stds, means + stds, alpha=0.5)
            axs[n_row - int(i/n_row) -1, n_row - int(i%n_row) -1].set_title(f"Per sample attn. variability - {split}")
            axs[n_row - int(i/n_row) -1, n_row - int(i%n_row) -1].set_ylim(0, 1.1)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        path = f'GOOD/kernel/pipelines/plots/panels/{self.config.load_split}_{self.config.dataset.dataset_name}_{self.config.dataset.domain}_{self.config.util_model_dirname}_{self.config.random_seed}.png'
        plt.savefig(path)
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
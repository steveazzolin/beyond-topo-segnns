r"""Training pipeline: training/evaluation structure, batch training.
"""
import datetime
import os
import shutil
from typing import Dict
from typing import Union
from random import randint

import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from munch import Munch
# from torch.utils.data import DataLoader
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


class CustomDataset(InMemoryDataset):
    def __init__(self, root, samples, belonging, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        data_list = []
        for i , G in enumerate(samples):
            data = from_networkx(G)
            data.belonging = belonging[i]
            # data = Data(x=features, 
            #             edge_index=t.edge_index, 
            #             edge_attr=torch.tensor(t.weight).reshape(-1, 1),
            #             num_nodes=adj.shape[0],
            #             y=torch.tensor(int(y[i]), dtype=torch.long) if y is not None else None, # the type of local explanation
            #             task_y=torch.tensor(int(task_y[belonging[i]]), dtype=torch.long) if y is not None else None, # the class of the original input graph
            #             le_id=torch.tensor(i, dtype=torch.long),
            #             graph_id=belonging[i])
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


    def compute_sufficiency2(self, split: str, debug=False):
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
        def remove_from_graph(G, what_to_remove):
            G = G.copy()
            edge_remove = []
            for (u,v), val in nx.get_edge_attributes(G, 'origin').items():
                if val == what_to_remove:
                    edge_remove.append((u,v))
            G.remove_edges_from(edge_remove)
            G.remove_edges_from([(v,u) for v,u in G.edges() if not G.has_edge(u,v)])
            G.remove_nodes_from(list(nx.isolates(G)))
            return G

        def mark_edges(G, inv_edge_index, spu_edge_index):
            nx.set_edge_attributes(
                G,
                name="origin",
                values={(u.item(), v.item()): "inv" for u,v in inv_edge_index.T}
            )
            nx.set_edge_attributes(
                G,
                name="origin",
                values={(u.item(), v.item()): "spu" for u,v in spu_edge_index.T}
            )

        def mark_frontier(G, G_filt):
            # mark frontier nodes as nodes attached to both inv and spu parts
            # to mark nodes check which nodes have a change in the degree between original and filtered graph
            frontier = []
            for n in G_filt.nodes():
                if G.degree[n] != G_filt.degree[n]:                    
                    frontier.append(n)            
            nx.set_node_attributes(G_filt, name="frontier", values=False)
            nx.set_node_attributes(G_filt, name="frontier", values={n: True for n in frontier})
            return len(frontier)

        def draw(G, name, pos=None):
            if pos is None:
                pos = nx.kamada_kawai_layout(G)
            nx.draw(
                G,
                with_labels = True,
                pos=pos,
                edge_color=list(map(lambda x: edge_colors[x], nx.get_edge_attributes(G,'origin').values())),
                node_color=list(map(lambda x: node_colors[x], [nx.get_node_attributes(G,'frontier').get(n, False) for n in G.nodes()])),
            )
            plt.savefig(f'GOOD/kernel/pipelines/plots/{name}.png')
            plt.close()
            return pos
        
        def random_attach(S, T):
            # random attach frontier nodes in S and T

            S_frontier = list(filter(lambda x: nx.get_node_attributes(S,'frontier').get(x, False), S.nodes()))
            T_frontier = list(filter(lambda x: nx.get_node_attributes(T,'frontier').get(x, False), T.nodes()))

            ret = nx.union(S, T, rename=("", "T"))
            for n in S_frontier:
                # pick random node v in G_t_spu
                # add edge (u,v) and (v,u)
                idx = randint(0, len(T_frontier)-1)
                v = "T" + str(T_frontier[idx])

                assert str(n) in ret.nodes() and v in ret.nodes()

                ret.add_edge(str(n), v, origin="added")
                ret.add_edge(v, str(n), origin="added")
            return ret
        
        edge_colors = {
            "inv": "green",
            "spu": "blue",
            "added": "red"
        }
        node_colors = {
            True: "red",
            False: "#1f78b4"
        }

        print(f"#D#Computing SUFF over {split}")
        print(self.loader[split].dataset)

        loader = DataLoader(self.loader[split].dataset[:100], batch_size=1, shuffle=False)
        if self.config.numsamples_budget == "all":
            self.config.numsamples_budget = len(loader)

        self.model.eval()

        pbar_setting["disable"] = False
        pbar = tqdm(loader, desc=f'Eval {split.capitalize()}', total=len(loader), **pbar_setting)
        preds_all = []
        graphs = []
        causal_subgraphs = []
        spu_subgraphs = []
        labels = []
        for data in pbar:
            data: Batch = data.to(self.config.device)
            output = self.model.log_probs(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm)
            preds_all.append(output[0].detach().cpu().numpy().tolist())
            labels.extend(data.y.detach().cpu().numpy().tolist())

            (causal_edge_index, causal_x, causal_batch, causal_edge_weight), \
                (spu_edge_index, spu_x, spu_batch, causal_edge_weight) = self.model.get_subgraph(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm, do_relabel=False)

            graphs.append(data.detach().cpu())
            causal_subgraphs.append(causal_edge_index.detach().cpu())
            spu_subgraphs.append(spu_edge_index.detach().cpu())
        labels = np.array(labels)


        ##
        # Create interventional distribution
        ##
        
        eval_samples = []
        belonging = []
        preds_ori= []
        for i in range(self.config.numsamples_budget):
            preds_ori.append(preds_all[i])
            G = to_networkx(
                graphs[i],
                node_attrs=["x", "x_debug"]
            )
            mark_edges(G, causal_subgraphs[i], spu_subgraphs[i])                

            G_filt = remove_from_graph(G, "spu")
            num_elem = mark_frontier(G, G_filt)

            if num_elem == 0:
                print("\nZero frontier here ", i)
                draw(G_filt, name=f"debug_graph_{i}")

            if debug:
                pos = draw(G, name=f"graph_{i}")
                draw(G, name=f"inv_graph_{i}", pos=pos)

            
            z, c = -1, 0
            idxs = np.random.permutation(np.arange(len(labels))[labels == labels[i]]) #pick random from same class
            while c < self.config.expval_budget:
            # for j in idxs[:self.config.expval_budget]:
                z += 1
                j = idxs[z]
                G_t = to_networkx(
                    graphs[j],
                    node_attrs=["x", "x_debug"]
                )

                mark_edges(G_t, causal_subgraphs[j], spu_subgraphs[j])

                G_t_filt = remove_from_graph(G_t, "inv")
                num_elem = mark_frontier(G_t, G_t_filt)
                if num_elem == 0:
                    print("\nZero frontier here2 ", i)
                    # draw(G_t, name=f"debug2_graph_{i}")
                    # draw(G_t_filt, name=f"debug2_filtgraph_{i}")
                    continue

                G_union = random_attach(G_filt, G_t_filt)
                eval_samples.append(G_union)
                belonging.append(i)
                c += 1

                if debug:
                    draw(G_t_filt, name=f"spu_graph_{j}")
                    draw(G_union, name=f"joined_graph_{i}_{j}")


        ##
        # Compute new prediction and evaluate KL
        ##

        dataset = CustomDataset("", eval_samples, belonging)
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
            
        pbar = tqdm(loader, desc=f'Eval intervened graphs', total=len(loader), **pbar_setting)
        preds_eval = []
        belonging = []
        for data in pbar:
            data: Batch = data.to(self.config.device)
            output = self.model.log_probs(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm)
            preds_eval.extend(output.detach().cpu().numpy().tolist())
            belonging.extend(data.belonging.detach().cpu().numpy().tolist())
        
        # assert len(preds_eval) == len(preds_ori) * self.config.expval_budget, f"{len(preds_eval)}_{len(preds_ori)}"

        preds_eval = torch.tensor(preds_eval)
        preds_ori = torch.tensor(preds_ori)
        preds_ori = preds_ori.repeat_interleave(self.config.expval_budget, dim=0)

        print(preds_ori.shape, preds_eval.shape)

        div = torch.nn.KLDivLoss(reduction="none", log_target=True)(preds_ori, preds_eval).sum(-1)
        div_aggr = scatter_mean(div, torch.tensor(belonging), dim=0)
        print(div_aggr)
        return div_aggr.mean()




    # @torch.no_grad()
    # def compute_sufficiency(self, split: str, debug=True):
    #     """
    #         Algorithm:
    #         1. compute and store P(Y|G')
    #         2. extract explanation and complement for each sample
    #         3. for each sample (or subset thereof)
    #             3.1 for a certain budget
    #                 3.1.1 replace its complement with the complement of another sample
    #                 3.1.2 compute P(Y|G')
    #                 3.1.3 compute d_i = d(P(Y|G'), P(Y|G))
    #         4. average d_i across all samples
    #     """
    #     colors = {
    #         "inv": "green",
    #         "spu": "blue",
    #         "added": "red"
    #     }
    #     print(f"#D#Computing SUFF over {split}")
    #     print(self.loader[split].dataset)

    #     loader = DataLoader(self.loader[split].dataset[:5], batch_size=1, shuffle=False)

    #     self.model.eval()

    #     pbar_setting["disable"] = False
    #     pbar = tqdm(loader, desc=f'Eval {split.capitalize()}', total=len(loader), **pbar_setting)
    #     preds_ori = []
    #     graphs = []
    #     causal_subgraphs = []
    #     compl_subgraphs = []
    #     causal_xs = []
    #     labels = []
    #     for data in pbar:
    #         data: Batch = data.to(self.config.device)
    #         output = self.model.probs(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm)
    #         preds_ori.append(output.detach().cpu().numpy().tolist())
    #         labels.extend(data.y.detach().cpu().numpy().tolist())

    #         (causal_edge_index, causal_x, causal_batch, causal_edge_weight), \
    #             (spu_edge_index, spu_x, spu_batch, causal_edge_weight) = self.model.get_subgraph(data=data, edge_weight=None, ood_algorithm=self.ood_algorithm, do_relabel=False)

    #         graphs.append(data.edge_index.detach().cpu())
    #         causal_subgraphs.append(causal_edge_index.detach().cpu())
    #         compl_subgraphs.append(spu_edge_index.detach().cpu())
    #         causal_xs.append(causal_x.detach().cpu())

    #     labels = np.array(labels)

    #     if debug and False:
    #         causal_subgraphs[0] = torch.tensor([
    #             [14,15],[15,14],[15,16],[16,15],[16,17],[17,16],[17,18],[18,17],[18,14],[14,18],
    #         ]).T
    #         compl_subgraphs[0] = torch.tensor([
    #             [0,7],[7,0],[7,8],[8,7],[8,1],[1,8],[1,0],[0,1],
    #             [8,9],[9,8],[9,10],[10,9],[10,1],[1,10],
    #             [9,2],[2,9],[2,3],[3,2],[3,10],[10,3],
    #             [3,4],[4,3],[4,11],[11,4],[11,10],[10,11],
    #             [4,5],[5,4],[5,12],[12,5],[12,11],[11,12],
    #             [5,6],[6,5],[6,13],[13,6],[13,12],[12,13],
    #             [13,14],[14,13],
    #         ]).T
        
    #     for i in range(1): #insert max budget
    #         G_s_inv = to_networkx(
    #             Data(edge_index=causal_subgraphs[i], x=causal_xs[i]), 
    #             node_attrs=["x"]
    #         )
    #         print(causal_xs[i])
    #         print(nx.get_node_attributes(G_s_inv, 'x'))
            
    #         G_s_inv.remove_nodes_from(list(nx.isolates(G_s_inv)))
            
    #         if debug:
    #             G_s = to_networkx(Data(edge_index=graphs[i]))
    #             pos = nx.kamada_kawai_layout(G_s)
    #             nx.draw(G_s, with_labels = True, pos=pos, edge_color=["green" if e in G_s_inv.edges() else "blue" for e in G_s.edges()],)
    #             plt.savefig(f'GOOD/kernel/pipelines/plots/graph_{i}.png')
    #             plt.close()
    #             nx.draw(G_s_inv, with_labels = True, pos=pos, edge_color="green")
    #             plt.savefig(f'GOOD/kernel/pipelines/plots/inv_graph_{i}.png')
    #             plt.close()
            
    #         idxs = np.random.permutation(np.arange(5)[labels == labels[i]]) #pick from same class
    #         for j in idxs[:5]: #insert max budget
    #             G_test = G_s_inv.copy()
    #             # j = j + 0
                
    #             sort_compl_subgraphs, _ = torch.sort(compl_subgraphs[i].T, dim=1)
    #             sort_compl_subgraphs, count_sort_compl_subgraphs = sort_compl_subgraphs.unique(dim=0, return_counts=True)
    #             common = torch.cat((causal_subgraphs[i].unique(), sort_compl_subgraphs[count_sort_compl_subgraphs > 1].unique())).unique(return_counts=True) # should contain only elments that have both incoming and outgoing edges in G_spu
    #             frontier_nodes = common[0][common[1] > 1] 
    #             if debug:
    #                 print("Frontier nodes = ", frontier_nodes)

    #             G_t_spu = to_networkx(Data(edge_index=compl_subgraphs[j]), )
    #             G_t_spu.remove_edges_from([(v,u) for v,u in G_t_spu.edges() if not G_t_spu.has_edge(u,v)])
    #             G_t_spu.remove_nodes_from(list(nx.isolates(G_t_spu)))

    #             if debug:
    #                 pos = nx.kamada_kawai_layout(to_networkx(Data(edge_index=graphs[j]), ))
    #                 nx.draw(G_t_spu, with_labels = True, pos=pos)
    #                 plt.savefig(f'GOOD/kernel/pipelines/plots/spu_graph_{j}.png')
    #                 plt.close()

    #             # join graphs
    #             upper_num_nodes = max(G_test.nodes())
    #             G_t_spu = nx.relabel_nodes(G_t_spu, {n: upper_num_nodes + i + 1 for i, n in enumerate(G_t_spu.nodes())})
    #             nx.set_edge_attributes(G_t_spu, name="origin", values="spu")
    #             nx.set_edge_attributes(G_test, name="origin", values="inv")
    #             G_test = nx.union(G_test, G_t_spu)
    #             for n in frontier_nodes:
    #                 # pick random node v in G_t_spu
    #                 # add edge (u,v) and (v,u) maybe
    #                 v = randint(1, len(G_t_spu.nodes()) + 1)
    #                 G_test.add_edge(n.item(), upper_num_nodes + v, origin="added")
    #                 G_test.add_edge(upper_num_nodes + v, n.item(), origin="added")

    #             if debug:
    #                 nx.draw(
    #                     G_test,
    #                     with_labels = True,
    #                     edge_color=list(map(lambda x: colors[x], nx.get_edge_attributes(G_test,'origin').values())),
    #                     pos=nx.kamada_kawai_layout(G_test)
    #                 )
    #                 plt.savefig(f'GOOD/kernel/pipelines/plots/joined_graph_{i}_{j}.png')
    #                 plt.close()






    #     print(preds_ori[:2], labels[:2])



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
            suff = self.compute_sufficiency2("id_val")
        else:
            suff = None

        print(f'{split.capitalize()} {self.config.metric.score_name}: {stat["score"]:.4f}'
              f'{split.capitalize()} Loss: {stat["loss"]:.4f}')

        self.model.train()

        return {'score': stat['score'], 'loss': stat['loss'], 'suff': suff}

    def load_task(self, load_param=False):
        r"""
        Launch a training or a test.
        """
        if self.task == 'train':
            self.train()
            return None, None
        elif self.task == 'test':
            # config model
            print('#D#Config model and output the best checkpoint info...')
            test_score, test_loss = self.config_model('test', load_param=load_param)
            return test_score, test_loss

    def config_model(self, mode: str, load_param=False):
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
                    self.model.load_state_dict(ckpt['state_dict'])
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

# Beyond Topological Self-Explainable GNNs: A Formal Explainability Perspective

Official supplemental material containing the sourcecode for the ICML2025 submission *Beyond Topological Self-Explainable GNNs: A Formal Explainability Perspective*.

This codebase is build on top of the one provided by [GOOD](https://github.com/divelab/GOOD).

The structure is as follows.

## Model Implementations

Modle implementations can be found here `GOOD\networks\models`.
Basic implementations of classifiers, shared modules, and Logic Explained Networks (referred to as *ConceptClassifier* in the sourcecode) are available in `GOOD\networks\models\Classifiers.py`.

## Training Details

The file `GOOD\ood_algorithms\algorithms\BaseOOD.py` contains the basic training logic for each model. Then, for specific training protocols of each architecture, please refer to the corresponding file in the same folder. Note that configurations of additional models not tested in this work may be present, as they are inhereted from the original GOOD implementation.

In this work, only the following models are considered:
 - GIN (ERM - Empirical Risk Minimization)
 - GSAT
 - GiSST
 - SMGNN

## Configurations files

Configuration files and hyper-parameter details for each experiment are available in `configs/final_configs`.

## Datasets

Dataset implementations are provided in `GOOD\data\good_datasets`.

For generating MNIST75sp the MNIST dataset, please refer to the [original paper](https://github.com/bknyaz/graph_attention_pool/tree/master/scripts). We included in our codebase the file `scripts\extract_mnist_superpixels.py` for ease of reproduction.

Explicit datasets are not included as they exceed the size limit. When downloading raw datasets, please place them in `storage/datasets`.

## Checkpoints

Checkpoints will be made available as a separata DRIVE folder, as they exceed the ICML size limit. By default, checkpoints are saved in `storage/checkpoints`.

## Reproducing the Experiments

We report in the following the explicit commands used to reproduce our experiments. The `--task` parameter regulates the behaviour of the code, and can be set as follows:

- `test`: Evaluate the model
- `train`: Train the model
- `plot_explanations`: Plot examples of explanations for the first 25 samples in the dataset
- `plot_global`: Plot the decision boundary of the side-channel linear interpretable model
- `plot_panel`: Plot the histogram distribution of explanatory scores

Also, the Dual-Channel variant of each SEGNN is trained by specifiying the `--global_side_channel` parameter. Possible choices are:

- `simple_concept2temperature`: B(LEN)
- `simple_concept2`: LEN
- `simple_concept2discrete`: Discrete LEN
- `simple_linear`: Linear
- `simple_mlp`: MLP
- `simple_product`: Product T-norm
- `simple_godel`: Godel T-norm

```shell
# TopoFeature (GIN)
goodtg --config_path final_configs/TopoFeature/basis/no_shift/ERM.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --use_norm none
# TopoFeature (GSAT)
goodtg --config_path final_configs/TopoFeature/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 0 --use_norm none
# TopoFeature (SMGNN)
goodtg --config_path final_configs/TopoFeature/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --global_pool sum --gpu_idx 0 --extra_param True 10 0.01 --ood_param 0.001 --lr_filternode 0.001 --lr 0.001 --use_norm none
# TopoFeature (DC-GSAT)
goodtg --config_path final_configs/TopoFeature/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1 --use_norm none --global_side_channel simple_concept2temperature
# TopoFeature (DC-SMGNN)
goodtg --config_path final_configs/TopoFeature/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --global_pool sum --gpu_idx 0 --global_side_channel simple_concept2temperature --extra_param True 10 0.01 --ood_param 0.001 --lr_filternode 0.001 --lr 0.001 --use_norm none
# TopoFeature (DC-SMGNN Discrete Gumbel)
goodtg --config_path final_configs/TopoFeature/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --global_pool sum --gpu_idx 1 --global_side_channel simple_concept2discrete --extra_param True 10 0.01 --ood_param 0.001 --lr_filternode 0.001 --lr 0.001 --use_norm none
# TopoFeature (DC-SMGNN Linear)
goodtg --config_path final_configs/TopoFeature/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --global_pool sum --gpu_idx 1 --global_side_channel simple_linear --extra_param True 10 0.01 --ood_param 0.001 --lr_filternode 0.001 --lr 0.001 --use_norm none
# TopoFeature (DC-SMGNN MLP)
goodtg --config_path final_configs/TopoFeature/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --global_pool sum --gpu_idx 1 --global_side_channel simple_mlp --extra_param True 10 0.01 --ood_param 0.001 --lr_filternode 0.001 --lr 0.001 --use_norm none


# BAColor (GIN)
goodtg --config_path final_configs/BAColor/basis/no_shift/ERM.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 0
# BAColor (GSAT)
goodtg --config_path final_configs/BAColor/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 0 --use_norm none
# BAColor (SMGNN)
goodtg --config_path final_configs/BAColor/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --global_pool sum --gpu_idx 1 --extra_param True 10 0.001 --ood_param 0.001 --lr_filternode 0.001 --lr 0.001 --use_norm none
# BAColor (DC-GSAT)
goodtg --config_path final_configs/BAColor/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1 --global_side_channel simple_concept2temperature --lr_filternode 0.001 --lr 0.001 --use_norm none
# BAColor (DC-SMGNN)
goodtg --config_path final_configs/BAColor/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --global_pool sum --gpu_idx 0 --global_side_channel simple_concept2temperature --extra_param True 10 0.001 --ood_param 0.001 --lr_filternode 0.001 --lr 0.001 --use_norm none


# Motif (GIN)
goodtg --config_path final_configs/GOODMotif/basis/covariate/ERM.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --gpu_idx 1 --use_norm none --average_edge_attn mean
goodtg --config_path final_configs/GOODMotif/basis/covariate/ERM.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --gpu_idx 1 --use_norm bn --average_edge_attn mean
# Motif (GSAT)
goodtg --config_path final_configs/GOODMotif/basis/covariate/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1 --use_norm bn
# Motif (SMGNN)
goodtg --config_path final_configs/GOODMotif/basis/covariate/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1 --use_norm bn
# Motif (DC-GSAT)
goodtg --config_path final_configs/GOODMotif/basis/covariate/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1 --use_norm bn --global_side_channel simple_concept2temperature
# Motif (DC-SMGNN)
goodtg --config_path final_configs/GOODMotif/basis/covariate/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1 --global_side_channel simple_concept2temperature


# SST2 (GIN)
goodtg --config_path final_configs/GOODSST2/length/covariate/ERM.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --gpu_idx 1  --use_norm none
# SST2 (GiSST)
goodtg --config_path final_configs/GOODSST2/length/covariate/GiSST.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1  --use_norm none --mitigation_sampling raw
# SST2 (GSAT)
goodtg --config_path final_configs/GOODSST2/length/covariate/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 0  --use_norm none --mitigation_sampling raw
# SST2 (SMGNN)
goodtg --config_path final_configs/GOODSST2/length/covariate/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1  --use_norm none --mitigation_sampling raw
# SST2 (DC_GiSST)
goodtg --config_path final_configs/GOODSST2/length/covariate/GiSST.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1  --use_norm none --mitigation_sampling raw --global_side_channel simple_concept2temperature
# SST2 (DC_GSAT)
goodtg --config_path final_configs/GOODSST2/length/covariate/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1  --use_norm none --mitigation_sampling raw --global_side_channel simple_concept2temperature
# SST2 (DC-SMGNN)
goodtg --config_path final_configs/GOODSST2/length/covariate/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1  --use_norm none --mitigation_sampling raw --global_side_channel simple_concept2temperature


# AIDS (GIN)
goodtg --config_path final_configs/AIDS/basis/no_shift/ERM.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --use_norm none
# AIDS (GiSST)
goodtg --config_path final_configs/AIDS/basis/no_shift/GiSST.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm none --average_edge_attn mean --weight_decay 0.001 --channel_weight_decay 0.001 --combinator_weight_decay 0.001
# AIDS (GSAT)
goodtg --config_path final_configs/AIDS/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --use_norm none --average_edge_attn mean
# AIDS (SMGNN)
goodtg --config_path final_configs/AIDS/basis/no_shift/SMGNN.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm none --average_edge_attn mean
# AIDS (DC-GiSST)
goodtg --config_path final_configs/AIDS/basis/no_shift/GiSST.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --use_norm none --average_edge_attn mean --global_side_channel simple_concept2temperature
# AIDS (DC-GSAT)
goodtg --config_path final_configs/AIDS/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --use_norm none --average_edge_attn mean --global_side_channel simple_concept2temperature
# AIDS (DC-SMGNN)
goodtg --config_path final_configs/AIDS/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 0 --global_side_channel simple_concept2temperature  --use_norm none 


# AIDSC1 (GIN)
goodtg --config_path final_configs/AIDSC1/basis/no_shift/ERM.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test
# AIDSC1 (GiSST)
goodtg --config_path final_configs/AIDSC1/basis/no_shift/GiSST.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm none --average_edge_attn mean
# AIDSC1 (GSAT)
goodtg --config_path final_configs/AIDSC1/basis/no_shift/GSAT.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm none --average_edge_attn mean --gpu_idx 0
# AIDSC1 (SMGNN)
goodtg --config_path final_configs/AIDSC1/basis/no_shift/SMGNN.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm none --average_edge_attn mean --gpu_idx 0
# AIDSC1 (DC-GiSST)
goodtg --config_path final_configs/AIDSC1/basis/no_shift/GiSST.yaml --task test --seeds "1/3/4/5/6/7/8/9/10" --use_norm none --average_edge_attn mean --gpu_idx 1 --global_side_channel simple_concept2temperature
# AIDSC1 (DC-GSAT)
goodtg --config_path final_configs/AIDSC1/basis/no_shift/GSAT.yaml --task test --seeds "1/2/3/5/6/7/8/9" --use_norm none --average_edge_attn mean --gpu_idx 1 --global_side_channel simple_concept2temperature
# AIDSC1 (DC-SMGNN)
goodtg --config_path final_configs/AIDSC1/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9" --task test --average_edge_attn mean --gpu_idx 0 --global_side_channel simple_concept2temperature  --use_norm none 


# MUTAG (GIN)
goodtg --config_path final_configs/MUTAG/basis/no_shift/ERM.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm none --gpu_idx 1
# MUTAG (GiSST)
goodtg --config_path final_configs/MUTAG/basis/no_shift/GiSST.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --gpu_idx 1 --use_norm none --average_edge_attn mean
# MUTAG (GSAT)
goodtg --config_path final_configs/MUTAG/basis/no_shift/GSAT.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --average_edge_attn mean --use_norm none
# MUTAG (SMGNN)
goodtg --config_path final_configs/MUTAG/basis/no_shift/SMGNN.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --average_edge_attn mean --use_norm none
# MUTAG (DC-GiSST)
goodtg --config_path final_configs/MUTAG/basis/no_shift/GiSST.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --gpu_idx 1 --use_norm none --average_edge_attn mean --global_side_channel simple_concept2temperature
# MUTAG (DC-GSAT)
goodtg --config_path final_configs/MUTAG/basis/no_shift/GSAT.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --average_edge_attn mean --use_norm none --global_side_channel simple_concept2temperature --gpu_idx 1
# MUTAG (DC-SMGNN)
goodtg --config_path final_configs/MUTAG/basis/no_shift/SMGNN.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --average_edge_attn mean --use_norm none --global_side_channel simple_concept2temperature


# BBBP (GIN)
goodtg --config_path final_configs/BBBP/basis/no_shift/ERM.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm bn --gpu_idx 0 --global_pool mean
# BBBP (GiSST)
goodtg --config_path final_configs/BBBP/basis/no_shift/GiSST.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm bn --gpu_idx 2 --average_edge_attn mean
# BBBP (GSAT)
goodtg --config_path final_configs/BBBP/basis/no_shift/GSAT.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm bn --gpu_idx 1 --average_edge_attn mean
# BBBP (SMGNN)
goodtg --config_path final_configs/BBBP/basis/no_shift/SMGNN.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm bn --gpu_idx 0 --average_edge_attn mean
# BBBP (DC-GiSST)
goodtg --config_path final_configs/BBBP/basis/no_shift/GiSST.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm bn --gpu_idx 2 --average_edge_attn mean --global_side_channel simple_concept2temperature
# BBBP (DC-GSAT)
goodtg --config_path final_configs/BBBP/basis/no_shift/GSAT.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm bn --average_edge_attn mean --global_side_channel simple_concept2temperature
# BBBP (DC-SMGNN)
goodtg --config_path final_configs/BBBP/basis/no_shift/SMGNN.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm bn --gpu_idx 0 --average_edge_attn mean --global_side_channel simple_concept2temperature

# MNIST (GIN)
goodtg --config_path final_configs/MNIST/basis/no_shift/ERM.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm bn --gpu_idx 0
# MNIST (GiSST)
goodtg --config_path final_configs/MNIST/basis/no_shift/GiSST.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm none --average_edge_attn mean --gpu_idx 1
# MNIST (GSAT)
goodtg --config_path final_configs/MNIST/basis/no_shift/GSAT.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm bn --average_edge_attn mean --gpu_idx 1
# MNIST (SMGNN)
goodtg --config_path final_configs/MNIST/basis/no_shift/SMGNN.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm none --gpu_idx 1 --average_edge_attn mean
# MNIST (DC-GiSST)
goodtg --config_path final_configs/MNIST/basis/no_shift/GiSST.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm none --average_edge_attn mean --gpu_idx 1 --global_side_channel simple_concept2temperature
# MNIST (DC-GSAT)
goodtg --config_path final_configs/MNIST/basis/no_shift/GSAT.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm bn --average_edge_attn mean --gpu_idx 1 --global_side_channel simple_concept2temperature
# MNIST (DC-SMGNN)
goodtg --config_path final_configs/MNIST/basis/no_shift/SMGNN.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm none --average_edge_attn mean --gpu_idx 1 --global_side_channel simple_concept2temperature
```

## Computing FAITH 

To compute the faithfulness score refer to `scripts\faithfulness_topofeature.sh` and `scripts\faithfulness_motif.sh`.
Files will be saved in `storage\metric_values`. To easily visialize them, swicth the `--task` parameter to `print_faith`.

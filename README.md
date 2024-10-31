# WORK IN PROGRESS


## Table of contents

* [Overview](#overview)
* [Installation](#installation)
* [Run LECI](#run-leci)
* [Citing LECI](#citing-leci)
* [License](#license)
* [Contact](#contact)

## Relevant Commands

```shell
# TopoFeature (SMGNN)
goodtg --config_path final_configs/TopoFeature/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --global_pool sum --gpu_idx 0 --global_side_channel simple_concept2temperature --extra_param True 10 0.01 --ood_param 0.001 --lr_filternode 0.001 --lr 0.001 --use_norm none

# TopoFeature (GSAT)
goodtg --config_path final_configs/TopoFeature/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 0 --use_norm none

# BAColor (SMGNN)
goodtg --config_path final_configs/BAColor/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --global_pool sum --gpu_idx 0 --global_side_channel simple_concept2temperature --extra_param True 10 0.001 --ood_param 0.001 --lr_filternode 0.001 --lr 0.001 --use_norm none

# Motif (SMGNN)
 goodtg --config_path final_configs/GOODMotif/basis/covariate/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --global_pool mean --gpu_idx 1 --global_side_channel simple_productscaled --extra_param True 10 0.01 --ood_param 0.001 --lr_filternode 0.005 --lr 0.005 --use_norm bn --channel_weight_decay 0.0 --train_bs 64

# Twitter (SMGNN)
goodtg --config_path final_configs/GOODTwitter/length/covariate/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --global_pool mean --gpu_idx 0 --extra_param True 10 0.01 --ood_param 0.001 --lr_filternode 0.001 --lr 0.001 --use_norm none --mitigation_sampling raw

# Twitter (GSAT)
goodtg --config_path final_configs/GOODTwitter/length/covariate/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 0  --use_norm none --mitigation_sampling raw

# SST2 (GSAT)
goodtg --config_path final_configs/GOODSST2/length/covariate/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 0  --use_norm none --mitigation_sampling raw

# AIDS (SMGNN)
goodtg --config_path final_configs/AIDS/length/covariate/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 0 --global_side_channel simple_concept2temperature  --use_norm none 
```
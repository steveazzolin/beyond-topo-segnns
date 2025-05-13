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
# TopoFeature (ERM)
goodtg --config_path final_configs/TopoFeature/basis/no_shift/ERM.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --use_norm none
# TopoFeature (GiSST)
goodtg --config_path final_configs/TopoFeature/basis/no_shift/GiSST.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 0 --use_norm bn
# TopoFeature (GSAT)
goodtg --config_path final_configs/TopoFeature/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 0 --use_norm none --weight_decay 0.0 # (WARNING: currently on Excel)
goodtg --config_path final_configs/TopoFeature/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 0 --use_norm none
# TopoFeature (SMGNN)
goodtg --config_path final_configs/TopoFeature/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --global_pool sum --gpu_idx 0 --extra_param True 10 0.01 --ood_param 0.001 --lr_filternode 0.001 --lr 0.001 --use_norm none
# TopoFeature (GL-GiSST)
goodtg --config_path final_configs/TopoFeature/basis/no_shift/GiSST.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1 --use_norm bn --global_side_channel simple_concept2temperature
# TopoFeature (GL-GSAT)
goodtg --config_path final_configs/TopoFeature/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1 --use_norm none --global_side_channel simple_concept2temperature
# TopoFeature (GL-SMGNN)
goodtg --config_path final_configs/TopoFeature/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --global_pool sum --gpu_idx 0 --global_side_channel simple_concept2temperature --extra_param True 10 0.01 --ood_param 0.001 --lr_filternode 0.001 --lr 0.001 --use_norm none
# TopoFeature (GL-SMGNN Discrete Gumbel)
goodtg --config_path final_configs/TopoFeature/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --global_pool sum --gpu_idx 1 --global_side_channel simple_concept2discrete --extra_param True 10 0.01 --ood_param 0.001 --lr_filternode 0.001 --lr 0.001 --use_norm none
# TopoFeature (GL-SMGNN Linear)
goodtg --config_path final_configs/TopoFeature/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --global_pool sum --gpu_idx 1 --global_side_channel simple_linear --extra_param True 10 0.01 --ood_param 0.001 --lr_filternode 0.001 --lr 0.001 --use_norm none
# TopoFeature (GL-SMGNN MLP)
goodtg --config_path final_configs/TopoFeature/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --global_pool sum --gpu_idx 1 --global_side_channel simple_mlp --extra_param True 10 0.01 --ood_param 0.001 --lr_filternode 0.001 --lr 0.001 --use_norm none


# BAColor (ERM)
goodtg --config_path final_configs/BAColor/basis/no_shift/ERM.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 0
# BAColor (GiSST)
goodtg --config_path final_configs/BAColor/basis/no_shift/GiSST.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 0 --use_norm bn
# BAColor (GSAT)
goodtg --config_path final_configs/BAColor/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 0 --use_norm none
# BAColor (GSAT) r=0.5 for rebuttal
goodtg --config_path final_configs/BAColor/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 0 --use_norm none --extra_param True 10 0.5
# BAColor (GSAT) r=0.7 for rebuttal
goodtg --config_path final_configs/BAColor/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 0 --use_norm none --extra_param True 10 0.7
# BAColor (SMGNN)
goodtg --config_path final_configs/BAColor/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --global_pool sum --gpu_idx 1 --extra_param True 10 0.001 --ood_param 0.001 --lr_filternode 0.001 --lr 0.001 --use_norm none
# BAColor (DIR)

# BAColor (GL-GiSST)
goodtg --config_path final_configs/BAColor/basis/no_shift/GiSST.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1 --global_side_channel simple_concept2temperature --lr_filternode 0.01 --lr 0.001 --use_norm bn --weight_decay 0.01 --channel_weight_decay 0.01
# BAColor (GL-GSAT)
goodtg --config_path final_configs/BAColor/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1 --global_side_channel simple_concept2temperature --lr_filternode 0.001 --lr 0.001 --use_norm none
# BAColor (GL-GSAT) r=0.5 for rebuttal
goodtg --config_path final_configs/BAColor/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1 --global_side_channel simple_concept2temperature --lr_filternode 0.001 --lr 0.001 --use_norm none --extra_param True 10 0.5
# BAColor (GL-GSAT) r=0.7 for rebuttal
goodtg --config_path final_configs/BAColor/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1 --global_side_channel simple_concept2temperature --lr_filternode 0.001 --lr 0.001 --use_norm none --extra_param True 10 0.7
# BAColor (GL-SMGNN)
goodtg --config_path final_configs/BAColor/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --global_pool sum --gpu_idx 0 --global_side_channel simple_concept2temperature --extra_param True 10 0.001 --ood_param 0.001 --lr_filternode 0.001 --lr 0.001 --use_norm none
# BAColor (GL-DIR)
goodtg --config_path final_configs/BAColor/basis/no_shift/DIR.yaml --task train --seeds "6" --use_norm none --gpu_idx 0 --average_edge_attn mean --global_side_channel simple_concept2temperature


# Motif (ERM)
goodtg --config_path final_configs/GOODMotif/basis/covariate/ERM.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --gpu_idx 1 --use_norm none --average_edge_attn mean
goodtg --config_path final_configs/GOODMotif/basis/covariate/ERM.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --gpu_idx 1 --use_norm bn --average_edge_attn mean
# Motif (GiSST)
goodtg --config_path final_configs/GOODMotif/basis/covariate/GiSST.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1 --use_norm bn
# Motif (GSAT)
goodtg --config_path final_configs/GOODMotif/basis/covariate/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1 --use_norm bn
# Motif (SMGNN)
goodtg --config_path final_configs/GOODMotif/basis/covariate/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1 --use_norm bn
# Motif (GL-GiSST)
goodtg --config_path final_configs/GOODMotif/basis/covariate/GiSST.yaml --seeds "2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1 --use_norm bn --global_side_channel simple_concept2temperature
# Motif (GL-GSAT)
goodtg --config_path final_configs/GOODMotif/basis/covariate/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1 --use_norm bn --global_side_channel simple_concept2temperature
# Motif (GL-SMGNN)
goodtg --config_path final_configs/GOODMotif/basis/covariate/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1 --global_side_channel simple_concept2temperature


# REBUTTAL HYPER-TUNING MOTIF
# Motif GL-GSAT hidden size of (B)LEN: 15hiddenblen | 64hiddenblen
# Motif GL-GSAT r parameter of GSAT: rebuttal_0.1end_temp_ | rebuttal_0.5end_temp_
# Motif GL-GSAT r parameter of GSAT: rebuttal_r0.5 | rebuttal_r0.9
# Motif GL-GSAT r parameter of GSAT: rebuttal_1blenlayer | rebuttal_3blenlayer
goodtg --config_path final_configs/GOODMotif/basis/covariate/GSAT.yaml --seeds "1/2/3/4/5" --task train --average_edge_attn mean --gpu_idx 0 --use_norm bn --global_side_channel simple_concept2temperature --train_bs 256 --extra_param True 20 0.9


# Twitter (SMGNN)
# goodtg --config_path final_configs/GOODTwitter/length/covariate/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --global_pool mean --gpu_idx 0 --extra_param True 10 0.01 --ood_param 0.001 --lr_filternode 0.001 --lr 0.001 --use_norm none --mitigation_sampling raw
# Twitter (GSAT)
# goodtg --config_path final_configs/GOODTwitter/length/covariate/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 0  --use_norm none --mitigation_sampling raw


# SST2 (ERM)
goodtg --config_path final_configs/GOODSST2/length/covariate/ERM.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --gpu_idx 1  --use_norm none
# SST2 (GiSST)
goodtg --config_path final_configs/GOODSST2/length/covariate/GiSST.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1  --use_norm none --mitigation_sampling raw
# SST2 (GSAT)
goodtg --config_path final_configs/GOODSST2/length/covariate/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 0  --use_norm none --mitigation_sampling raw
# SST2 (SMGNN)
goodtg --config_path final_configs/GOODSST2/length/covariate/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1  --use_norm none --mitigation_sampling raw
# SST2 (DIR)
goodtg --config_path final_configs/GOODSST2/length/covariate/DIR.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --use_norm none --gpu_idx 0 --average_edge_attn mean
# SST2 (GL_GiSST)
goodtg --config_path final_configs/GOODSST2/length/covariate/GiSST.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1  --use_norm none --mitigation_sampling raw --global_side_channel simple_concept2temperature
# SST2 (GL_GSAT)
goodtg --config_path final_configs/GOODSST2/length/covariate/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1  --use_norm none --mitigation_sampling raw --global_side_channel simple_concept2temperature
# SST2 (GL-SMGNN)
goodtg --config_path final_configs/GOODSST2/length/covariate/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 1  --use_norm none --mitigation_sampling raw --global_side_channel simple_concept2temperature
# SST2 (GL-DIR)
goodtg --config_path final_configs/GOODSST2/length/covariate/DIR.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --use_norm none --gpu_idx 0 --average_edge_attn mean  --global_side_channel simple_concept2temperature


# AIDS (ERM)
goodtg --config_path final_configs/AIDS/basis/no_shift/ERM.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --use_norm none
# AIDS (GiSST)
goodtg --config_path final_configs/AIDS/basis/no_shift/GiSST.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm none --average_edge_attn mean --weight_decay 0.001 --channel_weight_decay 0.001 --combinator_weight_decay 0.001
# AIDS (GSAT)
goodtg --config_path final_configs/AIDS/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --use_norm none --average_edge_attn mean
# AIDS (SMGNN)
goodtg --config_path final_configs/AIDS/basis/no_shift/SMGNN.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm none --average_edge_attn mean
# AIDS (DIR)
goodtg --config_path final_configs/AIDS/basis/no_shift/DIR.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm none --gpu_idx 0 --average_edge_attn mean
# AIDS (GL-GiSST)
goodtg --config_path final_configs/AIDS/basis/no_shift/GiSST.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --use_norm none --average_edge_attn mean --global_side_channel simple_concept2temperature
# AIDS (GL-GSAT)
goodtg --config_path final_configs/AIDS/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --use_norm none --average_edge_attn mean --global_side_channel simple_concept2temperature
# AIDS (GL-SMGNN)
goodtg --config_path final_configs/AIDS/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --average_edge_attn mean --gpu_idx 0 --global_side_channel simple_concept2temperature  --use_norm none 
# AIDS (GL-DIR)
goodtg --config_path final_configs/AIDS/basis/no_shift/DIR.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm none --gpu_idx 0 --average_edge_attn mean --global_side_channel simple_concept2temperature


# AIDSC1 (ERM)
goodtg --config_path final_configs/AIDSC1/basis/no_shift/ERM.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test
# AIDSC1 (GiSST)
goodtg --config_path final_configs/AIDSC1/basis/no_shift/GiSST.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm none --average_edge_attn mean
# AIDSC1 (GSAT)
goodtg --config_path final_configs/AIDSC1/basis/no_shift/GSAT.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm none --average_edge_attn mean --gpu_idx 0
# AIDSC1 (SMGNN)
goodtg --config_path final_configs/AIDSC1/basis/no_shift/SMGNN.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm none --average_edge_attn mean --gpu_idx 0
# AIDSC1 (GL-GiSST)
goodtg --config_path final_configs/AIDSC1/basis/no_shift/GiSST.yaml --task test --seeds "1/3/4/5/6/7/8/9/10" --use_norm none --average_edge_attn mean --gpu_idx 1 --global_side_channel simple_concept2temperature
# AIDSC1 (GL-GSAT)
goodtg --config_path final_configs/AIDSC1/basis/no_shift/GSAT.yaml --task test --seeds "1/2/3/5/6/7/8/9" --use_norm none --average_edge_attn mean --gpu_idx 1 --global_side_channel simple_concept2temperature
# AIDSC1 (GL-SMGNN)
goodtg --config_path final_configs/AIDSC1/basis/no_shift/SMGNN.yaml --seeds "1/2/3/4/5/6/7/8/9" --task test --average_edge_attn mean --gpu_idx 0 --global_side_channel simple_concept2temperature  --use_norm none 


# MUTAG (ERM)
goodtg --config_path final_configs/MUTAG/basis/no_shift/ERM.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm none --gpu_idx 1
# MUTAG (GiSST)
goodtg --config_path final_configs/MUTAG/basis/no_shift/GiSST.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --gpu_idx 1 --use_norm none --average_edge_attn mean
# MUTAG (GSAT)
goodtg --config_path final_configs/MUTAG/basis/no_shift/GSAT.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --average_edge_attn mean --use_norm none
# MUTAG (SMGNN)
goodtg --config_path final_configs/MUTAG/basis/no_shift/SMGNN.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --average_edge_attn mean --use_norm none
# MUTAG (GL-GiSST)
goodtg --config_path final_configs/MUTAG/basis/no_shift/GiSST.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --gpu_idx 1 --use_norm none --average_edge_attn mean --global_side_channel simple_concept2temperature
# MUTAG (GL-GSAT)
goodtg --config_path final_configs/MUTAG/basis/no_shift/GSAT.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --average_edge_attn mean --use_norm none --global_side_channel simple_concept2temperature --gpu_idx 1
# MUTAG (GL-SMGNN)
goodtg --config_path final_configs/MUTAG/basis/no_shift/SMGNN.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --average_edge_attn mean --use_norm none --global_side_channel simple_concept2temperature


# MUTAG0 (ERM)
goodtg --config_path final_configs/MUTAG0/basis/no_shift/ERM.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm none --gpu_idx 1
# MUTAG0 (GiSST)
goodtg --config_path final_configs/MUTAG0/basis/no_shift/GiSST.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --average_edge_attn mean --use_norm none
# MUTAG0 (GSAT)
goodtg --config_path final_configs/MUTAG0/basis/no_shift/GSAT.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --average_edge_attn mean --use_norm none
# MUTAG0 (SMGNN)
goodtg --config_path final_configs/MUTAG0/basis/no_shift/GSAT.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --average_edge_attn mean --use_norm none
# MUTAG0 (DIR)
goodtg --config_path final_configs/MUTAG/basis/no_shift/DIR.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --average_edge_attn mean --use_norm none
# MUTAG0 (GL-GiSST)
goodtg --config_path final_configs/MUTAG0/basis/no_shift/GiSST.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm none --average_edge_attn mean --global_side_channel simple_concept2temperature
# MUTAG0 (GL-GSAT)
goodtg --config_path final_configs/MUTAG0/basis/no_shift/GSAT.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm none --average_edge_attn mean --global_side_channel simple_concept2temperature
# MUTAG0 (GL-SMGNN)
goodtg --config_path final_configs/MUTAG0/basis/no_shift/SMGNN.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --average_edge_attn mean --use_norm none --global_side_channel simple_concept2temperature
# MUTAG0 (GL-DIR)
goodtg --config_path final_configs/MUTAG0/basis/no_shift/DIR.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --average_edge_attn mean --use_norm none --global_side_channel simple_concept2temperature


# BBBP (ERM)
goodtg --config_path final_configs/BBBP/basis/no_shift/ERM.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm bn --gpu_idx 0 --global_pool mean
# BBBP (GiSST)
goodtg --config_path final_configs/BBBP/basis/no_shift/GiSST.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm bn --gpu_idx 2 --average_edge_attn mean
# BBBP (GSAT)
goodtg --config_path final_configs/BBBP/basis/no_shift/GSAT.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm bn --gpu_idx 1 --average_edge_attn mean
# BBBP (SMGNN)
goodtg --config_path final_configs/BBBP/basis/no_shift/SMGNN.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm bn --gpu_idx 0 --average_edge_attn mean
# BBBP (DIR)
goodtg --config_path final_configs/BBBP/basis/no_shift/DIR.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm none --gpu_idx 1 --average_edge_attn mean
# BBBP (GL-GiSST)
goodtg --config_path final_configs/BBBP/basis/no_shift/GiSST.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm bn --gpu_idx 2 --average_edge_attn mean --global_side_channel simple_concept2temperature
# BBBP (GL-GSAT)
goodtg --config_path final_configs/BBBP/basis/no_shift/GSAT.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm bn --average_edge_attn mean --global_side_channel simple_concept2temperature
# BBBP (GL-SMGNN)
goodtg --config_path final_configs/BBBP/basis/no_shift/SMGNN.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm bn --gpu_idx 0 --average_edge_attn mean --global_side_channel simple_concept2temperature
# BBBP (GL-DIR)
goodtg --config_path final_configs/BBBP/basis/no_shift/DIR.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm none --gpu_idx 1 --average_edge_attn mean --global_side_channel simple_concept2temperature


# MNIST (ERM)
goodtg --config_path final_configs/MNIST/basis/no_shift/ERM.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm bn --gpu_idx 0
# MNIST (GiSST)
goodtg --config_path final_configs/MNIST/basis/no_shift/GiSST.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm none --average_edge_attn mean --gpu_idx 1
# MNIST (GSAT)
goodtg --config_path final_configs/MNIST/basis/no_shift/GSAT.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm bn --average_edge_attn mean --gpu_idx 1
# MNIST (SMGNN)
goodtg --config_path final_configs/MNIST/basis/no_shift/SMGNN.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm none --gpu_idx 1 --average_edge_attn mean
# MNIST (DIR)
goodtg --config_path final_configs/MNIST/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --use_norm none --gpu_idx 0 --average_edge_attn mean
# MNIST (GL-GiSST)
goodtg --config_path final_configs/MNIST/basis/no_shift/GiSST.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm none --average_edge_attn mean --gpu_idx 1 --global_side_channel simple_concept2temperature
# MNIST (GL-GSAT)
goodtg --config_path final_configs/MNIST/basis/no_shift/GSAT.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm bn --average_edge_attn mean --gpu_idx 1 --global_side_channel simple_concept2temperature
# MNIST (GL-SMGNN)
goodtg --config_path final_configs/MNIST/basis/no_shift/SMGNN.yaml --task test --seeds "1/2/3/4/5/6/7/8/9/10" --use_norm none --average_edge_attn mean --gpu_idx 1 --global_side_channel simple_concept2temperature
# MNIST (GL-DIR)
goodtg --config_path final_configs/MNIST/basis/no_shift/DIR.yaml --seeds "1/2/3/4/5/6/7/8/9/10" --task test --use_norm none --gpu_idx 0 --average_edge_attn mean  --global_side_channel simple_concept2temperature
```
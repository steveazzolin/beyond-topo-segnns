# Beyond Topological Self-Explainable GNNs: A Formal Explainability Perspective

Official supplemental material containing the sourcecode for the ICML2025 paper [Beyond Topological Self-Explainable GNNs: A Formal Explainability Perspective](https://openreview.net/pdf?id=mkqcUWBykZ).

This codebase is build on top of the one provided by [GOOD](https://github.com/divelab/GOOD).

## Model Implementations

Models implementation can be found here `GOOD\networks\models`.
Basic implementations of classifiers, shared modules, and Logic Explained Networks (referred to as *ConceptClassifier* in the source code) are available in `GOOD\networks\models\Classifiers.py`.

## Training Details

The file `GOOD\ood_algorithms\algorithms\BaseOOD.py` contains the basic training logic for each model. Then, for specific training protocols of each architecture, please refer to the corresponding file in the same folder. Note that configurations of additional models not tested in this work may be present, as they are inhereted from the original GOOD implementation.

In this work, only the following models are considered:
 - GIN (ERM - Empirical Risk Minimization)
 - GSAT
 - GiSST
 - SMGNN

## Configurations files

Configuration files and hyper-parameter details for each experiment are available in `configs/final_configs/`.

## Datasets

Dataset implementations are provided in `GOOD\data\good_datasets`.

For generating MNIST75sp the MNIST dataset, please refer to the [original paper](https://github.com/bknyaz/graph_attention_pool/tree/master/scripts). We included in our codebase the file `scripts\extract_mnist_superpixels.py` for ease of reproduction.

Full datasets can be downloaded [here](https://drive.google.com/file/d/1ZBPRnpwMTs1bpADATgC_LljtxwrqnT0D/view?usp=sharing), and stored in `storage/datasets`.

## Checkpoints

Checkpoints are available in [this GDrive folder](https://drive.google.com/file/d/1hGIHmbgCVFxuemUQU6AbFwW-AdPyiGz-/view?usp=drive_link). Checkpoints shall be placed in `storage/checkpoints`. Beware that the same folder is also the default location where checkpoints will be stored.

## Reproducing the Experiments

We report in `reproducing_commands.md` the explicit commands used to reproduce our experiments. The `--task` parameter regulates the behaviour of the code, and can be set as follows:

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


## Computing FAITH 

To compute the faithfulness score refer to `scripts\faithfulness_topofeature.sh` and `scripts\faithfulness_motif.sh`.
Files will be saved in `storage\metric_values`. 

To easily print the final faithfulness values, switch the `--task` parameter to `print_faith`. We provide in `scripts\print_faithfulness_{dataset}.sh` a utility script to showcase this. The final results will be printed for each budget `b` (aka `NECALPHA` in the code) independently as follows:

```
    suff_simple_L1        :      0.77 +- 0.07; 0.78 +- 0.07; 0.80 +- 0.04; 0.84 +- 0.04; 0.97 +- 0.04
    suff_simple_L1 Rnd    :      0.77 +- 0.06; 0.78 +- 0.05; 0.79 +- 0.04; 0.80 +- 0.04; 0.89 +- 0.04
    nec_L1                :      0.30 +- 0.09; 0.35 +- 0.13; 0.26 +- 0.12; 0.19 +- 0.06; 0.16 +- 0.04
    nec_L1 Rnd            :      0.01 +- 0.01; 0.04 +- 0.02; 0.09 +- 0.04; 0.13 +- 0.04; 0.15 +- 0.04
    faith_armon_L1        :      0.42 +- 0.08; 0.47 +- 0.11; 0.38 +- 0.13; 0.30 +- 0.08; 0.27 +- 0.06
    faith_armon_L1 Rnd    :      0.02 +- 0.02; 0.07 +- 0.04; 0.16 +- 0.06; 0.23 +- 0.05; 0.25 +- 0.05
    faith_armon_L1 ratio  :      0.05 +- 0.05; 0.15 +- 0.09; 0.42 +- 0.21; 0.77 +- 0.26; 0.93 +- 0.28
```

where each column corresponds to the value computed for a different value of explanation-cutting ratio `k` (aka `RATIOS` in the code). To replicate the results in *Table 11* of the paper, please collect, for a desired value of `k`, the values of `faith_armon_L1 ratio` for $\text{NECALPHA} \in [0.01, 0.05, 0.1]$ and then average these value toghether. In the paper, we picked as an heuristics the best `k` as the one achieving the highest `faith_armon_L1`.
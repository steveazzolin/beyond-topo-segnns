set -e

SPLITS="id_val/val/test"
SEEDS="1/2/3/4/5/6/7/8/9/10"
RATIOS="0.05/0.1/0.2/0.4/0.8"
NUMSAMPLES_BUDGET=9999999
NECALPHA=0.01
for DATASET in TopoFeature/basis/no_shift; do #GOODMotif/basis GOODMotif/size GOODMotif2/basis GOODSST2/length GOODTwitter/length GOODHIV/scaffold GOODCMNIST/color LBAPcore/assay
              
       # RATIOS="0.5/0.8/0.9/0.95" #WEIGHTS for is_weight experiment GSAT TopoFeature
       # goodtg --config_path final_configs/${DATASET}/GSAT.yaml \
       #        --seeds ${SEEDS} \
       #        --task print_faith \
       #        --metrics "suff_simple/nec" \
       #        --splits ${SPLITS} \
       #        --ratios ${RATIOS} \
       #        --log_id allclasses_recompute \
       #        --nec_alpha_1 ${NECALPHA} \
       #        --numsamples_budget ${NUMSAMPLES_BUDGET} \
       #        --average_edge_attn mean \
       #        --use_norm none \
       #        --gpu_idx 0 \
       #        --samplingtype deconfounded \
       #        --nec_number_samples prop_G_dataset
       # echo "DONE GSAT ${DATASET}"

       goodtg --config_path final_configs/${DATASET}/GSAT.yaml \
              --seeds ${SEEDS} \
              --task print_faith \
              --metrics "suff_simple/nec" \
              --splits ${SPLITS} \
              --ratios ${RATIOS} \
              --nec_alpha_1 ${NECALPHA} \
              --log_id allclasses_recompute \
              --numsamples_budget ${NUMSAMPLES_BUDGET} \
              --average_edge_attn mean \
              --use_norm none \
              --global_side_channel simple_concept2temperature \
              --gpu_idx 1 \
              --samplingtype deconfounded \
              --nec_number_samples prop_G_dataset
       echo "DONE GL-GSAT ${DATASET} NEC_ALPHA ${NECALPHA}"

    #    exit 1
              
       # RATIOS="0.01/0.1/0.25/0.6" #WEIGHTS for is_weight experiment SMGNN temp TopoFeature
       # goodtg --config_path final_configs/${DATASET}/SMGNN.yaml \
       #        --seeds ${SEEDS} \
       #        --task print_faith \
       #        --metrics "suff_simple/nec" \
       #        --splits ${SPLITS} \
       #        --ratios ${RATIOS} \
       #        --samplingtype deconfounded \
       #        --nec_number_samples prop_G_dataset \
       #        --log_id allclasses_recompute \
       #        --nec_alpha_1 ${NECALPHA} \
       #        --numsamples_budget ${NUMSAMPLES_BUDGET} \
       #        --average_edge_attn mean \
       #        --global_pool sum \
       #        --gpu_idx 0 \
       #        --global_side_channel simple_concept2temperature \
       #        --extra_param True 10 0.01 \
       #        --ood_param 0.001 \
       #        --lr_filternode 0.001 \
       #        --lr 0.001 \
       #        --use_norm none
       # echo "DONE GL-SMGNN ${DATASET}"

       # RATIOS="0.01/0.1/0.25/0.6" #WEIGHTS for is_weight experiment SMGNN temp TopoFeature
       # goodtg --config_path final_configs/${DATASET}/SMGNN.yaml \
       #        --seeds ${SEEDS} \
       #        --task print_faith \
       #        --metrics "suff_simple/nec" \
       #        --splits ${SPLITS} \
       #        --ratios ${RATIOS} \
       #        --samplingtype deconfounded \
       #        --nec_number_samples prop_G_dataset \
       #        --save_metrics \
       #        --log_id allclasses_recompute \
       #        --nec_alpha_1 ${NECALPHA} \
       #        --numsamples_budget ${NUMSAMPLES_BUDGET} \
       #        --average_edge_attn mean \
       #        --global_pool sum \
       #        --extra_param True 10 0.01 \
       #        --ood_param 0.001 \
       #        --lr 0.001 \
       #        --gpu_idx 1 \
       #        --use_norm none
       # echo "DONE SMGNN ${DATASET}"

       # RATIOS="0.6/0.5/0.4/0.3"
       # goodtg --config_path final_configs/${DATASET}/GiSST.yaml \
       #        --seeds ${SEEDS} \
       #        --task print_faith \
       #        --metrics "suff_simple/nec" \
       #        --splits ${SPLITS} \
       #        --ratios ${RATIOS} \
       #        --save_metrics \
       #        --nec_alpha_1 ${NECALPHA} \
       #        --log_id isweight \
       #        --numsamples_budget ${NUMSAMPLES_BUDGET} \
       #        --average_edge_attn mean \
       #        --use_norm bn \
       #        --gpu_idx 1 \
       #        --samplingtype deconfounded \
       #        --nec_number_samples prop_G_dataset
       # echo "DONE GiSST ${DATASET} NEC_ALPHA ${NECALPHA}"
done

set -e

SPLITS="id_val/"
SEEDS="1/2/3/4/5/6/7/8/9/10"
RATIOS="0.05/0.1/0.2/0.4/0.8"
NUMSAMPLES_BUDGET=9999999
for DATASET in TopoFeature/basis/no_shift; do
       for NECALPHA in 0.01 0.05 0.1; do

              goodtg --config_path final_configs/${DATASET}/GSAT.yaml \
                     --seeds ${SEEDS} \
                     --task eval_metric \
                     --metrics "suff_simple/nec" \
                     --splits ${SPLITS} \
                     --ratios ${RATIOS} \
                     --save_metrics \
                     --nec_alpha_1 ${NECALPHA} \
                     --log_id allclasses \
                     --numsamples_budget ${NUMSAMPLES_BUDGET} \
                     --average_edge_attn mean \
                     --use_norm none \
                     --global_side_channel simple_concept2temperature \
                     --gpu_idx 1 \
                     --samplingtype deconfounded \
                     --nec_number_samples prop_G_dataset
              echo "DONE DC-GSAT ${DATASET} NEC_ALPHA ${NECALPHA}"
              
              
              # Compute FAITH over random explanations
              goodtg --config_path final_configs/${DATASET}/GSAT.yaml \
                     --seeds ${SEEDS} \
                     --task eval_metric \
                     --metrics "suff_simple/nec" \
                     --splits ${SPLITS} \
                     --ratios ${RATIOS} \
                     --save_metrics \
                     --random_expl \
                     --nec_alpha_1 ${NECALPHA} \
                     --log_id allclasses \
                     --numsamples_budget ${NUMSAMPLES_BUDGET} \
                     --average_edge_attn mean \
                     --use_norm none \
                     --global_side_channel simple_concept2temperature \
                     --gpu_idx 1 \
                     --samplingtype deconfounded \
                     --nec_number_samples prop_G_dataset
              echo "DONE DC-GSAT ${DATASET} NEC_ALPHA ${NECALPHA} RANDOM"
       done
done

echo "DONE all :)"
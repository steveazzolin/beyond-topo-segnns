set -e

echo "I'm computing faithfulness :)"
echo "The PID of this script is: $$"

SPLITS="id_val"
SEEDS="1/2/3/4/5/6/7/8/9/10"
RATIOS="0.3/0.6/0.9"
NUMSAMPLES_BUDGET=9999999

for DATASET in MUTAG/basis/no_shift; do
       for NECALPHA in 0.01 0.05 0.1; do

            goodtg --config_path final_configs/${DATASET}/GSAT.yaml \
                    --seeds ${SEEDS} \
                    --task eval_metric \
                    --metrics "suff_simple/nec" \
                    --splits ${SPLITS} \
                    --ratios ${RATIOS} \
                    --save_metrics \
                    --nec_alpha_1 ${NECALPHA} \
                    --log_id rebuttal \
                    --numsamples_budget ${NUMSAMPLES_BUDGET} \
                    --average_edge_attn mean \
                    --use_norm none \
                    --gpu_idx 0 \
                    --samplingtype deconfounded \
                    --nec_number_samples prop_G_dataset
            echo "DONE GSAT ${DATASET} NEC_ALPHA ${NECALPHA}"

            goodtg --config_path final_configs/${DATASET}/GSAT.yaml \
                     --seeds ${SEEDS} \
                     --task eval_metric \
                     --metrics "suff_simple/nec" \
                     --splits ${SPLITS} \
                     --ratios ${RATIOS} \
                     --save_metrics \
                     --random_expl \
                     --nec_alpha_1 ${NECALPHA} \
                     --log_id rebuttal \
                     --numsamples_budget ${NUMSAMPLES_BUDGET} \
                     --average_edge_attn mean \
                     --use_norm none \
                     --gpu_idx 0 \
                     --samplingtype deconfounded \
                     --nec_number_samples prop_G_dataset
            echo "DONE GSAT ${DATASET} NEC_ALPHA ${NECALPHA} RANDOM"





            goodtg --config_path final_configs/${DATASET}/GiSST.yaml \
                    --seeds ${SEEDS} \
                    --task eval_metric \
                    --metrics "suff_simple/nec" \
                    --splits ${SPLITS} \
                    --ratios ${RATIOS} \
                    --save_metrics \
                    --nec_alpha_1 ${NECALPHA} \
                    --log_id rebuttal \
                    --numsamples_budget ${NUMSAMPLES_BUDGET} \
                    --average_edge_attn mean \
                    --use_norm none \
                    --gpu_idx 2 \
                    --samplingtype deconfounded \
                    --nec_number_samples prop_G_dataset
            echo "DONE GiSST ${DATASET} NEC_ALPHA ${NECALPHA}"

            goodtg --config_path final_configs/${DATASET}/GiSST.yaml \
                    --seeds ${SEEDS} \
                    --task eval_metric \
                    --metrics "suff_simple/nec" \
                    --splits ${SPLITS} \
                    --ratios ${RATIOS} \
                    --save_metrics \
                    --random_expl \
                    --nec_alpha_1 ${NECALPHA} \
                    --log_id rebuttal \
                    --numsamples_budget ${NUMSAMPLES_BUDGET} \
                    --average_edge_attn mean \
                    --use_norm none \
                    --gpu_idx 2 \
                    --samplingtype deconfounded \
                    --nec_number_samples prop_G_dataset
            echo "DONE GiSST ${DATASET} NEC_ALPHA ${NECALPHA} RANDOM"



            goodtg --config_path final_configs/${DATASET}/GiSST.yaml \
                    --seeds ${SEEDS} \
                    --task eval_metric \
                    --metrics "suff_simple/nec" \
                    --splits ${SPLITS} \
                    --ratios ${RATIOS} \
                    --save_metrics \
                    --nec_alpha_1 ${NECALPHA} \
                    --log_id rebuttal \
                    --numsamples_budget ${NUMSAMPLES_BUDGET} \
                    --average_edge_attn mean \
                    --global_side_channel simple_concept2temperature \
                    --use_norm none \
                    --gpu_idx 2 \
                    --samplingtype deconfounded \
                    --nec_number_samples prop_G_dataset
            echo "DONE GiSST simple_concept2temperature ${DATASET} NEC_ALPHA ${NECALPHA}"

            goodtg --config_path final_configs/${DATASET}/GiSST.yaml \
                    --seeds ${SEEDS} \
                    --task eval_metric \
                    --metrics "suff_simple/nec" \
                    --splits ${SPLITS} \
                    --ratios ${RATIOS} \
                    --save_metrics \
                    --random_expl \
                    --nec_alpha_1 ${NECALPHA} \
                    --log_id rebuttal \
                    --numsamples_budget ${NUMSAMPLES_BUDGET} \
                    --average_edge_attn mean \
                    --global_side_channel simple_concept2temperature \
                    --use_norm none \
                    --gpu_idx 2 \
                    --samplingtype deconfounded \
                    --nec_number_samples prop_G_dataset
            echo "DONE GiSST simple_concept2temperature ${DATASET} NEC_ALPHA ${NECALPHA} RANDOM"



                     
                     
            goodtg --config_path final_configs/${DATASET}/SMGNN.yaml \
                    --seeds ${SEEDS} \
                    --task eval_metric \
                    --metrics "suff_simple/nec" \
                    --splits ${SPLITS} \
                    --ratios ${RATIOS} \
                    --samplingtype deconfounded \
                    --nec_number_samples prop_G_dataset \
                    --save_metrics \
                    --nec_alpha_1 ${NECALPHA} \
                    --log_id rebuttal \
                    --numsamples_budget ${NUMSAMPLES_BUDGET} \
                    --average_edge_attn mean \
                    --gpu_idx 2 \
                    --global_side_channel simple_concept2temperature \
                    --use_norm none
            echo "DONE SMGNN simple_concept2temperature ${DATASET} NEC_ALPHA ${NECALPHA}"

            goodtg --config_path final_configs/${DATASET}/SMGNN.yaml \
                    --seeds ${SEEDS} \
                    --task eval_metric \
                    --metrics "suff_simple/nec" \
                    --splits ${SPLITS} \
                    --ratios ${RATIOS} \
                    --samplingtype deconfounded \
                    --nec_number_samples prop_G_dataset \
                    --save_metrics \
                    --random_expl \
                    --nec_alpha_1 ${NECALPHA} \
                    --log_id rebuttal \
                    --numsamples_budget ${NUMSAMPLES_BUDGET} \
                    --average_edge_attn mean \
                    --gpu_idx 2 \
                    --global_side_channel simple_concept2temperature \
                    --use_norm none
            echo "DONE SMGNN simple_concept2temperature ${DATASET} NEC_ALPHA ${NECALPHA} RANDOM"




            goodtg --config_path final_configs/${DATASET}/SMGNN.yaml \
                    --seeds ${SEEDS} \
                    --task eval_metric \
                    --metrics "suff_simple/nec" \
                    --splits ${SPLITS} \
                    --ratios ${RATIOS} \
                    --samplingtype deconfounded \
                    --nec_number_samples prop_G_dataset \
                    --save_metrics \
                    --nec_alpha_1 ${NECALPHA} \
                    --log_id rebuttal \
                    --numsamples_budget ${NUMSAMPLES_BUDGET} \
                    --average_edge_attn mean \
                    --gpu_idx 2 \
                    --use_norm none
            echo "DONE SMGNN PLAIN ${DATASET} NEC_ALPHA ${NECALPHA}"

            goodtg --config_path final_configs/${DATASET}/SMGNN.yaml \
                    --seeds ${SEEDS} \
                    --task eval_metric \
                    --metrics "suff_simple/nec" \
                    --splits ${SPLITS} \
                    --ratios ${RATIOS} \
                    --samplingtype deconfounded \
                    --nec_number_samples prop_G_dataset \
                    --save_metrics \
                    --random_expl \
                    --nec_alpha_1 ${NECALPHA} \
                    --log_id rebuttal \
                    --numsamples_budget ${NUMSAMPLES_BUDGET} \
                    --average_edge_attn mean \
                    --gpu_idx 2 \
                    --use_norm none
            echo "DONE SMGNN PLAIN ${DATASET} NEC_ALPHA ${NECALPHA} RANDOM"



            goodtg --config_path final_configs/${DATASET}/GSAT.yaml \
                    --seeds ${SEEDS} \
                    --task eval_metric \
                    --metrics "suff_simple/nec" \
                    --splits ${SPLITS} \
                    --ratios ${RATIOS} \
                    --save_metrics \
                    --nec_alpha_1 ${NECALPHA} \
                    --log_id rebuttal \
                    --numsamples_budget ${NUMSAMPLES_BUDGET} \
                    --average_edge_attn mean \
                    --use_norm none \
                    --global_side_channel simple_concept2temperature \
                    --gpu_idx 2 \
                    --samplingtype deconfounded \
                    --nec_number_samples prop_G_dataset
            echo "DONE GL-GSAT temp ${DATASET} NEC_ALPHA ${NECALPHA}"

            goodtg --config_path final_configs/${DATASET}/GSAT.yaml \
                     --seeds ${SEEDS} \
                     --task eval_metric \
                     --metrics "suff_simple/nec" \
                     --splits ${SPLITS} \
                     --ratios ${RATIOS} \
                     --save_metrics \
                     --random_expl \
                     --nec_alpha_1 ${NECALPHA} \
                     --log_id rebuttal \
                     --numsamples_budget ${NUMSAMPLES_BUDGET} \
                     --average_edge_attn mean \
                     --use_norm none \
                     --global_side_channel simple_concept2temperature \
                     --gpu_idx 2 \
                     --samplingtype deconfounded \
                     --nec_number_samples prop_G_dataset
            echo "DONE GL-GSAT temp ${DATASET} NEC_ALPHA ${NECALPHA} RANDOM"

            
       done
done

echo "DONE all :)"
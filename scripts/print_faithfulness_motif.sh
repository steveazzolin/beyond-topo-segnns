set -e

SPLITS="id_val/val/test"
SEEDS="1/2/3/4/5/6/7/8/9/10"
RATIOS="0.3/0.6/0.9"
NUMSAMPLES_BUDGET=9999999
NECALPHA=0.05

for DATASET in GOODMotif/basis/covariate; do

            # goodtg --config_path final_configs/${DATASET}/GSAT.yaml \
            #         --seeds ${SEEDS} \
            #         --task print_faith \
            #         --metrics "suff_simple/nec" \
            #         --splits ${SPLITS} \
            #         --ratios ${RATIOS} \
            #         --nec_alpha_1 ${NECALPHA} \
            #         --log_id allclasses_recompute_motif \
            #         --numsamples_budget ${NUMSAMPLES_BUDGET} \
            #         --average_edge_attn mean \
            #         --use_norm bn \
            #         --gpu_idx 1 \
            #         --samplingtype deconfounded \
            #         --nec_number_samples prop_G_dataset
            # echo "DONE GSAT ${DATASET} NEC_ALPHA ${NECALPHA}"

            # goodtg --config_path final_configs/${DATASET}/GSAT.yaml \
            #         --seeds ${SEEDS} \
            #         --task print_faith \
            #         --metrics "suff_simple/nec" \
            #         --splits ${SPLITS} \
            #         --ratios ${RATIOS} \
            #         --nec_alpha_1 ${NECALPHA} \
            #         --log_id allclasses_recompute_motif \
            #         --numsamples_budget ${NUMSAMPLES_BUDGET} \
            #         --average_edge_attn mean \
            #         --use_norm bn \
            #         --global_side_channel simple_concept2temperature \
            #         --gpu_idx 1 \
            #         --samplingtype deconfounded \
            #         --nec_number_samples prop_G_dataset
            # echo "DONE GL-GSAT temperature ${DATASET} NEC_ALPHA ${NECALPHA}"


        #     goodtg --config_path final_configs/${DATASET}/GiSST.yaml \
        #             --seeds ${SEEDS} \
        #             --task print_faith \
        #             --metrics "suff_simple/nec" \
        #             --splits ${SPLITS} \
        #             --ratios ${RATIOS} \
        #             --nec_alpha_1 ${NECALPHA} \
        #             --log_id allclasses \
        #             --numsamples_budget ${NUMSAMPLES_BUDGET} \
        #             --average_edge_attn mean \
        #             --use_norm bn \
        #             --gpu_idx 1 \
        #             --samplingtype deconfounded \
        #             --nec_number_samples prop_G_dataset
        #     echo "DONE GiSST ${DATASET} NEC_ALPHA ${NECALPHA}"
                     
                     
            # goodtg --config_path final_configs/${DATASET}/SMGNN.yaml \
            #         --seeds ${SEEDS} \
            #         --task print_faith \
            #         --metrics "suff_simple/nec" \
            #         --splits ${SPLITS} \
            #         --ratios ${RATIOS} \
            #         --samplingtype deconfounded \
            #         --nec_number_samples prop_G_dataset \
            #         --nec_alpha_1 ${NECALPHA} \
            #         --log_id allclasses_recompute_motif \
            #         --numsamples_budget ${NUMSAMPLES_BUDGET} \
            #         --average_edge_attn mean \
            #         --gpu_idx 1 \
            #         --global_side_channel simple_concept2temperature \
            #         --use_norm bn
            # echo "DONE SMGNN simple_concept2temperature ${DATASET} NEC_ALPHA ${NECALPHA}"


            goodtg --config_path final_configs/${DATASET}/SMGNN.yaml \
                    --seeds ${SEEDS} \
                    --task print_faith \
                    --metrics "suff_simple/nec" \
                    --splits ${SPLITS} \
                    --ratios ${RATIOS} \
                    --samplingtype deconfounded \
                    --nec_number_samples prop_G_dataset \
                    --nec_alpha_1 ${NECALPHA} \
                    --log_id allclasses_recompute_motif \
                    --numsamples_budget ${NUMSAMPLES_BUDGET} \
                    --average_edge_attn mean \
                    --gpu_idx 1 \
                    --use_norm bn
            echo "DONE SMGNN PLAIN ${DATASET} NEC_ALPHA ${NECALPHA} RANDOM"
            

done

echo "DONE all :)"

# --log_id allclasses | isweight | class1
set -e

echo "Time to train models with hard masking!"
echo "The PID of this script is: $$"

# for DATASET in GOODTwitter/length; do
#     goodtg --config_path final_configs/${DATASET}/covariate/CIGA.yaml \
#             --seeds "1/2/3/4/5" \
#             --mitigation_sampling feat \
#             --task train \
#             --average_edge_attn mean \
#             --gpu_idx 2 \
#             --model_name CIGAGIN
#             # --mitigation_expl_scores topK \
#             # --mitigation_expl_scores_topk 0.8 \
#             # --mitigation_readout weighted \
#             # --mitigation_virtual weighted
#     echo "DONE TRAINCIGA ${DATASET}"

#     goodtg --config_path final_configs/${DATASET}/covariate/GSAT.yaml \
#             --seeds "1/2/3/4/5" \
#             --mitigation_sampling feat \
#             --task train \
#             --average_edge_attn mean \
#             --gpu_idx 2 \
#             --model_name GSATGIN
#     echo "DONE TRAIN GSAT ${DATASET}"
# done


for DATASET in LBAPcore/assay GOODHIV/scaffold; do
    goodtg --config_path final_configs/${DATASET}/covariate/CIGA.yaml \
            --seeds "1/2/3/4/5" \
            --mitigation_sampling feat \
            --task train \
            --average_edge_attn mean \
            --gpu_idx 1 \
            --model_name CIGAGIN \
            --train_bs 256 \
            --val_bs 512 \
            --test_bs 512 \
            # --mitigation_expl_scores topK \
            # --mitigation_expl_scores_topk 0.8 \
            # --mitigation_readout weighted \
            # --mitigation_virtual weighted
    echo "DONE TRAIN CIGA ${DATASET}"

    goodtg --config_path final_configs/${DATASET}/covariate/GSAT.yaml \
            --seeds "1/2/3/4/5" \
            --mitigation_sampling feat \
            --task train \
            --average_edge_attn mean \
            --gpu_idx 1 \
            --model_name GSATGIN \
            --train_bs 256 \
            --val_bs 512 \
            --test_bs 512 \
    echo "DONE TRAIN GSAT ${DATASET}"
done

echo "DONE all :)"

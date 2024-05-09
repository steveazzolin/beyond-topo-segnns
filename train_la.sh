set -e

echo "Time to train models with hard masking!"


# for DATASET in GOODMotif2/basis GOODSST2/length GOODMotif/basis GOODTwitter/length; do
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


for DATASET in GOODCMNIST/color LBAPcore/assay GOODHIV/scaffold; do
    goodtg --config_path final_configs/${DATASET}/covariate/CIGA.yaml \
            --seeds "1/2/3/4/5" \
            --mitigation_sampling feat \
            --task train \
            --average_edge_attn mean \
            --gpu_idx 0 \
            --model_name CIGAGIN
            # --mitigation_expl_scores topK \
            # --mitigation_expl_scores_topk 0.8 \
            # --mitigation_readout weighted \
            # --mitigation_virtual weighted
    echo "DONE TRAINCIGA ${DATASET}"

    if [ "$DATASET" = "GOODCMNIST/color" ]; then
        goodtg --config_path final_configs/${DATASET}/covariate/GSAT.yaml \
            --seeds "1/2/3/4/5" \
            --mitigation_sampling feat \
            --task train \
            --average_edge_attn mean \
            --gpu_idx 0 \
            --train_bs 256 \
            --val_bs 256 \
            --test_bs 256 \
            --num_workers 2 \
            --model_name GSATGIN
    else
        goodtg --config_path final_configs/${DATASET}/covariate/GSAT.yaml \
            --seeds "1/2/3/4/5" \
            --mitigation_sampling feat \
            --task train \
            --average_edge_attn mean \
            --gpu_idx 0 \
            --model_name GSATGIN
    fi
    echo "DONE TRAIN GSAT ${DATASET}"
done

echo "DONE all :)"

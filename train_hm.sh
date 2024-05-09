set -e

echo "Time to train models with hard masking!"


for DATASET in GOODMotif2/basis GOODSST2/length LBAPcore/assay GOODCMNIST/color GOODMotif/basis GOODMotif/size GOODTwitter/length GOODHIV/scaffold; do
       for MODEL in GSAT; do
            for EXPL_SCORES in hard; do
                if [ "$DATASET" = "GOODCMNIST/color" ]; then
                    goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                        --seeds "1/2/3/4/5" \
                        --mitigation_sampling feat \
                        --task train \
                        --average_edge_attn mean \
                        --gpu_idx 1 \
                        --num_workers 2 \
                        --train_bs 256 \
                        --val_bs 256 \
                        --test_bs 256 \
                        --mitigation_expl_scores ${EXPL_SCORES}
                else
                    goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                        --seeds "1/2/3/4/5" \
                        --mitigation_sampling feat \
                        --task train \
                        --average_edge_attn mean \
                        --gpu_idx 1 \
                        --mitigation_expl_scores ${EXPL_SCORES}
                fi
                        # --mitigation_expl_scores topK \
                        # --mitigation_expl_scores_topk 0.8 \
                        # --mitigation_readout weighted \
                        # --mitigation_virtual weighted
                echo "DONE TRAIN ${MODEL} ${DATASET} ${EXPL_SCORES}"
            done
       done
done

# for DATASET in BBBP/basis MultiShapes/basis; do
#     for EXPL_SCORES in hard anneal topK; do
#         goodtg --config_path final_configs/${DATASET}/no_shift/GSAT.yaml \
#                 --seeds "1/2/3/4/5" \
#                 --mitigation_sampling feat \
#                 --task train \
#                 --average_edge_attn mean \
#                 --gpu_idx 1 \
#                 --mitigation_expl_scores ${EXPL_SCORES} \
#                 --mitigation_expl_scores_topk 0.8
#         echo "DONE TRAIN GSAT ${DATASET}"
#     done
# done

echo "DONE all :)"

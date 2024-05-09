set -e

echo "Time to train models with hard masking!"


for DATASET in GOODMotif/basis GOODMotif2/basis GOODSST2/length GOODTwitter/length; do
       for MODEL in LECI; do
            for EXPL_SCORES in hard anneal; do
                goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                        --seeds "1/2/3/4/5" \
                        --mitigation_sampling feat \
                        --task test \
                        --average_edge_attn mean \
                        --gpu_idx 1 \
                        --mitigation_expl_scores ${EXPL_SCORES}
                        # --mitigation_expl_scores topK \
                        # --mitigation_expl_scores_topk 0.8 \
                        # --mitigation_readout weighted \
                        # --mitigation_virtual weighted
                echo "DONE TRAIN ${MODEL} ${DATASET} ${EXPL_SCORES}"
            done
       done
done

# for DATASET in GOODMotif/size GOODHIV/scaffold LBAPcore/assay; do
#        for MODEL in LECI; do
#             for EXPL_SCORES in hard anneal; do
#                 goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
#                         --seeds "1/2/3/4/5" \
#                         --mitigation_sampling feat \
#                         --task train \
#                         --average_edge_attn mean \
#                         --gpu_idx 1 \
#                         --num_workers 2 \
#                         --mitigation_expl_scores ${EXPL_SCORES}
#                         # --mitigation_expl_scores topK \
#                         # --mitigation_expl_scores_topk 0.8 \
#                         # --mitigation_readout weighted \
#                         # --mitigation_virtual weighted
#                 echo "DONE TRAIN ${MODEL} ${DATASET} ${EXPL_SCORES}"
#             done
#        done
# done

echo "DONE all :)"

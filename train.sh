set -e

echo "Time to train models!"

for DATASET in GOODMotif/basis GOODMotif/size GOODMotif2/basis GOODSST2/length GOODTwitter/length GOODHIV/scaffold LBAPcore/assay GOODCMNIST/color; do #GOODMotif/basis GOODMotif/size GOODMotif2/basis GOODSST2/length GOODTwitter/length GOODHIV/scaffold GOODCMNIST/color LBAPcore/assay
       for MODEL in CIGA; do
              goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                     --seeds "4/5" \
                     --mitigation_sampling raw \
                     --task train \
                     --average_edge_attn mean \
                     --gpu_idx 0
                     # --save_metrics
                     # --mitigation_expl_scores topK \
                     # --mitigation_expl_scores_topk 0.8 \
                     # --mitigation_readout weighted \
                     # --mitigation_virtual weighted
              echo "DONE TRAIN ${MODEL} ${DATASET} CF"
       done
done

# for DATASET in LBAPcore/assay; do #GOODMotif/basis GOODMotif/size GOODMotif2/basis GOODSST2/length GOODTwitter/length GOODHIV/scaffold GOODCMNIST/color LBAPcore/assay
#        for MODEL in LECI; do
#               goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
#                      --seeds "1/2/3/4/5" \
#                      --mitigation_sampling feat \
#                      --task train \
#                      --average_edge_attn mean \
#                      --gpu_idx 1 \
#                      --mitigation_expl_scores topK \
#                      --mitigation_expl_scores_topk 0.4 \
#                      --mitigation_readout weighted \
#                      --mitigation_virtual weighted \
#                      --num_workers 2 \
#                      # --save_metrics
#                      # --mitigation_expl_scores topK \
#                      # --mitigation_expl_scores_topk 0.8 \
#                      # --mitigation_readout weighted \
#                      # --mitigation_virtual weighted
#               echo "DONE TRAIN ${MODEL} ${DATASET}"
#        done
# done

echo "DONE all :)"

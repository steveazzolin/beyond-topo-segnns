set -e

echo "Time to train models!"

for DATASET in GOODMotif/basis GOODMotif/size GOODMotif2/basis GOODSST2/length GOODTwitter/length GOODHIV/scaffold LBAPcore/assay GOODCMNIST/color; do #GOODMotif/basis GOODMotif/size GOODMotif2/basis GOODSST2/length GOODTwitter/length GOODHIV/scaffold GOODCMNIST/color LBAPcore/assay
       for MODEL in LECI; do
              goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                     --seeds "10" \
                     --mitigation_sampling feat \
                     --mitigation_expl_scores topK \
                     --mitigation_expl_scores_topk 0.6 \
                     --task train \
                     --average_edge_attn mean \
                     --gpu_idx 2
                     # --num_workers 2
                     # --mitigation_expl_scores_topk 0.6 \
                     # --mitigation_readout weighted \
                     # --mitigation_virtual weighted
              echo "DONE TRAIN ${MODEL} ${DATASET}"
       done
done

echo "DONE all :)"

set -e

echo "Time to evaluate models!"

for DATASET in GOODMotif/basis GOODMotif2/basis GOODMotif/size GOODSST2/length GOODTwitter/length; do #GOODMotif/basis GOODMotif2/basis GOODMotif/size GOODSST2/length GOODTwitter/length GOODHIV/scaffold GOODCMNIST/color LBAPcore/assay
       for MODEL in CIGA; do
              goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                     --seeds "1/2/3/4/5" \
                     --mitigation_sampling feat \
                     --task test \
                     --average_edge_attn mean \
                     --mitigation_readout weighted \
                     --gpu_idx 1
              echo "DONE EVAL ${MODEL} ${DATASET}"
       done
done


# --save_metrics
# --mitigation_expl_scores topK \
# --mitigation_expl_scores_topk 0.8 \
# --mitigation_readout weighted \
# --mitigation_virtual weighted

echo "DONE all :)"

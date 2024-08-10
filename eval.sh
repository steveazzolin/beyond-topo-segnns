set -e

echo "Time to evaluate models!"

# GOODMotif2/basis LBAPcore/assay GOODCMNIST/color GOODSST2/length
for DATASET in GOODSST2/length; do #GOODMotif/basis GOODMotif2/basis GOODMotif/size GOODSST2/length GOODTwitter/length GOODHIV/scaffold GOODCMNIST/color LBAPcore/assay
       for MODEL in GSAT LECI; do
              for EXPL_SCORE in topK; do
                     goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                            --seeds "1/2/3/4/5" \
                            --mitigation_sampling feat \
                            --task test \
                            --average_edge_attn mean \
                            --gpu_idx 0 \
                            --val_bs 64 \
                            --test_bs 64 \
                            --mitigation_expl_scores ${EXPL_SCORE} \
                            --mitigation_expl_scores_topk 0.8
                            # -ood_param 10 \
                            # --extra_param True 10 0.2
                            # --train_bs 64 \
                            # --val_bs 64 \
                            # --test_bs 64
                     echo "DONE EVAL ${MODEL} ${DATASET} ${EXPL_SCORE}"
              done
       done
done


# --save_metrics
# --mitigation_expl_scores topK \
# --mitigation_expl_scores_topk 0.8 \
# --mitigation_readout weighted \
# --mitigation_virtual weighted
# --ood_param 0.5 \
# --extra_param True 10 0.2 \

echo "DONE all :)"

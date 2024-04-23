set -e

echo "Time to evaluate models!"

# for DATASET in GOODMotif/basis GOODMotif2/basis GOODMotif/size GOODSST2/length GOODTwitter/length GOODHIV/scaffold LBAPcore/assay; do #GOODMotif/basis GOODMotif2/basis GOODMotif/size GOODSST2/length GOODTwitter/length GOODHIV/scaffold GOODCMNIST/color LBAPcore/assay
#        for MODEL in GSAT; do
#               goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
#                      --seeds "1/2/3" \
#                      --mitigation_sampling feat \
#                      --task test \
#                      --average_edge_attn mean \
#                      --gpu_idx 1 \
#                      --mitigation_readout weighted \
#                      --num_workers 2
#                      # --save_metrics
#                      # --mitigation_expl_scores topK \
#                      # --mitigation_expl_scores_topk 0.8 \
#                      # --mitigation_readout weighted \
#                      # --mitigation_virtual weighted
#               echo "DONE EVAL ${MODEL} ${DATASET}"
#        done
# done

goodtg --config_path final_configs/GOODCMNIST/color/covariate/GSAT.yaml \
        --seeds "1/2/3" \
        --mitigation_sampling feat \
        --task test \
        --average_edge_attn mean \
        --gpu_idx 1 \
        --mitigation_readout weighted \
        --num_workers 2 \
        --train_bs 32 \
        --eval_bs 32 \
        --test_bs 32

echo "DONE all :)"

set -e

echo "Time to train models!"

goodtg --config_path final_configs/GOODMotif2/basis/covariate/GSAT.yaml \
       --seeds "4/5" \
       --mitigation_sampling feat \
       --task train \
       --average_edge_attn mean \
       --gpu_idx 0 \
       --ood_param 0.5 \
       --extra_param True 10 0.2
echo "DONE TRAIN Motif2"

goodtg --config_path final_configs/GOODMotif2/basis/covariate/GSAT.yaml \
       --seeds "1/2/3/4/5" \
       --mitigation_sampling feat \
       --task train \
       --average_edge_attn mean \
       --mitigation_readout weighted \
       --gpu_idx 0 \
       --ood_param 0.5 \
       --extra_param True 10 0.2

echo "DONE TRAIN Motif2 readout"

for DATASET in GOODMotif/size; do #GOODMotif/basis GOODMotif/size GOODMotif2/basis GOODSST2/length GOODTwitter/length GOODHIV/scaffold GOODCMNIST/color LBAPcore/assay
       for MODEL in GSAT; do
              goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                     --seeds "1/2/3/4/5" \
                     --mitigation_sampling feat \
                     --task train \
                     --average_edge_attn mean \
                     --gpu_idx 0 \
                     --ood_param 0.5 \
                     --extra_param True 10 0.2
                     # --save_metrics
                     # --mitigation_expl_scores topK \
                     # --mitigation_expl_scores_topk 0.8 \
                     # --mitigation_readout weighted \
                     # --mitigation_virtual weighted
              echo "DONE TRAIN ${MODEL} ${DATASET} regularized"

              goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                     --seeds "1/2/3/4/5" \
                     --mitigation_sampling feat \
                     --task train \
                     --average_edge_attn mean \
                     --gpu_idx 0 \
                     --ood_param 0.5 \
                     --extra_param True 10 0.2 \
                     --mitigation_readout weighted
                     # --save_metrics
                     # --mitigation_expl_scores topK \
                     # --mitigation_expl_scores_topk 0.8 \
                     # --mitigation_readout weighted \
                     # --mitigation_virtual weighted
              echo "DONE TRAIN ${MODEL} ${DATASET} regularized mitigation_readout"
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

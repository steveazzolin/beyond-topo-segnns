# This script was created from a copy of train_hm.sh to train Hard masking versions of missing models.
# I did not modify the original script just to have an original copy of the copy submitted as Supplementary

set -e

echo "Time to train models with hard masking!"
echo "The PID of this script is: $$"

MODEL=$1

#GOODMotif2/basis LBAPcore/assay GOODSST2/length GOODCMNIST/color 
for DATASET in GOODSST2/length; do
    
    BS=$( [ "$DATASET" = "GOODCMNIST/color" ] && echo 64 || echo 1024 )
    GPU=1

    goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
            --seeds "1/2/3/4/5" \
            --mitigation_sampling feat \
            --task train \
            --average_edge_attn mean \
            --gpu_idx ${GPU} \
            --mitigation_expl_scores anneal \
            --val_bs ${BS} \
            --test_bs ${BS}
    echo "DONE TRAIN ${MODEL} ${DATASET} anneal"

    goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
            --seeds "1/2/3/4/5" \
            --mitigation_sampling feat \
            --task train \
            --average_edge_attn mean \
            --gpu_idx ${GPU} \
            --mitigation_expl_scores topK \
            --mitigation_expl_scores_topk 0.4 \
            --val_bs ${BS} \
            --test_bs ${BS}
    echo "DONE TRAIN ${MODEL} ${DATASET} Top0.4"

    goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
            --seeds "1/2/3/4/5" \
            --mitigation_sampling feat \
            --task train \
            --average_edge_attn mean \
            --gpu_idx ${GPU} \
            --mitigation_expl_scores topK \
            --mitigation_expl_scores_topk 0.8 \
            --val_bs ${BS} \
            --test_bs ${BS}
    echo "DONE TRAIN ${MODEL} ${DATASET} Top0.8"
done

echo "DONE all :)"






# GOODMotif2/basis GOODSST2/length
# LBAPcore/assay GOODMotif/basis GOODMotif/size GOODTwitter/length GOODHIV/scaffold  running as *hm_large.txt
# for DATASET in GOODMotif/size GOODTwitter/length GOODHIV/scaffold; do
#        for MODEL in GSAT; do
#             for EXPL_SCORES in hard; do
#                 if [ "$DATASET" = "GOODHIV/assay" ]; then
#                     goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
#                         --seeds "1/2/3/4/5" \
#                         --mitigation_sampling feat \
#                         --task train \
#                         --average_edge_attn mean \
#                         --gpu_idx 1 \
#                         --train_bs 1024 \
#                         --val_bs 1024 \
#                         --test_bs 1024 \
#                         --mitigation_expl_scores ${EXPL_SCORES}
#                 else
#                     goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
#                         --seeds "1/2/3/4/5" \
#                         --mitigation_sampling feat \
#                         --task train \
#                         --average_edge_attn mean \
#                         --gpu_idx 1 \
#                         --train_bs 128 \
#                         --val_bs 512 \
#                         --test_bs 512 \
#                         --mitigation_expl_scores ${EXPL_SCORES}
#                 fi
#                         # --mitigation_expl_scores topK \
#                         # --mitigation_expl_scores_topk 0.8 \
#                         # --mitigation_readout weighted \
#                         # --mitigation_virtual weighted
#                 echo "DONE TRAIN ${MODEL} ${DATASET} ${EXPL_SCORES}"
#             done
#        done
# done
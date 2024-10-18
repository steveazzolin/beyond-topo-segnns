set -e

echo "Time to train models with hard masking!"
echo "The PID of this script is: $$"

for DATASET in GOODMotif2/basis GOODCMNIST/color LBAPcore/assay GOODSST2/length GOODTwitter/length GOODMotif/basis GOODMotif/size GOODHIV/scaffold; do
       for MODEL in GSAT; do
            for EXPL_SCORES in hard; do
                if [ "$DATASET" = "GOODCMNIST/color" ]; then
                    goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                        --seeds "1/2/3/4/5" \
                        --mitigation_sampling feat \
                        --task test \
                        --average_edge_attn mean \
                        --gpu_idx 0 \
                        --train_bs 64 \
                        --val_bs 64 \
                        --test_bs 64 \
                        --model_name GSATGIN \
                        --mitigation_expl_scores ${EXPL_SCORES} \
                        --mitigation_readout weighted
                elif [ "$DATASET" = "LBAPcore/assay" ]; then
                    goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                        --seeds "1/2/3/4/5" \
                        --mitigation_sampling feat \
                        --task test \
                        --average_edge_attn mean \
                        --gpu_idx 0 \
                        --train_bs 128 \
                        --val_bs 1024 \
                        --test_bs 1024 \
                        --model_name GSATGIN \
                        --mitigation_expl_scores ${EXPL_SCORES} \
                        --mitigation_readout weighted
                else
                    goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                        --seeds "1/2/3/4/5" \
                        --mitigation_sampling feat \
                        --task test \
                        --average_edge_attn mean \
                        --gpu_idx 0 \
                        --train_bs 128 \
                        --val_bs 1024 \
                        --test_bs 1024 \
                        --model_name GSATGIN \
                        --mitigation_expl_scores ${EXPL_SCORES} \
                        --mitigation_readout weighted
                fi
                echo "DONE TRAIN ${MODEL} ${DATASET} ${EXPL_SCORES}"
            done
       done
done

echo "DONE all :)"


# --mitigation_expl_scores topK \
# --mitigation_expl_scores_topk 0.8 \
# --mitigation_readout weighted \
# --mitigation_virtual weighted


# --train_bs 32 \
# --val_bs 512 \
# --test_bs 512 \
# --ood_param 10 \
# --extra_param True 10 0.2
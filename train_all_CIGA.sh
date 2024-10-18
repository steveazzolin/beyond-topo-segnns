set -e

echo "Time to train models with hard masking!"
echo "The PID of this script is: $$"

# GOODMotif2/basis LBAPcore/assay GOODSST2/length GOODCMNIST/color GOODTwitter/length GOODMotif/basis GOODMotif/size GOODHIV/scaffold TRAIN 
for DATASET in GOODSST2/length; do
       for MODEL in CIGA; do
            if [ "$DATASET" = "GOODCMNIST/color" ]; then
                goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                    --seeds "1/2/3/4/5" \
                    --mitigation_sampling raw \
                    --task test \
                    --average_edge_attn mean \
                    --gpu_idx 0 \
                    --train_bs 64 \
                    --val_bs 64 \
                    --test_bs 64 \
                    --model_name CIGAGIN \
                    --mitigation_readout weighted
            else
                goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                    --seeds "1/2/3/4/5" \
                    --mitigation_sampling raw \
                    --task test \
                    --average_edge_attn mean \
                    --gpu_idx 0 \
                    --train_bs 64 \
                    --val_bs 512 \
                    --test_bs 512 \
                    --model_name CIGAGIN \
                    --mitigation_readout weighted
            fi
            echo "DONE TRAIN ${MODEL} ${DATASET}"
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
set -e

echo "I'm tuning the hyper-parameters of GSAT :)"
echo "The PID of this script is: $$"

for OOD_PARAM in 0.001 0.0001 0.01; do
    for EXTRA_PARAM in 0.1 0.01 0.001; do
        for WD in 0.1 0.01 0.001 0.005; do
            for COWD in 0.0 0.1 0.01 0.001; do
                for TMP in 0.5 0.6 0.8; do
                    for DATASET in GOODTwitter/length/covariate; do
                        goodtg --config_path final_configs/GOODTwitter/length/covariate/SMGNN.yaml \
                            --seeds "1/2/3/4/5/6/7/8/9/10" \
                            --task train \
                            --average_edge_attn mean \
                            --global_pool mean \
                            --gpu_idx 1 \
                            --extra_param True 10 ${EXTRA_PARAM} \
                            --ood_param ${OOD_PARAM} \
                            --weight_decay ${WD} \
                            --channel_weight_decay ${WD} \
                            --combinator_weight_decay ${COWD} \
                            --end_temp ${TMP}\
                            --lr_filternode 0.001 \
                            --lr 0.001 \
                            --use_norm none \
                            --mitigation_sampling raw \
                            --global_side_channel simple_concept2temperature
                        echo "DONE ${DATASET} ${OOD_PARAM};True 10 ${EXTRA_PARAM};${WD};${WD};${COWD};${TMP}"
                    done
                done
            done
        done
    done
done

echo "DONE all :)"




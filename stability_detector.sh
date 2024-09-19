set -e

echo "I'm the stability of the DET :))"
echo "The PID of this script is: $$"

for DATASET in GOODMotif2/basis GOODMotif/size; do
    for MODEL in LECI; do
        for BUDGET in 0.05 0.10; do
            SPLITS="id_test"
                
            goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                    --seeds "1/2/3/4/5" \
                    --task stability_detector \
                    --metrics "suff_simple" \
                    --ratios "0.3/0.6" \
                    --splits "${SPLITS}" \
                    --average_edge_attn mean \
                    --mitigation_sampling feat \
                    --gpu_idx 0 \
                    --mask  \
                    --expval_budget 5 \
                    --numsamples_budget 99999999 \
                    --samplingtype deconfounded \
                    --nec_number_samples prop_G_dataset \
                    --nec_alpha_1 ${BUDGET} \
                    --debias \
                    # --extra_param True 10 0.2 --ood_param 10.0 # for GSAT Self-Explainable experiments
            echo "DONE ${MODEL} ${DATASET} ${BUDGET}"
        done
    done
done


echo "DONE all :)"

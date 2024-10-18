set -e

if [ -z "$1" ]; then
    GPU=1
else
    GPU=$1
fi


echo "I'm the stability of the DET :))"
echo "The PID of this script is: $$ running on GPU-$GPU"

for DATASET in GOODMotif2/basis MultiShapes/basis LBAPcore/assay GOODMotif2/basis GOODCMNIST/color GOODSST2/length; do # GOODMotif2/basis GOODMotif/size GOODTwitter/length LBAPcore/assay GOODSST2/length
    for MODEL in GSAT; do
        for BUDGET in 0.05 0.10; do
            SPLITS="id_val"
                
            goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                    --seeds "1/2/3/10/11" \
                    --task stability_detector \
                    --metrics "suff_simple" \
                    --ratios "0.3/0.6" \
                    --splits "${SPLITS}" \
                    --average_edge_attn mean \
                    --mitigation_sampling feat \
                    --gpu_idx ${GPU} \
                    --mask  \
                    --expval_budget 5 \
                    --numsamples_budget 20 \
                    --samplingtype deconfounded \
                    --nec_number_samples prop_G_dataset \
                    --nec_alpha_1 ${BUDGET} \
                    # --debias \
                    # --extra_param True 10 0.2 --ood_param 0.5 #10.0 (Motif) 0.5 (BaMS) # for GSAT Self-Explainable experiments
                    # --model_name ${MODEL}GIN \
                    # --mitigation_readout weighted
                    # --mitigation_expl_scores hard
                    # --debias \
                    # --mitigation_readout weighted \
                    # --mitigation_expl_scores hard \
            echo "DONE ${MODEL} ${DATASET} ${BUDGET}"
            exit 1
        done
    done
done


echo "DONE all :)"

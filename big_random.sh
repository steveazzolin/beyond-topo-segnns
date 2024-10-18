# for running experiments on random explanations

echo "Time to compute metrics for random explanations!"
set -e

for DATASET in GOODMotif2/basis GOODSST2/length LBAPcore/assay GOODCMNIST/color; do #GOODMotif/basis GOODMotif/size GOODMotif2/basis GOODSST2/length GOODTwitter/length GOODHIV/scaffold GOODCMNIST/color LBAPcore/assay
       for MODEL in LECI GSAT; do # CIGA LECI GSAT
            
            # goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
            #     --seeds "1/2/3/4/5" \
            #     --task eval_metric \
            #     --metrics "suff++/nec" \
            #     --splits "id_val/test" \
            #     --average_edge_attn mean \
            #     --mitigation_sampling raw \
            #     --model_name CIGAGIN \
            #     --mitigation_readout weighted \
            #     --gpu_idx 2 \
            #     --mask  \
            #     --debias \
            #     --random_expl True \
            #     --samplingtype deconfounded \
            #     --nec_number_samples prop_G_dataset \
            #     --save_metrics \
            #     --log_id suff++_old_randomexpl_ALL
            # echo "DONE ${MODEL} ${DATASET}"


            goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                --seeds "1/2/3/4/5" \
                --task eval_metric \
                --metrics "suff++/nec" \
                --splits "id_val/test" \
                --average_edge_attn mean \
                --mitigation_sampling feat \
                --mitigation_expl_scores hard \
                --gpu_idx 0 \
                --mask  \
                --debias \
                --random_expl True \
                --samplingtype deconfounded \
                --nec_number_samples prop_G_dataset \
                --save_metrics \
                --log_id suff++_old_randomexpl_HS
                # --mitigation_readout weighted \
                # --mitigation_virtual weighted \
                # --mitigation_expl_scores ${EXPL_SCORES} \
                # --mitigation_expl_scores_topk 0.8 \
                # --model_name LECIGIN \
            echo "DONE ${MODEL} ${DATASET}"
       done
done

echo "DONE all :)"



# --mitigation_readout weighted \
# --mitigation_virtual weighted \
# --mitigation_expl_scores ${EXPL_SCORES} \
# --mitigation_expl_scores_topk 0.8 \
# --model_name LECIGIN \
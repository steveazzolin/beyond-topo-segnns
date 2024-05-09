set -e

# GOODMotif/size GOODTwitter/length GOODSST2/length GOODHIV/scaffold LBAPcore/assay   # for nov only
for DATASET in GOODSST2/length GOODTwitter/length GOODHIV/scaffold GOODCMNIST/color LBAPcore/assay; do #GOODMotif/basis GOODMotif/size GOODMotif2/basis GOODSST2/length GOODTwitter/length GOODHIV/scaffold GOODCMNIST/color LBAPcore/assay
       for MODEL in LECI; do # CIGA LECI GSAT
              if [ "$DATASET" = "GOODSST2/length" ]; then
                     SPLITS="id_val/test"
              else
                     SPLITS="id_test/id_val//test"
              fi

              goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                     --seeds "1/2/3/4/5" \
                     --task eval_metric \
                     --metrics "suff++/nec" \
                     --splits "${SPLITS}" \
                     --average_edge_attn mean \
                     --mitigation_sampling feat \
                     --mitigation_expl_scores hard \
                     --gpu_idx 2 \
                     --mask  \
                     --debias \
                     --samplingtype deconfounded \
                     --nec_number_samples prop_G_dataset \
                     --save_metrics \
                     --log_id suff++_old_hardonly
              echo "DONE ${MODEL} ${DATASET}"

              goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                     --seeds "1/2/3/4/5" \
                     --task eval_metric \
                     --metrics "suff++/nec" \
                     --splits ${SPLITS} \
                     --average_edge_attn mean \
                     --mitigation_sampling feat \
                     --mitigation_expl_scores anneal \
                     --gpu_idx 2 \
                     --mask  \
                     --debias \
                     --samplingtype deconfounded \
                     --nec_number_samples prop_G_dataset \
                     --save_metrics \
                     --log_id suff++_old_annealonly
              echo "DONE ${MODEL} ${DATASET}"
       done
done

echo "DONE all :)"


# --save_metrics \
# --log_id suff++_old_allmitig_${EXPL_SCORES} #suff_simple_old  suff++_old_allmitig_${EXPL_SCORES}  suff++_old_inputfeatures
# --mitigation_readout weighted \
# --mitigation_virtual weighted \
# --mitigation_expl_scores ${EXPL_SCORES} \
# --mitigation_expl_scores_topk 0.8 \
# --model_name LECIGIN \
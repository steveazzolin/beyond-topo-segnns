set -e


for DATASET in GOODMotif/basis GOODMotif2/basis GOODSST2/length GOODTwitter/length GOODMotif/size GOODHIV/scaffold GOODCMNIST/color LBAPcore/assay; do #GOODMotif/basis GOODMotif/size GOODMotif2/basis GOODSST2/length GOODTwitter/length GOODHIV/scaffold GOODCMNIST/color LBAPcore/assay
       for MODEL in LECI; do # CIGA LECI GSAT
              for EXPL_SCORES in topk; do #hard anneal
                     goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                            --seeds "1/2/3/4/5" \
                            --task eval_metric \
                            --metrics "suff++/nec" \
                            --average_edge_attn mean \
                            --mitigation_sampling feat \
                            --mitigation_readout weighted \
                            --mitigation_virtual weighted \
                            --mitigation_expl_scores ${EXPL_SCORES} \
                            --mitigation_expl_scores_topk 0.4 \
                            --gpu_idx 0 \
                            --mask  \
                            --debias \
                            --samplingtype deconfounded \
                            --nec_number_samples prop_G_dataset \
                            --save_metrics \
                            --log_id suff++_old_allmitig_${EXPL_SCORES}0.4 #suff_simple_old  suff++_old_allmitig_${EXPL_SCORES}  suff++_old_inputfeatures
                            # --mitigation_readout weighted \
                            # --mitigation_virtual weighted \
                            # --mitigation_expl_scores ${EXPL_SCORES} \
                            # --mitigation_expl_scores_topk 0.8 \
                            # --model_name LECIGIN \
                     echo "DONE ${MODEL} ${DATASET}"
              done
       done
done

echo "DONE all :)"

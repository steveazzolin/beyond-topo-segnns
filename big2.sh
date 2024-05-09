# for running experiments on GSAT and CIGA

echo "Time to compute metrics!"
set -e


for DATASET in GOODMotif/basis GOODMotif/size GOODMotif2/basis GOODSST2/length GOODTwitter/length; do #GOODMotif/basis GOODMotif/size GOODMotif2/basis GOODSST2/length GOODTwitter/length GOODHIV/scaffold GOODCMNIST/color LBAPcore/assay
       for MODEL in CIGA; do # CIGA LECI GSAT
              goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                     --seeds "1/2/3/4/5" \
                     --task eval_metric \
                     --metrics "suff++/nec" \
                     --splits "test" \
                     --average_edge_attn mean \
                     --mitigation_sampling feat \
                     --mitigation_readout weighted \
                     --gpu_idx 2 \
                     --mask  \
                     --debias \
                     --samplingtype deconfounded \
                     --nec_number_samples prop_G_dataset \
                     # --save_metrics \
                     # --log_id suff++_old_mitigreadout_weighted
              echo "DONE ${MODEL} ${DATASET} readout"
       done
done

echo "DONE all :)"
# --ood_param 0.5 \
# --extra_param True 10 0.2
# --nec_alpha_1 0.05 \
# --mitigation_readout weighted \
# --save_metrics \
# --log_id suff++_old_mitigreadout_weighted #suff_simple_old  suff++_old_allmitig_${EXPL_SCORES}  suff++_old_inputfeatures
# --mitigation_readout weighted \
# --mitigation_virtual weighted \
# --mitigation_expl_scores ${EXPL_SCORES} \
# --mitigation_expl_scores_topk 0.8 \
# --model_name LECIGIN \
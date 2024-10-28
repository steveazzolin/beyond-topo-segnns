set -e

echo "I'm BIG :)"
echo "The PID of this script is: $$"

SPLITS=id_val
for DATASET in TopoFeature/basis; do #GOODMotif/basis GOODMotif/size GOODMotif2/basis GOODSST2/length GOODTwitter/length GOODHIV/scaffold GOODCMNIST/color LBAPcore/assay
       for MODEL in GSAT; do # GSAT SMGNN
              
              goodtg --config_path final_configs/${DATASET}/no_shift/${MODEL}.yaml \
                     --seeds "1" \
                     --task eval_metric \
                     --metrics "suff++/nec" \
                     --splits "${SPLITS}" \
                     --average_edge_attn mean \
                     --gpu_idx 0 \
                     --mask  \
                     --debias \
                     --samplingtype deconfounded \
                     --nec_number_samples prop_G_dataset
              echo "DONE ${MODEL} ${DATASET} anneal"
              
              
       done
done

echo "DONE all :)"

# --global_side_channel simple_concept2temperature \
# --save_metrics \
# --log_id suff++_old_allmitig_${EXPL_SCORES} #suff_simple_old  suff++_old_allmitig_${EXPL_SCORES}  suff++_old_inputfeatures
# --mitigation_readout weighted \
# --mitigation_virtual weighted \
# --mitigation_expl_scores ${EXPL_SCORES} \
# --mitigation_expl_scores_topk 0.8 \
# --model_name LECIGIN \
set -e

echo "I'm computing faithfulness :)"
echo "The PID of this script is: $$"

SPLITS="id_val/val/test"
SEEDS="1/2/3/4/5/6/7/8/9/10"
RATIOS="0.05/0.1/0.2/0.4/0.8"
NUMSAMPLES_BUDGET=9999999
for DATASET in TopoFeature/basis/no_shift; do #GOODMotif/basis GOODMotif/size GOODMotif2/basis GOODSST2/length GOODTwitter/length GOODHIV/scaffold GOODCMNIST/color LBAPcore/assay
       for NECALPHA in 0.01 0.05 0.1; do              
              goodtg --config_path final_configs/${DATASET}/GSAT.yaml \
                     --seeds ${SEEDS} \
                     --task eval_metric \
                     --metrics "suff_simple/nec" \
                     --splits ${SPLITS} \
                     --ratios ${RATIOS} \
                     --save_metrics \
                     --random_expl \
                     --nec_alpha_1 ${NECALPHA} \
                     --log_id allclasses \
                     --numsamples_budget ${NUMSAMPLES_BUDGET} \
                     --average_edge_attn mean \
                     --use_norm none \
                     --gpu_idx 0 \
                     --samplingtype deconfounded \
                     --nec_number_samples prop_G_dataset
              echo "DONE GSAT ${DATASET} NEC_ALPHA ${NECALPHA}"
                     
                     
              goodtg --config_path final_configs/${DATASET}/SMGNN.yaml \
                     --seeds ${SEEDS} \
                     --task eval_metric \
                     --metrics "suff_simple/nec" \
                     --splits ${SPLITS} \
                     --ratios ${RATIOS} \
                     --samplingtype deconfounded \
                     --nec_number_samples prop_G_dataset \
                     --save_metrics \
                     --random_expl \
                     --nec_alpha_1 ${NECALPHA} \
                     --log_id allclasses \
                     --numsamples_budget ${NUMSAMPLES_BUDGET} \
                     --average_edge_attn mean \
                     --global_pool sum \
                     --gpu_idx 0 \
                     --global_side_channel simple_concept2temperature \
                     --extra_param True 10 0.01 \
                     --ood_param 0.001 \
                     --lr_filternode 0.001 \
                     --lr 0.001 \
                     --use_norm none
              echo "DONE SMGNN ${DATASET} NEC_ALPHA ${NECALPHA}"
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
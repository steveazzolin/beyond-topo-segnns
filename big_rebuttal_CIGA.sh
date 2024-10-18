echo "Time to compute metrics!"
echo "The PID of this script is: $$"
set -e


for DATASET in GOODMotif2/basis GOODSST2/length GOODCMNIST/color; do #GOODMotif/basis GOODMotif/size GOODMotif2/basis GOODSST2/length GOODTwitter/length GOODHIV/scaffold GOODCMNIST/color LBAPcore/assay
       for MODEL in CIGA; do # CIGA LECI GSAT
              goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                     --seeds "1/2/3/4/5" \
                     --task eval_metric \
                     --metrics "suff++/nec" \
                     --splits "id_val" \
                     --average_edge_attn mean \
                     --mitigation_sampling raw \
                     --mitigation_readout weighted \
                     --model_name ${MODEL}GIN \
                     --gpu_idx 2 \
                     --mask  \
                     --debias \
                     --samplingtype deconfounded \
                     --nec_number_samples prop_G_dataset \
                     --save_metrics \
                     --log_id suff++_old_allmitig_paper
              echo "DONE ${MODEL} ${DATASET} ALL"
                     
                     # --save_metrics \
                     # --log_id suff++_old_allmitig_paper \
                     # --mitigation_readout weighted \
                     # --mitigation_expl_scores hard
       done
done

echo "DONE all :)"
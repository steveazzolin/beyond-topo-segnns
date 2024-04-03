set -e


for DATASET in GOODSST2/length GOODTwitter/length GOODHIV/scaffold LBAPcore/assay GOODCMNIST/color; do #GOODMotif/basis GOODMotif/size GOODMotif2/basis GOODSST2/length GOODTwitter/length GOODHIV/scaffold GOODCMNIST/color LBAPcore/assay
       for MODEL in LECI CIGA GSAT; do # CIGA LECI GSAT
              goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                     --seeds "1/2/3/4/5" \
                     --mitigation_sampling feat \
                     --task eval_metric \
                     --metrics "suff++/nec" \
                     --average_edge_attn mean \
                     --gpu_idx 2 \
                     --mask  \
                     --debias \
                     --samplingtype deconfounded \
                     --nec_number_samples prop_G_dataset \
                     --save_metrics \
                     --log_id suff++_old #suff_simple_old
              echo "DONE ${MODEL} ${DATASET}"
       done
done

echo "DONE all :)"

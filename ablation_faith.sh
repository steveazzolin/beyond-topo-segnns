set -e


for DATASET in GOODCMNIST/color GOODMotif2/basis GOODMotif/size; do #GOODMotif/basis GOODMotif/size GOODMotif2/basis GOODSST2/length GOODTwitter/length GOODHIV/scaffold GOODCMNIST/color LBAPcore/assay
       for MODEL in LECI; do # CIGA LECI GSAT
            for BUDGET in 4 8 12 16 20; do #100 500 800 1500 2500
                    goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                            --seeds "1/2/3/4/5" \
                            --task eval_metric \
                            --metrics "suff++/nec" \
                            --average_edge_attn mean \
                            --mitigation_sampling feat \
                            --gpu_idx 2 \
                            --mask  \
                            --debias \
                            --expval_budget ${BUDGET} \
                            --samplingtype deconfounded \
                            --nec_number_samples prop_G_dataset \
                            --save_metrics \
                            --log_id suff++_old_ablation_expval_budget_${BUDGET}
                        #     --log_id suff++_old_ablation_numsamples_budget_${BUDGET}                            
                        #     --numsamples_budget ${BUDGET} \
                    echo "DONE ${MODEL} ${DATASET}"
            done
       done
done

echo "DONE all :)"

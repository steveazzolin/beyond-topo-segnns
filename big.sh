set -e


for DATASET in GOODSST2/length GOODTwitter/length GOODHIV/scaffold ; do #GOODMotif/basis GOODMotif/size GOODSST2/length GOODTwitter/length GOODHIV/scaffold
       for MODEL in CIGA; do # CIGA LECI GSAT
              goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                     --seeds "1/2/3" \
                     --mitigation_sampling feat \
                     --task plot_panel \
                     --metrics "suff/nec++" \
                     --average_edge_attn mean \
                     --mask  
                     #--debias     
              echo "DONE ${MODEL} ${DATASET}"
       done
done

echo "DONE all :)"

set -e


for DATASET in GOODCMNIST/color; do #GOODMotif/basis GOODMotif/size GOODMotif2/size GOODTwitter/length GOODHIV/scaffold GOODCMNIST/color LBAPcore/assay
       for MODEL in CIGA GSAT; do # CIGA LECI GSAT
              goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                     --seeds "1/2/3" \
                     --mitigation_sampling feat \
                     --task plot_panel \
                     --metrics "nulla" \
                     --average_edge_attn default \
                     --gpu_idx 1 \
                     # --mask  \
                     # --debias     
              echo "DONE ${MODEL} ${DATASET}"
       done
done

echo "DONE all :)"

echo "Generating sampling type NEC plots :/"
set -e


for DATASET in GOODMotif/basis GOODMotif2/basis GOODSST2/length GOODTwitter/length GOODMotif/size GOODHIV/scaffold LBAPcore/assay GOODCMNIST/color; do
       goodtg --config_path final_configs/${DATASET}/covariate/LECI.yaml \
              --seeds "1/2/3/4/5" \
              --task plot_sampling \
              --average_edge_attn mean \
              --mitigation_sampling feat \
              --gpu_idx 2
       echo "DONE ${MODEL} ${DATASET}"
done

echo "DONE all :)"

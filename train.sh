set -e

echo "Time to train models!"

for DATASET in GOODMotif/size GOODSST2/length GOODTwitter/length GOODHIV/scaffold LBAPcore/assay; do #GOODMotif/basis GOODMotif/size GOODMotif2/basis GOODSST2/length GOODTwitter/length GOODHIV/scaffold GOODCMNIST/color LBAPcore/assay
       for MODEL in LECI; do
              goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                     --seeds "1/2/3/4/5" \
                     --mitigation_sampling feat \
                     --task train \
                     --average_edge_attn mean \
                     --gpu_idx 2 \
                     --mitigation_readout weighted \
                     --num_workers 2 \
                     --model_name LECIGIN
              echo "DONE TRAIN ${MODEL} ${DATASET}"
       done
done

echo "DONE all :)"

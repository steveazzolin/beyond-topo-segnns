set -e

echo "Time to train models!"

for DATASET in GOODSST2/length GOODTwitter/length GOODHIV/scaffold LBAPcore/assay GOODCMNIST/color; do #GOODMotif/basis GOODMotif/size GOODMotif2/basis GOODSST2/length GOODTwitter/length GOODHIV/scaffold GOODCMNIST/color LBAPcore/assay
       for MODEL in LECI; do
              goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                     --seeds "1/2/3/4/5" \
                     --mitigation_sampling feat \
                     --task train \
                     --average_edge_attn mean \
                     --gpu_idx 1 \
                     --mitigation_readout weighted \
                     --model_name LECIGIN \
                     --num_workers 2
              echo "DONE TRAIN ${MODEL} ${DATASET}"
       done
done

echo "DONE all :)"

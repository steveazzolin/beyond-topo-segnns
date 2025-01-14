set -e

echo "Time to train models!"

GPU=1
SEEDS="1/2/3/4/5/6/7/8/9/10"



for DATASET in MNIST/basis/no_shift; do
       goodtg --config_path final_configs/${DATASET}/SMGNN.yaml \
              --seeds ${SEEDS} \
              --task train \
              --average_edge_attn mean \
              --gpu_idx ${GPU} \
              --use_norm none \
              --global_side_channel simple_concept2temperature \
              --max_epoch 200

       echo "DONE TRAIN GL-SMGNN ${DATASET}"

       goodtg --config_path final_configs/${DATASET}/GSAT.yaml \
              --seeds ${SEEDS} \
              --task train \
              --average_edge_attn mean \
              --gpu_idx ${GPU} \
              --use_norm bn \
              --global_side_channel simple_concept2temperature \
              --max_epoch 200

       echo "DONE TRAIN GL-GSAT ${DATASET}"

       goodtg --config_path final_configs/${DATASET}/GiSST.yaml \
              --seeds ${SEEDS} \
              --task train \
              --average_edge_attn mean \
              --gpu_idx ${GPU} \
              --use_norm none \
              --max_epoch 200

       echo "DONE TRAIN GiSST ${DATASET}"

       goodtg --config_path final_configs/${DATASET}/GiSST.yaml \
              --seeds ${SEEDS} \
              --task train \
              --average_edge_attn mean \
              --gpu_idx ${GPU} \
              --use_norm none \
              --global_side_channel simple_concept2temperature \
              --max_epoch 200

       echo "DONE TRAIN GiSST ${DATASET}"

       


       goodtg --config_path final_configs/${DATASET}/ERM.yaml \
              --seeds ${SEEDS} \
              --task train \
              --gpu_idx ${GPU} \
              --use_norm bn \
              --max_epoch 200
       echo "DONE TRAIN ERM ${DATASET}"

       goodtg --config_path final_configs/${DATASET}/GSAT.yaml \
              --seeds ${SEEDS} \
              --task train \
              --average_edge_attn mean \
              --gpu_idx ${GPU} \
              --use_norm bn \
              --max_epoch 200
       echo "DONE TRAIN GSAT ${DATASET}"

       goodtg --config_path final_configs/${DATASET}/SMGNN.yaml \
              --seeds ${SEEDS} \
              --task train \
              --average_edge_attn mean \
              --gpu_idx ${GPU} \
              --use_norm none \
              --max_epoch 200
       echo "DONE TRAIN SMGNN ${DATASET}"
done








# for DATASET in GOODSST2/length/covariate; do
#        goodtg --config_path final_configs/${DATASET}/SMGNN.yaml \
#               --seeds ${SEEDS} \
#               --task train \
#               --average_edge_attn mean \
#               --gpu_idx ${GPU} \
#               --use_norm none \
#               --global_side_channel simple_product \
#               --mitigation_sampling raw              
#        echo "DONE TRAIN ${MODEL} ${DATASET} product"

#        goodtg --config_path final_configs/${DATASET}/SMGNN.yaml \
#               --seeds ${SEEDS} \
#               --task train \
#               --average_edge_attn mean \
#               --gpu_idx ${GPU} \
#               --use_norm none \
#               --global_side_channel simple_productscaled \
#               --mitigation_sampling raw              
#        echo "DONE TRAIN ${MODEL} ${DATASET} productscaled"

#        goodtg --config_path final_configs/${DATASET}/SMGNN.yaml \
#               --seeds ${SEEDS} \
#               --task train \
#               --average_edge_attn mean \
#               --gpu_idx ${GPU} \
#               --use_norm none \
#               --global_side_channel simple_godel \
#               --mitigation_sampling raw              
#        echo "DONE TRAIN ${MODEL} ${DATASET} godel"

#        goodtg --config_path final_configs/${DATASET}/SMGNN.yaml \
#               --seeds ${SEEDS} \
#               --task train \
#               --average_edge_attn mean \
#               --gpu_idx ${GPU} \
#               --use_norm none \
#               --global_side_channel simple_concept2 \
#               --mitigation_sampling raw              
#        echo "DONE TRAIN ${MODEL} ${DATASET} concept2"
# done

echo "DONE all :)"

set -e

echo "I'm computing faithfulness :)"
echo "The PID of this script is: $$"

SPLITS="id_val/val/test"
SEEDS="1/2/3/4/5/6/7/8/9/10"
NUMSAMPLES_BUDGET=9999999

for DATASET in TopoFeature/basis/no_shift; do #GOODMotif/basis GOODMotif/size GOODMotif2/basis GOODSST2/length GOODTwitter/length GOODHIV/scaffold GOODCMNIST/color LBAPcore/assay
       for NECALPHA in 0.05 0.1 0.01; do              
              
        #     WEIGHTS="0.5/0.8/0.9/0.95"
        #     goodtg --config_path final_configs/${DATASET}/GSAT.yaml \
        #             --seeds ${SEEDS} \
        #             --task eval_metric \
        #             --metrics "suff_simple/nec" \
        #             --splits ${SPLITS} \
        #             --ratios ${WEIGHTS} \
        #             --nec_alpha_1 ${NECALPHA} \
        #             --save_metrics \
        #             --log_id isweight \
        #             --numsamples_budget ${NUMSAMPLES_BUDGET} \
        #             --average_edge_attn mean \
        #             --use_norm none \
        #             --gpu_idx 0 \
        #             --samplingtype deconfounded \
        #             --nec_number_samples prop_G_dataset
        #     echo "DONE GSAT ${DATASET} NEC_ALPHA ${NECALPHA}"                     

        #     #   exit 1
                     
        #     WEIGHTS="0.01/0.1/0.25/0.6"
        #     goodtg --config_path final_configs/${DATASET}/SMGNN.yaml \
        #             --seeds ${SEEDS} \
        #             --task eval_metric \
        #             --metrics "suff_simple/nec" \
        #             --splits ${SPLITS} \
        #             --ratios ${WEIGHTS} \
        #             --samplingtype deconfounded \
        #             --nec_number_samples prop_G_dataset \
        #             --nec_alpha_1 ${NECALPHA} \
        #             --save_metrics \
        #             --log_id isweight \
        #             --numsamples_budget ${NUMSAMPLES_BUDGET} \
        #             --average_edge_attn mean \
        #             --global_pool sum \
        #             --gpu_idx 0 \
        #             --global_side_channel simple_concept2temperature \
        #             --extra_param True 10 0.01 \
        #             --ood_param 0.001 \
        #             --lr_filternode 0.001 \
        #             --lr 0.001 \
        #             --use_norm none
        #     echo "DONE SMGNN ${DATASET} NEC_ALPHA ${NECALPHA}"

       WEIGHTS="0.6/0.5/0.4/0.3"
       goodtg --config_path final_configs/${DATASET}/GiSST.yaml \
              --seeds ${SEEDS} \
              --task eval_metric \
              --metrics "suff_simple/nec" \
              --splits ${SPLITS} \
              --ratios ${WEIGHTS} \
              --save_metrics \
              --nec_alpha_1 ${NECALPHA} \
              --log_id isweight \
              --numsamples_budget ${NUMSAMPLES_BUDGET} \
              --average_edge_attn mean \
              --use_norm bn \
              --gpu_idx 1 \
              --samplingtype deconfounded \
              --nec_number_samples prop_G_dataset
       echo "DONE GiSST ${DATASET} NEC_ALPHA ${NECALPHA}"

       goodtg --config_path final_configs/${DATASET}/GiSST.yaml \
              --seeds ${SEEDS} \
              --task eval_metric \
              --metrics "suff_simple/nec" \
              --splits ${SPLITS} \
              --ratios ${WEIGHTS} \
              --save_metrics \
              --random_expl \
              --nec_alpha_1 ${NECALPHA} \
              --log_id isweight \
              --numsamples_budget ${NUMSAMPLES_BUDGET} \
              --average_edge_attn mean \
              --use_norm bn \
              --gpu_idx 1 \
              --samplingtype deconfounded \
              --nec_number_samples prop_G_dataset
       echo "DONE GiSST ${DATASET} NEC_ALPHA ${NECALPHA} RANDOM"

       #      WEIGHTS="0.01/0.1/0.25/0.6"
       #      goodtg --config_path final_configs/${DATASET}/SMGNN.yaml \
       #               --seeds ${SEEDS} \
       #               --task eval_metric \
       #               --metrics "suff_simple/nec" \
       #               --splits ${SPLITS} \
       #               --ratios ${WEIGHTS} \
       #               --samplingtype deconfounded \
       #               --nec_number_samples prop_G_dataset \
       #               --save_metrics \
       #               --log_id isweight \
       #               --nec_alpha_1 ${NECALPHA} \
       #               --numsamples_budget ${NUMSAMPLES_BUDGET} \
       #               --average_edge_attn mean \
       #               --global_pool sum \
       #               --gpu_idx 0 \
       #               --use_norm none
       #        echo "DONE SMGNN plain ${DATASET} NEC_ALPHA ${NECALPHA}"

       #        goodtg --config_path final_configs/${DATASET}/SMGNN.yaml \
       #               --seeds ${SEEDS} \
       #               --task eval_metric \
       #               --metrics "suff_simple/nec" \
       #               --splits ${SPLITS} \
       #               --ratios ${WEIGHTS} \
       #               --samplingtype deconfounded \
       #               --nec_number_samples prop_G_dataset \
       #               --save_metrics \
       #               --log_id isweight \
       #               --random_expl \
       #               --nec_alpha_1 ${NECALPHA} \
       #               --numsamples_budget ${NUMSAMPLES_BUDGET} \
       #               --average_edge_attn mean \
       #               --global_pool sum \
       #               --gpu_idx 0 \
       #               --use_norm none
       #        echo "DONE SMGNN plain ${DATASET} NEC_ALPHA ${NECALPHA} RANDOM"

       done
done

echo "DONE all :)"
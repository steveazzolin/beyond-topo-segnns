set -e

echo "I'm tuning the hyper-parameters of GSAT :)"
echo "The PID of this script is: $$"

for MODEL in GSAT; do # GSAT SMGNN
    for OOD_PARAM in 0.1 1 10; do
        for EXTRA_PARAM in 0.7 0.5 0.3; do
            for WD in 0.0 0.1 0.001; do
                for NORM in none bn in; do
                    for DATASET in TopoFeature/basis BAColor/basis; do #GOODMotif/basis GOODMotif/size GOODMotif2/basis GOODSST2/length GOODTwitter/length GOODHIV/scaffold GOODCMNIST/color LBAPcore/assay   
                        goodtg --config_path final_configs/${DATASET}/no_shift/${MODEL}.yaml \
                                --seeds "1/2/3/4/5/6/7/8/9/10" \
                                --task train \
                                --average_edge_attn mean \
                                --gpu_idx 0 \
                                --ood_param ${OOD_PARAM}\
                                --extra_param True 10 ${EXTRA_PARAM}\
                                --weight_decay ${WD}\
                                --use_norm ${NORM}
                        echo "DONE ${DATASET} ${OOD_PARAM};True 10 ${EXTRA_PARAM};${WD};${NORM}"
                    done
                done
            done
        done
    done
done

echo "DONE all :)"




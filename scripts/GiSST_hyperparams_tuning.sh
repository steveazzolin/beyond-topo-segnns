set -e

echo "I'm tuning the hyper-parameters of GiSST :)"
echo "The PID of this script is: $$"

for MODEL in GiSST; do # GSAT SMGNN
    for EDGE_L1 in 0.01 0.1 1; do
        for EDGE_ENTR in 0.01 0.1 1; do
            for FEAT_L1 in 0.01 0.1 1; do
                for FEAT_ENTR in 0.01 0.1 1; do
                    for WD in 0.0; do
                        for NORM in none bn; do
                            for DATASET in TopoFeature/basis BAColor/basis; do #GOODMotif/basis GOODMotif/size GOODMotif2/basis GOODSST2/length GOODTwitter/length GOODHIV/scaffold GOODCMNIST/color LBAPcore/assay   
                                goodtg --config_path final_configs/${DATASET}/no_shift/${MODEL}.yaml \
                                        --seeds "1/2/3/4/5" \
                                        --task train \
                                        --average_edge_attn mean \
                                        --gpu_idx 1 \
                                        --extra_param True ${EDGE_L1} ${EDGE_ENTR} ${FEAT_L1} ${FEAT_ENTR}\
                                        --weight_decay ${WD}\
                                        --use_norm ${NORM}
                                echo "DONE ${DATASET} ${OOD_PARAM};True ${EDGE_L1};${EDGE_ENTR};${FEAT_L1};${FEAT_ENTR};${WD};${NORM}"
                            done
                        done
                    done
                done
            done
        done
    done
done

echo "DONE all :)"




# for running experiments on GSAT and CIGA

echo "Time to compute metrics!"
echo "The PID of this script is: $$"
set -e


for DATASET in GOODMotif2/basis LBAPcore/assay GOODMotif/basis GOODMotif/size GOODTwitter/length GOODHIV/scaffold GOODCMNIST/color GOODSST2/length; do #GOODMotif/basis GOODMotif/size GOODMotif2/basis GOODSST2/length GOODTwitter/length GOODHIV/scaffold GOODCMNIST/color LBAPcore/assay
       for MODEL in LECI GSAT CIGA; do # CIGA LECI GSAT
              goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
                     --seeds "1/2/3/4/5" \
                     --task plot_sampling \
                     --average_edge_attn mean \
                     --mitigation_sampling feat \
                     --gpu_idx 0
              echo "DONE ${MODEL} ${DATASET} plotting"
       done
done

echo "DONE all :)"
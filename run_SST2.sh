set -e

goodtg --config_path final_configs/GOODSST2/length/covariate/LECI.yaml \
       --seeds "1/2/3/4/5" \
       --mitigation_sampling feat \
       --average_edge_attn mean 
echo "DONE LECI"

goodtg --config_path final_configs/GOODSST2/length/covariate/CIGA.yaml \
       --seeds "1/2/3/4/5" \
       --mitigation_sampling feat \
       --average_edge_attn mean 
echo "DONE CIGA"

goodtg --config_path final_configs/GOODSST2/length/covariate/GSAT.yaml \
       --seeds "1/2/3/4/5" \
       --mitigation_sampling feat \
       --average_edge_attn mean 
echo "DONE GSAT"
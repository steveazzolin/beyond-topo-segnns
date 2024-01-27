set -e

goodtg --config_path final_configs/GOODCMNIST/color/covariate/GSAT.yaml \
       --seeds "1/2/3/4/5" \
       --mitigation_sampling feat \
       --task eval_metric \
       --metrics "suff/nec/nec++/fidp/fidm" \
       --average_edge_attn mean \
       --debias \
       --mask
echo "DONE metrics GSAT"

goodtg --config_path final_configs/GOODCMNIST/color/covariate/GSAT.yaml \
       --seeds "1/2/3/4/5" \
       --mitigation_sampling feat \
       --task eval_metric \
       --metrics "acc" \
       --average_edge_attn mean \
       --debias \
       --mask
echo "DONE acc GSAT"



goodtg --config_path final_configs/GOODCMNIST/color/covariate/LECI.yaml \
       --seeds "1/2/3/4/5" \
       --mitigation_sampling feat \
       --task eval_metric \
       --metrics "suff/nec/nec++/fidp/fidm" \
       --average_edge_attn mean \
       --debias \
       --mask
echo "DONE metrics LECI"

goodtg --config_path final_configs/GOODCMNIST/color/covariate/LECI.yaml \
       --seeds "1/2/3/4/5" \
       --mitigation_sampling feat \
       --task eval_metric \
       --metrics "acc" \
       --average_edge_attn mean \
       --debias \
       --mask
echo "DONE acc LECI"



goodtg --config_path final_configs/GOODCMNIST/color/covariate/CIGA.yaml \
       --seeds "1/2/3/4/5" \
       --mitigation_sampling feat \
       --task eval_metric \
       --metrics "suff/nec/nec++/fidp/fidm" \
       --average_edge_attn mean \
       --debias \
       --mask
echo "DONE metrics CIGA"

goodtg --config_path final_configs/GOODCMNIST/color/covariate/CIGA.yaml \
       --seeds "1/2/3/4/5" \
       --mitigation_sampling feat \
       --task eval_metric \
       --metrics "acc" \
       --average_edge_attn mean \
       --debias \
       --mask
echo "DONE acc CIGA"
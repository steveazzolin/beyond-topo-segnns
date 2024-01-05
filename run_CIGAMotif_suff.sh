#!/bin/bash
#SBATCH -p long-disi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=suff
#SBATCH -t 0-00:30
#SBATCH --output=/home/steve.azzolin/sedignn/LECI_fork/sbatch_outputs/CIGA_GOODMotif_suff.txt
#SBATCH --error=/home/steve.azzolin/sedignn/LECI_fork/sbatch_outputs/CIGA_GOODMotif_suff.txt
#SBATCH --ntasks=1
#SBATCH -N 1

set -e


goodtg --config_path final_configs/GOODMotif/basis/covariate/CIGA.yaml \
       --seeds "1" \
       --mitigation_sampling feat \
       --task eval_suff

echo "DONE feat"

goodtg --config_path final_configs/GOODMotif/basis/covariate/CIGA.yaml \
       --seeds "1" \
       --mitigation_sampling feat \
       --mitigation_backbone soft \
       --task eval_suff

echo "DONE feat+soft"

goodtg --config_path final_configs/GOODMotif/basis/covariate/CIGA.yaml \
       --seeds "1" \
       --mitigation_sampling feat \
       --mitigation_backbone soft2 \
       --task eval_suff

echo "DONE feat+soft2"


goodtg --config_path final_configs/GOODMotif/basis/covariate/CIGA.yaml \
       --seeds "1" \
       --mitigation_sampling feat \
       --mitigation_backbone hard \
       --task eval_suff

echo "DONE feat+hard"
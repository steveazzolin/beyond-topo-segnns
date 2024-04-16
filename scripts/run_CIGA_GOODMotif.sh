#!/bin/bash
#SBATCH -p long-disi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --job-name=CIGA_GOODMotif
#SBATCH -t 0-10:00
#SBATCH --output=/home/steve.azzolin/sedignn/LECI_fork/sbatch_outputs/CIGA_GOODMotif_mitigation.txt
#SBATCH --error=/home/steve.azzolin/sedignn/LECI_fork/sbatch_outputs/CIGA_GOODMotif_mitigation.txt
#SBATCH --ntasks=1
#SBATCH -N 1

goodtg --config_path final_configs/GOODMotif/basis/covariate/CIGA.yaml --seeds "1/2/3" --mitigation_backbone "soft"

echo "DONE FIRST"

goodtg --config_path final_configs/GOODMotif/basis/covariate/CIGA.yaml --seeds "1/2/3" --mitigation_backbone "soft2"

echo "DONE ALL"
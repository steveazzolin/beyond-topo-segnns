#!/bin/bash
#SBATCH -p long-disi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=LECI
#SBATCH -t 0-10:00
#SBATCH --output=/home/steve.azzolin/sedignn/LECI_fork/sbatch_outputs/LECI_GOODMotif.txt
#SBATCH --error=/home/steve.azzolin/sedignn/LECI_fork/sbatch_outputs/LECI_GOODMotif.txt
#SBATCH --ntasks=1
#SBATCH -N 1

goodtg --config_path final_configs/GOODMotif/basis/covariate/LECI.yaml --seeds "1/2/3"
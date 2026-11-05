#!/bin/bash
# 06b only (rims), one task per heart. Use after a change to 06b_rims.py; 05b and 06 outputs are reused.
#SBATCH --job-name=rh100_rims
#SBATCH --array=1-94
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/storage/homefs/pb26a271/data/inselspital/rh100/logs/rims_%A_%a.out
module load Python/3.12.3-GCCcore-13.3.0
source ~/venvs/rh/bin/activate
RH=~/data/inselspital/rh100
ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $RH/qc/included_cases.txt | tr -d '\r')
cd ~/code/ImageSegmentation/scripts
python 06b_rims.py --cases_dir $RH/cases --ids $ID

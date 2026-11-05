#!/bin/bash
#SBATCH --job-name=rh100_surf
#SBATCH --array=1-94
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/storage/homefs/pb26a271/data/inselspital/rh100/logs/surf_%A_%a.out
# 05b openings (centre, axis) -> 06 SSM input surface with interface footprints -> 06b rims, one heart per task
module load Python/3.12.3-GCCcore-13.3.0
source ~/venvs/rh/bin/activate
RH=~/data/inselspital/rh100
ID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $RH/qc/included_cases.txt | tr -d '\r')
cd ~/code/ImageSegmentation/scripts
echo "case $ID on $(hostname)"
python 05b_interface_orifices.py --cases_dir $RH/cases --ids $ID
python 06_labelled_surface.py --case_dir $RH/cases/$ID
python 06b_rims.py --cases_dir $RH/cases --ids $ID

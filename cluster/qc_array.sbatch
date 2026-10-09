#!/bin/bash
# Stage 2: screening table + plots, one CPU task per group.
#SBATCH --job-name=ts_qc
#SBATCH --account=gratis
#SBATCH --partition=epyc2
#SBATCH --qos=job_gratis
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-4
#SBATCH --output=%x_%A_%a.out

source "$HOME/thesis/ImageSegmentation/cluster/paths.sh"
activate_env
set -eo pipefail

group=${IMAGECAS_GROUPS[$SLURM_ARRAY_TASK_ID]}
python "$CODE/qc_screen_cases.py" --cases_dir "$CASES" --screen_dir "$(screen_dir_of "$group")"
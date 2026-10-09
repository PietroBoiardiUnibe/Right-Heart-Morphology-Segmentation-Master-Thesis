#!/bin/bash
# Stage 4 (submit by hand after reviewing the cohort plots): veins + PA landmarks on the cohort.
#SBATCH --job-name=ts_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-19%10
#SBATCH --output=%x_%A_%a.out

source "$HOME/thesis/ImageSegmentation/cluster/paths.sh"
activate_env
private_totalseg_config
set -eo pipefail
source "$HOME/thesis/ImageSegmentation/cluster/preflight.sh"     # stop at once if the environment is broken

python "$CODE/02b_run_totalsegmentator_batch.py" --stage full \
    --selection "$COHORT/selection.txt" --cases_dir "$CASES" \
    --chunk "$SLURM_ARRAY_TASK_ID" --n_chunks 20
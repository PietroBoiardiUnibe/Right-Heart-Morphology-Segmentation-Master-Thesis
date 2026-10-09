#!/bin/bash
# Stage 1: heartchambers_highres on EVERY case of all 5 ImageCAS groups (5 x 10 tasks).
#SBATCH --job-name=ts_screen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=gratis
#SBATCH --partition=gpu
#SBATCH --qos=job_gratis
#SBATCH --gres=gpu:rtx4090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-49%10
#SBATCH --output=%x_%A_%a.out

source "$HOME/thesis/ImageSegmentation/cluster/paths.sh"
activate_env
private_totalseg_config
set -eo pipefail
source "$HOME/thesis/ImageSegmentation/cluster/preflight.sh"     # stop at once if the environment is broken

group=${IMAGECAS_GROUPS[$((SLURM_ARRAY_TASK_ID / CHUNKS_PER_GROUP))]}
chunk=$((SLURM_ARRAY_TASK_ID % CHUNKS_PER_GROUP))
echo "task $SLURM_ARRAY_TASK_ID -> group $group, chunk $chunk/$CHUNKS_PER_GROUP on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python "$CODE/02b_run_totalsegmentator_batch.py" --stage screen --n 0 \
    --group "$group" --zip "$RAW/$group.zip" \
    --cases_dir "$CASES" --screen_dir "$(screen_dir_of "$group")" \
    --chunk "$chunk" --n_chunks "$CHUNKS_PER_GROUP"
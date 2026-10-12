#!/bin/bash
# Stage 5 (CPU): masks -> SSM input on the cohort. Runs the steps listed in STEPS for every
# case of the cohort selection; a step whose output already exists is skipped (resumable).
#   04 -> mask_rh.nii.gz      05 -> mask_rh_cut.nii.gz + orifices.json      06 -> surface_rh.vtp
# Submit e.g.:  sbatch --export=ALL,STEPS="04 05" bridge_array.sbatch
#SBATCH --job-name=rh_bridge
#SBATCH --account=gratis
#SBATCH --partition=epyc2
#SBATCH --qos=job_gratis
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-19%10
#SBATCH --output=%x_%A_%a.out

source "$HOME/thesis/ImageSegmentation/cluster/paths.sh"
activate env
set -o pipefail

STEPS=${STEPS:-"04 05 06"}
declare -A SCRIPT=([04]=04_merge_right_heart_labels.py [05]=05_cut_orifices.py [06]=06_labelled_surface.py)
declare -A OUTPUT=([04]=mask_rh.nii.gz [05]=mask_rh_cut.nii.gz [06]=surface_rh.vtp)
N_TASKS=20

mapfile -t ids < <(tr -d '\r' < "$COHORT/selection.txt" | grep -v '^$')
echo "task $SLURM_ARRAY_TASK_ID: steps [$STEPS], every ${N_TASKS}th of ${#ids[@]} cases"
for ((k = SLURM_ARRAY_TASK_ID; k < ${#ids[@]}; k += N_TASKS)); do
    case_dir=$CASES/cas_$(printf %04d "${ids[k]}")
    for step in $STEPS; do
        [ -f "$case_dir/${OUTPUT[$step]}" ] && continue                 # already done
        if ! python "$CODE/${SCRIPT[$step]}" --case_dir "$case_dir"; then
            echo "FAILED $step $case_dir"
            break                                                         # later steps need this one
        fi
    done
done
echo "task $SLURM_ARRAY_TASK_ID done"
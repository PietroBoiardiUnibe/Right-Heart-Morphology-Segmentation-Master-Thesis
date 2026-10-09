#!/bin/bash
# Stage 3: pool the 5 group screens -> one cohort, apply the selection rules to the pooled
# distribution, and write the exploratory plots (per group + cohort).
#SBATCH --job-name=ts_select
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=gratis
#SBATCH --partition=epyc2
#SBATCH --qos=job_gratis
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out

source "$HOME/thesis/ImageSegmentation/cluster/paths.sh"
activate_env
set -eo pipefail

screens=()
for g in "${IMAGECAS_GROUPS[@]}"; do screens+=("$(screen_dir_of "$g")"); done
tables=()
for s in "${screens[@]}"; do tables+=("$s/screen_table.csv"); done

python "$CODE/pool_screens.py" "${screens[@]}" --out_dir "$COHORT"
python "$CODE/select_cases.py" --cases_dir "$CASES" --screen_dir "$COHORT"
python "$CODE/explore_screen_table.py" "${tables[@]}" --out_dir "$RH/qc" --name imagecas_groups
python "$CODE/explore_screen_table.py" "$COHORT/screen_table.csv" --out_dir "$COHORT/qc" --name cohort
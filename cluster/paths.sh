# Shared settings for every UBELIX job (each .sbatch file sources this).
CODE=$HOME/thesis/ImageSegmentation/scripts
RAW=$HOME/data/ImageCAS/raw/imagecas               # 1-200.zip ... 801-1000.zip (+ .z01-.z04 parts)
RH=$HOME/data/rh_test
CASES=$RH/cases
COHORT=$RH/cohorts/imagecas_v1
IMAGECAS_GROUPS=(1-200 201-400 401-600 601-800 801-1000)
CHUNKS_PER_GROUP=10                                 # 10 screen tasks per group -> 20 cases per task

screen_dir_of() {                                   # 1-200 -> $RH/screens/g0001-0200_all
    local first=${1%-*} last=${1#*-}
    printf "%s/screens/g%04d-%04d_all" "$RH" "$first" "$last"
}

activate_env() {
    module load Anaconda3
    eval "$(conda shell.bash hook)"
    conda activate totalseg
}

private_totalseg_config() {
    # Every TotalSegmentator run rewrites config.json (prediction counter). Parallel tasks
    # sharing one file can corrupt it, so each task gets its own copy; weights stay shared.
    export TOTALSEG_WEIGHTS_PATH=$HOME/.totalsegmentator/nnunet/results
    export TOTALSEG_HOME_DIR=${TMPDIR:-/tmp}/totalseg_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
    mkdir -p "$TOTALSEG_HOME_DIR"
    cp "$HOME/.totalsegmentator/config.json" "$TOTALSEG_HOME_DIR/"
}
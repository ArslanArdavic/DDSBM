#!/bin/bash
#SBATCH --account=project_465002822
#SBATCH --partition=small-g
#SBATCH --job-name=ddsbm_zinc_analysis_random
#SBATCH --output=/project/project_465002822/DDSBM/slurm/log/repro_zinc_analysis_random_%j.out
#SBATCH --error=/project/project_465002822/DDSBM/slurm/log/repro_zinc_analysis_random_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem-per-gpu=60G
#SBATCH --time=24:00:00

PRJ=project_465002822
DDSBM_SRC=/project/$PRJ/DDSBM
SIF=/project/$PRJ/containers/graph-found-20260401.sif

# Experiment identifiers — must match training and test runs exactly
EXP_NAME=2026-04-02_SB_0.999_repro
DATASET=zinc
IDX=5
DIRECTION=forward
SEED=42
NUM_WORKERS=6

OUTPUTS_DIR=$DDSBM_SRC/outputs/$DATASET/$EXP_NAME
DATA_DIR=$DDSBM_SRC/data/$DATASET/$EXP_NAME
RESULTS_DIR=$DDSBM_SRC/results
ANALYSIS_DIR=$DDSBM_SRC/experiments/analysis

# Generated .pt file from test run — nfe100 because diffusion_steps=100
TEST_RESULT_DIR=$OUTPUTS_DIR/test_${DIRECTION}_${IDX}_last
GEN_PT=$TEST_RESULT_DIR/generated_joint_test_seed${SEED}_nfe100.pt

ITERATIONS=5   #5

# Expected output files — used for idempotency checks
RESULT_CSV=$TEST_RESULT_DIR/result_${DIRECTION}_seed${SEED}.csv
NLL_OUT=$RESULTS_DIR/nll/${DATASET}-${EXP_NAME}-${DIRECTION}-${SEED}.csv
NSPDK_OUT=$RESULTS_DIR/nspdk/${DATASET}-${EXP_NAME}-${DIRECTION}-${SEED}.csv
VAL_PROPS_OUT=$RESULTS_DIR/val_props/${DATASET}-${EXP_NAME}-${DIRECTION}-val-${SEED}.csv
PROP_DIFF_OUT=$RESULTS_DIR/prop_diff/${DATASET}-${EXP_NAME}-${DIRECTION}-${SEED}.csv
FCD_OUT=$RESULTS_DIR/fcd/${DATASET}-${EXP_NAME}-${DIRECTION}-${SEED}.csv

# Singularity + LUMI bindings
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# MIOpen temp folder (required for AMD GPUs)
MIOPEN_DIR=$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_DIR/cache
export MIOPEN_USER_DB=$MIOPEN_DIR/config

export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0

echo "============================================================"
echo "Step 1: graph_to_mol — convert generated graphs to SMILES"
echo "============================================================"
if [ -f $RESULT_CSV ]; then
    echo "SKIP — $RESULT_CSV already exists"
else
    singularity exec \
        -B /scratch/$PRJ,/project/$PRJ \
        --env PYTHONPATH=$DDSBM_SRC/src \
        $SIF python $DDSBM_SRC/experiments/graph_to_mol.py \
            $GEN_PT \
            --num_workers $NUM_WORKERS
fi

echo "============================================================"
echo "Step 2: NLL analysis"
echo "============================================================"
if [ -f $NLL_OUT ]; then
    echo "SKIP — $NLL_OUT already exists"
else
    singularity exec \
        -B /scratch/$PRJ,/project/$PRJ \
        --env PYTHONPATH=$DDSBM_SRC/src \
        $SIF python $ANALYSIS_DIR/nll.py \
            --data_path $DATA_DIR \
            --seeds $SEED \
            --direction $DIRECTION \
            --iterations $ITERATIONS
fi

echo "============================================================"
echo "Step 3: NSPDK analysis"
echo "============================================================"
if [ -f $NSPDK_OUT ]; then
    echo "SKIP — $NSPDK_OUT already exists"
else
    singularity exec \
        -B /scratch/$PRJ,/project/$PRJ \
        --env PYTHONPATH=$DDSBM_SRC/src \
        $SIF bash -c "PYTHONHASHSEED=0 python $ANALYSIS_DIR/nspdk.py \
            --experiment_path $OUTPUTS_DIR \
            --seeds $SEED \
            --direction $DIRECTION \
            --iterations $ITERATIONS \
            --num_workers 1"
fi

echo "============================================================"
echo "Step 4: Validity + LogP Wasserstein distance"
echo "============================================================"
if [ -f $VAL_PROPS_OUT ]; then
    echo "SKIP — $VAL_PROPS_OUT already exists"
else
    singularity exec \
        -B /scratch/$PRJ,/project/$PRJ \
        --env PYTHONPATH=$DDSBM_SRC/src \
        $SIF python $ANALYSIS_DIR/val_prop_wd.py \
            --experiment_path $OUTPUTS_DIR \
            --seeds $SEED \
            --direction $DIRECTION \
            --iterations $ITERATIONS \
            --num_workers $NUM_WORKERS
fi

echo "============================================================"
echo "Step 5: QED + SAscore MAD (prop_diff)"
echo "============================================================"
if [ -f $PROP_DIFF_OUT ]; then
    echo "SKIP — $PROP_DIFF_OUT already exists"
else
    singularity exec \
        -B /scratch/$PRJ,/project/$PRJ \
        --env PYTHONPATH=$DDSBM_SRC/src \
        $SIF python $ANALYSIS_DIR/prop_diff.py \
            --experiment_path $OUTPUTS_DIR \
            --seeds $SEED \
            --direction $DIRECTION \
            --iterations $ITERATIONS \
            --num_workers $NUM_WORKERS
fi

echo "============================================================"
echo "Step 6: FCD analysis (GPU recommended)"
echo "============================================================"
if [ -f $FCD_OUT ]; then
    echo "SKIP — $FCD_OUT already exists"
else
    singularity exec \
        -B /scratch/$PRJ,/project/$PRJ \
        --env PYTHONPATH=$DDSBM_SRC/src \
        $SIF python $ANALYSIS_DIR/fcd.py \
            --experiment_path $OUTPUTS_DIR \
            --seeds $SEED \
            --direction $DIRECTION \
            --iterations $ITERATIONS \
            --num_workers $NUM_WORKERS
fi

echo "============================================================"
echo "Analysis complete. Results in $RESULTS_DIR"
echo "============================================================"

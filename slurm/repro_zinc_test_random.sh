#!/bin/bash
#SBATCH --account=project_465002822
#SBATCH --partition=standard-g
#SBATCH --job-name=ddsbm_zinc_test_random
#SBATCH --output=/project/project_465002822/DDSBM/slurm/log/repro_zinc_test_random_%j.out
#SBATCH --error=/project/project_465002822/DDSBM/slurm/log/repro_zinc_test_random_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=48:00:00

PRJ=project_465002822
DDSBM_SRC=/project/$PRJ/DDSBM
SIF=/project/$PRJ/containers/graph-found-20260401.sif

# Experiment identifiers — must match the training run exactly
EXP_NAME=2026-04-02_SB_0.999_repro   # date-prefixed name created by training
DATASET=zinc
IDX=5                                  # last SB outer loop index (0-indexed, outer_loops=6 → IDX=5)
SEED=42

CKPT=$DDSBM_SRC/outputs/$DATASET/$EXP_NAME/forward_${IDX}/checkpoints/last.ckpt

# Slurm log directory
mkdir -p $DDSBM_SRC/slurm/log

# Singularity + LUMI bindings
module purge
module use /appl/local/laifs/modules
module load lumi-aif-singularity-bindings

# MIOpen temp folder (required for AMD GPUs)
MIOPEN_DIR=$(mktemp -d)
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_DIR/cache
export MIOPEN_USER_DB=$MIOPEN_DIR/config

# Wandb monitoring
export WANDB_API_KEY="60eae699ddc5a31f103b2c7be45a2c4115cae2bd"

# RCCL — Slingshot interfaces and GPU RDMA
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export NCCL_NET_GDR_LEVEL=PHB

export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0

srun singularity run \
    -B /scratch/$PRJ,/project/$PRJ \
    --env PYTHONPATH=$DDSBM_SRC/src \
    $SIF python $DDSBM_SRC/src/main.py \
        --config-name config_test \
        general.test_only=$CKPT \
        general.gpus=8 \
        general.seed=$SEED \
        general.name=SB_0.999_repro \
        general.chains_to_save=0 \
        general.final_model_samples_to_save=0 \
        general.final_model_chains_to_save=0
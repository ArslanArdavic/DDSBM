#!/bin/bash
#SBATCH --account=project_465002822
#SBATCH --partition=standard-g
#SBATCH --job-name=ddsbm_zinc_train_random
#SBATCH --output=/project/project_465002822/DDSBM/slurm/log/repro_zinc_bridge_train_random_%j.out
#SBATCH --error=/project/project_465002822/DDSBM/slurm/log/repro_zinc_bridge_train_random_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=480G
#SBATCH --time=48:00:00

PRJ=project_465002822
DDSBM_SRC=/project/$PRJ/DDSBM
SIF=/project/$PRJ/containers/graph-found-20260401.sif

# Slurm log directory
mkdir -p $DDSBM_SRC/slurm/log/ddsbm

# Redirect outputs to /scratch via symlink (create once, idempotent)
mkdir -p /scratch/$PRJ/DDSBM/outputs
if [ ! -L $DDSBM_SRC/outputs ]; then
    ln -s /scratch/$PRJ/DDSBM/outputs $DDSBM_SRC/outputs
fi

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

EXP=Bridge
DATASET=zinc
MIN_ALPHA=0.999
EPOCHS=3000  # NOTE: We should change epochs to compare bridge model with SB model
SAVE_EVERY=100  # NOTE: We should save checkpoints of Bridge model for every SB model's n_epoch (or its divisor)
NUM_GPUS_TO_USE=8

EXP_NAME=${EXP}_${MIN_ALPHA}

srun singularity run \
    -B /scratch/$PRJ,/project/$PRJ \
    --env PYTHONPATH=$DDSBM_SRC/src \
    $SIF python $DDSBM_SRC/src/main.py \
        experiment=bridge \
    	dataset.name=${DATASET} \
    	general.name=${EXP_NAME} \
    	general.gpus=${NUM_GPUS_TO_USE} \
    	general.save_every_n_epochs=${SAVE_EVERY} \
    	model.min_alpha=${MIN_ALPHA} \
    	train.n_epochs=${EPOCHS} \
        general.chains_to_save=0 \
        general.samples_to_save=0 \
        graph_match.gpus=1 \
	experiment.current_direction=forward




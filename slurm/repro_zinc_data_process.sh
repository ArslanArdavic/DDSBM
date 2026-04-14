#!/bin/bash
#SBATCH --account=project_465002822
#SBATCH --partition=dev-g
#SBATCH --job-name=ddsbm_zinc_test_random
#SBATCH --output=/project/project_465002822/DDSBM/slurm/log/data_zinc_random_%j.out
#SBATCH --error=/project/project_465002822/DDSBM/slurm/log/data_zinc_random_%j.err
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1            # Number of GPUs per node (max of 8)
#SBATCH --ntasks=1          
#SBATCH --cpus-per-task=7           # Use --gpus-per-node*7 CPUs on LUMI-G nodes
#SBATCH --mem-per-gpu=60G           
#SBATCH --time=00:30:00               # time limit

PRJ=project_465002822
DDSBM_SRC=/project/$PRJ/DDSBM
SIF=/project/$PRJ/containers/graph-found-20260401.sif

ORIGINAL_CSV_FILE=$DDSBM_SRC/data/raw/ZINC250k_logp_2_4_random_matched_no_nH.csv
DATASET_NAME=zinc
ORIGINAL_DATA_DIR=ZINC250k_logp_2_4_random_matched_no_nH

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
    $SIF python $DDSBM_SRC//data/process_data.py \
        ${ORIGINAL_CSV_FILE} \
        --dataset_name ${DATASET_NAME} \
        --original_data_dir ${ORIGINAL_DATA_DIR}


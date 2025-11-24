#!/bin/bash

#SBATCH --array=1-100           # how many tasks in the array
#SBATCH --cpus-per-task=16      # number of cores
#SBATCH --nodes=1               # number of nodes
#SBATCH --mem=1GB               # memory per __node__

# set -euo pipefail
# cd "$SLURM_SUBMIT_DIR"

# module purge
module load python/3.11
echo "CPUs per task: $SLURM_CPUS_PER_TASK"

srun python3 CB_nonegpm25mpm1_K3_95th_bootstrap.py 



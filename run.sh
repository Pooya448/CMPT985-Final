#!/bin/bash -l
#
## This is your working directory
#SBATCH -D /u/gtiwari/ARCH/
#
# Job Name Shown in the Queue (squeue)
#SBATCH -J ARCH
#
## Standard ouput and error from your program. These are relative to the working directory
#SBATCH -o ./logs/run/out.run
#SBATCH -e ./logs/run/err.run
#
# We are interested in nodes with GPUs, i.e., GPU Queue (Partition):
#SBATCH --partition=gpu
#
# Node feature:
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:v100:1       # If using only 1 GPU of a shared node
#SBATCH --mem=92500
#
# Number of nodes and MPI tasks per node:
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     # This stays 1 for 2 GPU training too, in most cases
#SBATCH --cpus-per-task=16      # Number of processes ?? (Each GPU has 16 workers ??)
#
#
# wall clock limit:
#SBATCH --time=03:00:00         #Specify a smaller duration if your script takes less time

module load gcc/8
module load git/2.31
module load cuda/10.2
module load cudnn/8.2.1
module load anaconda/3/2020.02

source /mpcdf/soft/SLE_12/packages/x86_64/anaconda/3/2020.02/etc/profile.d/conda.sh
conda activate arch

cd ../Models/ARCH/
rm -rf *
cd ../../ARCH

python trainer_arch.py --config ./configs/arch.yaml

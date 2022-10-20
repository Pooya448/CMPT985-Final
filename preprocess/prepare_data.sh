#!/bin/bash -l
#
## This is your working directory
#SBATCH -D /u/gtiwari/ARCH/
#
# Job Name Shown in the Queue (squeue)
#SBATCH -J PREP
#
## Standard ouput and error from your program. These are relative to the working directory
#SBATCH -o ./logs/prepare/out.prepare
#SBATCH -e ./logs/prepare/err.prepare
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
module load anaconda/3/2020.02

source /mpcdf/soft/SLE_12/packages/x86_64/anaconda/3/2020.02/etc/profile.d/conda.sh
conda activate arch

cd preprocess/
python prepare_data.py --data '../../datasets/mgn/' --views 360 --cam 'orth'

cd ../../datasets/mgn/125611487366942/
cp norm_render_360/0.png norm_render_360/90.png norm_render_360/180.png norm_render_360/270.png ../../../ARCH/logs/
cd ../../../ARCH/
git add .
git commit -m "render complete"
git push

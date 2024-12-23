#!/bin/bash
#$ -S /bin/bash
#$ -cwd
#$ -N dominance
#$ -j y
#$ -l mem_free=2G
#$ -l h_rt=24:00:00
#$ -t 1-11
#$ -pe smp 8
#$ -o $HOME/logs
module load CBI miniconda3/23.5.2-0-py311
conda activate brainspace
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python3 diffusion_neuromaps/src/dominance.py
[[ -n "$JOB_ID" ]] && qstat -j "$JOB_ID"

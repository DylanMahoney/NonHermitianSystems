#!/usr/bin/bash
#SBATCH --job-name=test_job
#SBATCH --output=../SlurmOutputs/test_job.%j.out
#SBATCH --error=../SlurmOutputs/test_job.%j.err
#SBATCH --time=00:05:00
#SBATCH -c 1
#SBATCH --mail-type=all
#SBATCH --chdir=../ScriptsForDataGeneration

ml load system
ml load texlive/2019
ml load viz
ml load python/3.9.0
ml load py-numpy/1.24.2_py39
ml load py-matplotlib/3.7.1_py39
ml load py-scipy/1.10.1_py39
pip3 install --user imageio

mkdir TestWorked
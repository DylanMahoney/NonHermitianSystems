#!/usr/bin/bash
#SBATCH --job-name=HN_spin_spin_transport_job
#SBATCH --output=../SlurmOutputs/HN_spin_spin_transport_job.%j.out
#SBATCH --error=../SlurmOutputs/HN_spin_spin_transport_job.%j.err
#SBATCH --time=6:00:00
#SBATCH -c 1
#SBATCH --mem=8GB
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

python3.9 HN_spin_spin_transport.py
#!/usr/bin/bash
#SBATCH --job-name=random_potential_spin_spin_transport_job
#SBATCH --output=../SlurmOutputs/random_potential_spin_spin_transport_job.%j.out
#SBATCH --error=../SlurmOutputs/random_potential_spin_spin_transport_job.%j.err
#SBATCH --time=20:00:00
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

python3.9 -u random_potential_spin_spin_transport.py

#In this past this has run out of memory when given only 8GB
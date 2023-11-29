#!/usr/bin/bash
#SBATCH --job-name=random_potential_eigenvalue_statistics_job
#SBATCH --output=../SlurmOutputs/random_potential_eigenvalue_statistics_job.%j.out
#SBATCH --error=../SlurmOutputs/random_potential_eigenvalue_statistics_job.%j.err
#SBATCH --time=4:00:00
#SBATCH -c 2
#SBATCH --mem=4GB
#SBATCH --mail-type=all
#SBATCH --chdir=../ScriptsForDataGeneration
#SBATCH --ntasks=6
#SBATCH -N 6

ml load system
ml load texlive/2019
ml load viz
ml load python/3.9.0
ml load py-numpy/1.24.2_py39
ml load py-matplotlib/3.7.1_py39
ml load py-scipy/1.10.1_py39
pip3 install --user imageio

for W in 0.1 0.5 1.0
do
 for M in 0 1
 do
  srun --exclusive -N 1 --ntasks=1 -c 2 python3.9 random_potential_eigenvalue_statistics.py --W $W --M $M &
 done
done
wait
#!/usr/bin/bash
#SBATCH --job-name=ED_vs_typicality_spin_spin_job
#SBATCH --output=../SlurmOutputs/ED_vs_typicality_spin_spin_job.%j.out
#SBATCH --error=../SlurmOutputs/ED_vs_typicality_spin_spin_job.%j.err
#SBATCH --time=0:10:00
#SBATCH -c 1
#SBATCH --mem=8GB
#SBATCH --mail-type=all
#SBATCH --chdir=../ScriptsForDataGeneration
#SBATCH --ntasks=2
#SBATCH -N 2

ml load system
ml load texlive/2019
ml load viz
ml load python/3.9.0
ml load py-numpy/1.24.2_py39
ml load py-matplotlib/3.7.1_py39
ml load py-scipy/1.10.1_py39
pip3 install --user imageio

for g in 0 0.2
do
 srun -N 1 --ntasks=1 -c 1 python3.9 compare_ED_typicality_spin_spin.py --g $g &
done
wait
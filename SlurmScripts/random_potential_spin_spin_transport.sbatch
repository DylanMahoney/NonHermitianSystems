#!/usr/bin/bash
#SBATCH --job-name=random_potential_spin_spin_transport_job
#SBATCH --output=../SlurmOutputs/random_potential_spin_spin_transport_job.%j.out
#SBATCH --error=../SlurmOutputs/random_potential_spin_spin_transport_job.%j.err
#SBATCH --time=48:00:00
#SBATCH -c 2
#SBATCH --mem=8GB
#SBATCH --mail-type=all
#SBATCH --chdir=../ScriptsForDataGeneration
#SBATCH --ntasks=45
#SBATCH -N 45


ml load system
ml load texlive/2019
ml load viz
ml load python/3.9.0
ml load py-numpy/1.24.2_py39
ml load py-matplotlib/3.7.1_py39
ml load py-scipy/1.10.1_py39
pip3 install --user imageio

#W = 0 only involves two runs and it's really quick, so this script just does nonzero W
for L in 18
do
 for seed_start in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
 do
  for W in 0.5 1 5
  do
   srun --exclusive -N 1 --ntasks=1 -c 2 python3.9 -u random_potential_spin_spin_transport.py --L $L --D1 1 --W $W --SS $seed_start &
  done
 done
done
wait
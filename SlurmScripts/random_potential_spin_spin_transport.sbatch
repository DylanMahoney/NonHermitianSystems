#!/usr/bin/bash
#SBATCH --job-name=random_potential_spin_spin_transport_job
#SBATCH --output=../SlurmOutputs/random_potential_spin_spin_transport_job.%j.out
#SBATCH --error=../SlurmOutputs/random_potential_spin_spin_transport_job.%j.err
#SBATCH --time=48:00:00
#SBATCH -c 2
#SBATCH --mem=12GB
#SBATCH --mail-type=all
#SBATCH --chdir=../ScriptsForDataGeneration
#SBATCH --ntasks=8
#SBATCH -N 8


ml load system
ml load texlive/2019
ml load viz
ml load python/3.9.0
ml load py-numpy/1.24.2_py39
ml load py-matplotlib/3.7.1_py39
ml load py-scipy/1.10.1_py39
pip3 install --user imageio

#W = 0 only involves two runs and it's really quick, so this script just does W = 0.5 and W = 1
for L in 16 18
do
 for D1 in 1 1.5
 do
  for W in 0.5 1
  do
   srun --exclusive -N 1 --ntasks=1 -c 2 python3.9 -u random_potential_spin_spin_transport.py --L $L --D1 $D1 --W $W &
  done
 done
done
wait

#In this past this has run out of memory when given only 8GB, but later it seemed fine
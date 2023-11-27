#!/usr/bin/bash
#SBATCH --job-name=HN_spin_spin_transport_job
#SBATCH --output=../SlurmOutputs/HN_spin_spin_transport_job.%j.out
#SBATCH --error=../SlurmOutputs/HN_spin_spin_transport_job.%j.err
#SBATCH --time=24:00:00
#SBATCH -c 4
#SBATCH --mem=64GB
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


for L in 24 #already did 20 and 22
do
 for g in 0 0.1 0.2
 do
  for D1 in 1.5
  do
   for D2 in 0 1.5
   do
    srun --exclusive -N 1 --ntasks=1 -c 4 python3.9 HN_spin_spin_transport.py --L $L --g $g --D1 $D1 --D2 $D2 &
   done
  done
 done
 wait
done
wait
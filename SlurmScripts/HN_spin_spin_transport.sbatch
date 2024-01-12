#!/usr/bin/bash
#SBATCH --job-name=HN_spin_spin_transport_job
#SBATCH --output=../SlurmOutputs/HN_spin_spin_transport_job.%j.out
#SBATCH --error=../SlurmOutputs/HN_spin_spin_transport_job.%j.err
#SBATCH --time=36:00:00
#SBATCH -c 4
#SBATCH --mem=80GB
#SBATCH --mail-type=all
#SBATCH --chdir=../ScriptsForDataGeneration
#SBATCH --ntasks=8
#SBATCH -N 8

#36 hours of time to account for 6 runs instead of 2
#with 6 runs and 4 g's and 2 D2's and L=24, 64GB didn't suffice
ml load system
ml load texlive/2019
ml load viz
ml load python/3.9.0
ml load py-numpy/1.24.2_py39
ml load py-matplotlib/3.7.1_py39
ml load py-scipy/1.10.1_py39
pip3 install --user imageio

#For L = 22, need 16GB of RAM and about 2.5 hours with 150 time steps
#For L = 24, need 64GB of RAM and about 10 hours with 150 time steps
for L in 24
do
 for g in 0 0.1 0.15 0.2 #experimenting with adding 0.15
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
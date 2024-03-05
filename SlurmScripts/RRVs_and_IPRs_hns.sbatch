#!/usr/bin/bash
#SBATCH --job-name=RRVs_and_IPRs_job
#SBATCH --output=../SlurmOutputs/RRVs_and_IPRs_job.%j.out
#SBATCH --error=../SlurmOutputs/RRVs_and_IPRs_job.%j.err
#SBATCH --time=16:00:00
#SBATCH -c 2
#SBATCH --mem=32GB
#SBATCH --mail-type=all
#SBATCH --chdir=../ScriptsForDataGeneration
#SBATCH --ntasks=4
#SBATCH -N 4
#SBATCH --partition=hns

ml load system
ml load texlive/2019
ml load viz
ml load python/3.9.0
ml load py-numpy/1.24.2_py39
ml load py-matplotlib/3.7.1_py39
ml load py-scipy/1.10.1_py39
pip3 install --user imageio

for L in 14
do
 for g in 0 0.2
 do
  for D1 in 1 1.5
  do
   srun -N 1 --ntasks=1 -c 2 python3.9 RRVs_and_IPRs_fresh.py --g $g --D1 $D1 --L $L &
  done
 done
done
wait
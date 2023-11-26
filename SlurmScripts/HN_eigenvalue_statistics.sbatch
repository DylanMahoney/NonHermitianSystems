#!/usr/bin/bash
#SBATCH --job-name=HN_eigenvalue_statistics_job
#SBATCH --output=../SlurmOutputs/HN_eigenvalue_statistics_job.%j.out
#SBATCH --error=../SlurmOutputs/HN_eigenvalue_statistics_job.%j.err
#SBATCH --time=16:00:00
#SBATCH -c 2
#SBATCH --mem=24GB
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

#I gave this 16 hours but it actually only took 09:16:36
#But this depended very strongly on me giving each diagonalization two CPU's

for L in 16 18 20 22
do
 for g in 0 0.1 0.2 #3 g's
 do
  for Delta_1 in 1.5 #1 Delta_2
  do
   for Delta_2 in 0 1.5 #2 Delta_2's
   do
    srun --exclusive -N 1 --ntasks=1 -c 2 python3.9 -u HN_eigenvalue_statistics.py --g $g --L $L --D1 $Delta_1 --D2 $Delta_2 &
    #Consider removing the -u after I'm confident the parallelization works
    #Given that the number of total nodes I've requested equals the number of tasks, I'm not sure whether or not to go with the --exclusive option...
   done
  done
 done
 wait #THIS WAIT MEANS THAT IT WAITS FOR ALL OF ONE L TO FINISH BEFORE STARTING THE NEXT L
done
wait
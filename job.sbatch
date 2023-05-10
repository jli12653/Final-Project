#!/bin/bash

#SBATCH --job-name=inner-prod
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4
#SBATCH --mem=2GB 
#SBATCH --time=01:00:00 
#SBATCH --verbose

module purge 
module load  amber/openmpi/intel/20.06 

mpiexec ./inner

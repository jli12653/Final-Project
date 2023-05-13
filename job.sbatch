#!/bin/bash

#SBATCH --job-name=FMM
#SBATCH --nodes=4 
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB 
#SBATCH --time=01:00:00 
#SBATCH --verbose
#SBATCH --output=FMM.out


module purge 
module load  amber/openmpi/intel/20.06 

mpiexec ./FMM

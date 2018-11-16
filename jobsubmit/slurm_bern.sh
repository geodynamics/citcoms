#!/bin/bash

# You must specify a valid email address!
#SBATCH --mail-user=daniel.bower@csh.unibe.ch

# Mail on NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --mail-type=none

# Job name
#SBATCH --job-name="CitcomS job"

# Runtime and memory
#SBATCH --time=00:10:00

## ntasks is the number of MPI tasks

## I prefer to set nodes and then tasks per node to have greater control
## over how the job is distributed.  In this case, I can run a global model
## on one physical node using 12 cores

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
##SBATCH --ntasks=12

#### Your shell commands below this line ####
module load openmpi/1.10.2-intel
srun --mpi=pmi2 /home/ubelix/csh/bower/mc/CitcomS/citcoms_assim/src/bin/CitcomSFull ./input.sample

#srun echo Hello World

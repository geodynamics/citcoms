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

#SBATCH --nodes=4
#SBATCH --ntasks-per-node=12
##SBATCH --ntasks=12

#### Your shell commands below this line ####
module load openmpi/1.10.2-intel
srun --mpi=pmi2 /home/ubelix/csh/bower/mc/CitcomS/CitcomS-3.3.1-assim/bin/CitcomSFull ./input.sample
## --solver.datadir=/home/ubelix/csh/bower/mc/CitcomS/examples/Cookbook1/data/%RANK

#srun echo Hello World

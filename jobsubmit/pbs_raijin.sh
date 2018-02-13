#!/bin/bash
#PBS -N Full
#PBS -l ncpus=12
#PBS -l mem=12GB
#PBS -l walltime=00:20:00
#PBS -j oe
#PBS -P q97
#PBS -r n
#PBS -l wd
#PBS -M nicolas.flament@sydney.edu.au
#PBS -m bae
module load openmpi/1.10.2
mpirun /home/562/nif562/cig/citcoms_assim/src/bin/CitcomSFull ./input.sample
exit
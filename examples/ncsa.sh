#!/bin/csh
#
#  Sample Batch Script for a TeraGrid cluster job
#  Uses cluster-wide GPFS scratch
#
#  Submit this script using the command: qsub
#
#  Use the "qstat" command to check the status of a job.
#
# The following are embedded QSUB options. The syntax is #PBS (the # does
# _not_  denote that the lines are commented out so do not remove).
#
# walltime : maximum wall clock time (hh:mm:ss)
#PBS -l walltime=00:30:00
#
# nodes: number of 2-processor nodes
#   ppn: how many processors per node to use (1 or 2)
#       (you are always charged for the entire node)
#PBS -l nodes=2:ppn=2
#
# export all my environment variables to the job
#PBS -V
#
# job name (default = name of script file)
#PBS -N testjob
#
# filename for standard output (default = <job_name>.o<job_id>)
# at end of job, it is in directory from which qsub was executed
# remove extra ## from the line below if you want to name your own file
###PBS -o testjob.out
#
# filename for standard error (default = <job_name>.e<job_id>)
# at end of job, it is in directory from which qsub was executed
# remove extra ## from the line below if you want to name your own file
###PBS -e testjob.err
#
# send mail when the job begins and ends (optional)
# remove extra ## and provide your email address below if you wish to 
# receive email
###PBS -m be
###PBS -M myemail@myuniv.edu
# End of embedded QSUB options

set echo               # echo commands before execution; use for debugging

# calculate NP for mpirun line 
set NP=`wc -l $PBS_NODEFILE | cut -d'/' -f1`

# store just number portion of full job id
set JOBID=`echo $PBS_JOBID | cut -d'.' -f1`

# create scratch job directory
mkdir $TG_CLUSTER_PFS/$JOBID
cd $TG_CLUSTER_PFS/$JOBID
echo "Scratch Job Directory =" `pwd`

# Run the MPI program on all nodes/processors requested by the job
mpirun  -np $NP -machinefile $PBS_NODEFILE $EXPORT_ROOT/bin/mpipython.exe $EXPORT_ROOT/modules/CitcomS/SimpleApp.pyc --steps=1 --controller.monitoringFrequency=10 --launcher.nodes=4 --solver.datafile=example1 --solver.mesher.nprocx=2 --solver.mesher.nprocy=2 --solver.mesher.nodex=17 --solver.mesher.nodey=17 --solver.mesher.nodez=9 --mode=worker
                                                                                
# end of file

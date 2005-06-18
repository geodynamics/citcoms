#!/bin/tcsh 			  
				  # first line specifies shell
#BSUB -J jobname 		  #name the job "jobname"
#BSUB -o out.o%J   		  #output->   out.o&ltjobID>
#BSUB -e err.o%J   		  #error -> error.o&ltjobID>
#BSUB -M 1048576                  #1GB/task of memory
#BSUB -n 4 -W 0:30                #4 CPUs and 30min
#BSUB -q normal                   #Use normal queue.
set echo                          #Echo all commands.
cd $LS_SUBCWD                     #cd to directory of submission

#Use "pam -g 1 gmmpirun_wrapper".
#Instead of mpirun; CPUs are
#specified above in -n option.

pam -g 1 gmmpirun_wrapper $EXPORT_ROOT/bin/mpipython.exe $EXPORT_ROOT/modules/CitcomS/SimpleApp.pyc --steps=1 --controller.monitoringFrequency=10 --launcher.nodes=4 --solver.datafile=example1 --solver.mesher.nprocx=2 --solver.mesher.nprocy=2 --solver.mesher.nodex=17 --solver.mesher.nodey=17 --solver.mesher.nodez=9 --mode=worker


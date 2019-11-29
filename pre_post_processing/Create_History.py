#!/usr/bin/env python3
#
#=====================================================================
#
#               Python Scripts for CitcomS Data Assimilation
#                  ---------------------------------
#
#                              Authors:
#                 Dan Bower, Mike Gurnis, Rakib Hassan
#             (c) California Institute of Technology 2014
#                        ALL RIGHTS RESERVED
#
#
#=====================================================================
#
#  Copyright January 2014, by the California Institute of Technology.
#
#  Modified: 14th July 2014 by DJB
#=====================================================================
'''Generate history files for slab assimilation in CitcomS using one
 processor OR the parallel infrastructure of CITerra.'''
#=====================================================================
#=====================================================================
import os, subprocess, sys, multiprocessing
import Core_Util
from Core_Util import now
from subprocess import PIPE, Popen
from threading import Thread
from make_history_for_age import basic_setup, isSerial
#=====================================================================
verbose = True
#=====================================================================
#=====================================================================
#=====================================================================
def usage():
    """print usage message and exit"""

    print( now(),'''usage: Create_History.py [-d] [-e] configuration_file.cfg

options and arguments:

-d: if the optional -d argument is geiven this script will generate 
the (required) geodynamic_framework_defaults.conf file in the current 
working directory.  This file can then be modified by the user.

-e: if the optional -e argument is given this script will print to
standard out an example configuration control file.  The parameter 
values in the example configuration_file.cfg file may need to be 
edited or commented out depending on intended use.

citation:
    Bower, D.J., M. Gurnis, and N. Flament (2015)
    Assimilating lithosphere and slab history in 4-D Earth models,
    Physics of the Earth and Planetary Interiors,
    238, 8--22, doi:10.1016/j.pepi.2014.10.013
''' )

    sys.exit(0)

#====================================================================
#====================================================================
#====================================================================
def run(args, cwd = None, shell = False, kill_tree = True, timeout = -1, env = None, sout=PIPE, eout=PIPE):
    '''
    Run a command with a timeout after which it will be forcibly
    killed.
    '''
    class Alarm(Exception):
        pass
    def alarm_handler(signum, frame):
        raise Alarm
    p = Popen(args, shell = shell, cwd = cwd, stdout = sout, stderr = eout, env = env)
    if timeout != -1:
        signal(SIGALRM, alarm_handler)
        alarm(timeout)
    try:
        stdout, stderr = p.communicate()
        if timeout != -1:
            alarm(0)
    except Alarm:
        pids = [p.pid]
        if kill_tree:
            pids.extend(get_process_children(p.pid))
        for pid in pids:
            # process might have died before getting to this line
            # so wrap to avoid OSError: no such process
            try:
                kill(pid, SIGKILL)
            except OSError:
                pass
        return -9, '', ''
    return p.returncode, stdout, stderr
#end function

#====================================================================
#====================================================================
#====================================================================
def taskThreadFunction(config_filename, tasks, icAge):
    cwd = os.getcwd()
    IC = 0;
    for age in tasks:
        if (age==icAge): IC=1
        cmd = 'mkdir %(age)s' % vars()
        if verbose: print( now(), cmd)
        subprocess.call( cmd, shell=True )

        # commands for batch file
        line = 'cp geodynamic_framework_defaults.conf %(cwd)s/%(age)s;' % vars()
        line+= 'cd %(cwd)s/%(age)s; make_history_for_age.py ' % vars()
        line+= '%(cwd)s/%(config_filename)s %(age)s ' % vars()
        line+= '%(IC)s\n' % vars()
        
        fout=sys.stdout
        ferr=sys.stderr
        _errcode, _sout, _serr = run(line, shell = True, timeout = -1, sout=fout, eout=ferr)
        if(_errcode==-9):
            print ('Error detected. Check parameters..!')
        #end if

        IC = 0
    #end for
#end function

#====================================================================
#====================================================================
#====================================================================
def main():
    '''Main sequence of script actions.'''

    print( now(), 'Create_History.py:')
    print( now(), 'main:')

    # read settings from control file
    config_filename = sys.argv[1]
    control_d = Core_Util.parse_configuration_file( config_filename )
    
    # read job settings
    control_d['serial'] = isSerial(control_d)

    age_start = max( control_d['age_start'], control_d['age_end'] )
    age_end = min( control_d['age_end'], control_d['age_start'] )
    age_loop = list( range( age_end, age_start+1 ) )
    age_loop.reverse()
    

    IC = 1
    job = control_d['job']

    # smp and serial branch
    if (job=='smp'):
        serial = control_d['serial']
        if(serial):
            for age in age_loop:
                cmd  = 'make_history_for_age.py '
                cmd += '%(config_filename)s %(age)d %(IC)s' % vars()
                if verbose: print( now(), cmd )
                subprocess.call( cmd, shell=True )
                IC = 0
            #end for
            sys.exit(0)
        else:
            cpuCount = control_d['nproc'];
            if(cpuCount==-1): cpuCount = int(multiprocessing.cpu_count());
            else: cpuCount = min(cpuCount, int(multiprocessing.cpu_count()));
            
            div, mod = divmod(len(age_loop), cpuCount)
            
            taskList = []
            for i in range(0, cpuCount): taskList.append([])

            count = 0;
            for i in range(0, cpuCount):
                taskList[i] = age_loop[count:(count+div)];
                count = count + div
            #end for

            for i in range(0, mod):
                taskList[i].append(age_loop[count])
                count = count + 1
            #end for

            for i in range(0, cpuCount):
                #print (taskList[i])
                thread = Thread(target = taskThreadFunction, \
                                args = (config_filename, taskList[i], age_start))
                thread.start()
            #end for
            
            for i in range(0, cpuCount):
                thread.join()
            #end for       
        #end if
    # parallel branch
    else:

        # total number of ages to create (inclusive)
        # and therefore number of processors to use
        # this is req for PBS submission script
        control_d['nodes'] = len(age_loop)

        batchfile = 'commands.batch'
        file = open( batchfile, 'w')

        cwd = os.getcwd()

        # make directories for files of each age
        for age in age_loop:
            cmd = 'mkdir %(age)s' % vars()
            if verbose: print( now(), cmd)
            subprocess.call( cmd, shell=True )

            if (control_d['job']=='cluster'):
                # commands for batch file
                line = 'cp geodynamic_framework_defaults.conf %(cwd)s/%(age)s; ' % vars()
                line+= 'cd %(cwd)s/%(age)s; ' % vars()
                line+= 'source ~/.bash_profile; '
                line+= 'module load python; '
                line+= 'which python; '
                line+= 'module load gmt; '
                line+= 'make_history_for_age.py ' % vars()
                line+= '%(cwd)s/%(config_filename)s %(age)s ' % vars()
                line+= '%(IC)s\n' % vars()
                file.write( line )
                IC = 0

            if (control_d['job']=='raijin'):
                # commands for batch file
                line = 'cp geodynamic_framework_defaults.conf %(cwd)s/%(age)s; ' % vars()
                line+= 'cd %(cwd)s/%(age)s; ' % vars()
                line+= 'source ~/.profile; '
                line+= 'module load python3/3.3.0; '
                line+= 'module load python3/3.3.0-matplotlib; '
                line+= 'module load gmt/4.5.11; '
                line+= 'make_history_for_age.py ' % vars()
                line+= '%(cwd)s/%(config_filename)s %(age)s ' % vars()
                line+= '%(IC)s\n' % vars()
                file.write( line )
                IC = 0

            if (control_d['job']=='baloo'):
                # commands for batch file
                line = 'cp geodynamic_framework_defaults.conf %(cwd)s/%(age)s; ' % vars()
                line+= 'cd %(cwd)s/%(age)s; ' % vars()
                line+= 'source ~/.cshrc; '
                line+= 'make_history_for_age.py ' % vars()
                line+= '%(cwd)s/%(config_filename)s %(age)s ' % vars()
                line+= '%(IC)s\n' % vars()
                file.write( line )
                IC = 0

        file.close()

        # if cluster job:
        if (control_d['job']=='cluster'):
            make_pbs_sub_script( control_d )
        # if raijin cluster job:
        if (control_d['job']=='raijin'):
            make_raijin_pbs_sub_script( control_d )
        # if baloo cluster job:
        if (control_d['job']=='baloo'):
            make_baloo_pbs_sub_script( control_d )

        # for abigail
        #make_sbatch_sub_script( control_d )

        # submit to qsub
        qsub = control_d['qsub']
        cmd = 'qsub %(qsub)s' % vars()
        if verbose: print( now(), cmd)
        subprocess.call( cmd, shell=True )

#====================================================================
#====================================================================
#====================================================================
def make_raijin_pbs_sub_script( control_d ):

    '''Write PBS submission script to a file.'''

    if verbose: print( now(), 'make_raijin_pbs_submission_script:' )

    # get variables
    jobname = control_d.get('jobname','Create_History_Parallel.py')
    nodes = control_d['nodes']
    # 12 hour default walltime if not specified
    walltime = control_d.get('walltime','12:00:00')
    # by default send job information to Nico
    email = control_d.get('email','nicolas.flament@sydney.edu.au')
    # by default 8GB memory
    mem = control_d.get('mem','8')

    text='''#!/bin/bash
#PBS -N %(jobname)s
#PBS -l ncpus=%(nodes)s
#PBS -l mem=%(mem)dGB
#PBS -P q97
#PBS -l walltime=%(walltime)s
#PBS -r y
#PBS -m bae
#PBS -M %(email)s
#PBS -l wd
#PBS -j oe

# Set up job environment:
module load python3/3.3.0
module load python3/3.3.0-matplotlib
module load gmt/4.5.11
module load parallel/20150322

#change the working directory (default is home directory)
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

# Write out some information on the job
echo Running on host `hostname`
echo Time is `date`

### Define number of processors
NPROCS=`wc -l < $PBS_NODEFILE`
echo This job has allocated $NPROCS cpus

# Tell me which nodes it is run on
echo " "
echo This jobs runs on the following processors:
echo `cat $PBS_NODEFILE`
echo " "

# 
# Run the parallel job
#

parallel -a commands.batch''' % vars()

    filename = '%(jobname)s.pbs' % vars()
    control_d['qsub'] = filename
    file = open( filename, 'w' )
    file.write( '%(text)s' % vars() )
    file.close()


def make_baloo_pbs_sub_script( control_d ):

    '''Write PBS submission script to a file.'''

    if verbose: print( now(), 'make_baloo_pbs_submission_script:' )

    # get variables
    jobname = control_d.get('jobname','Create_History_Parallel.py')
    ppn = control_d.get('ppn','16')
    nodes = int(control_d['nodes']/ppn)
    # 12 hour default walltime if not specified
    walltime = control_d.get('walltime','12:00:00')
    # by default send job information to Nico
    email = control_d.get('email','rezg@statoil.com')

    text='''#!/bin/csh -f 
#PBS -N %(jobname)s
#PBS -l nodes=%(nodes)s:ppn=%(ppn)s
#PBS -l walltime=%(walltime)s
#PBS -m bae
#PBS -M %(email)s

#change the working directory (default is home directory)
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

# Write out some information on the job
echo Running on host `hostname`
echo Time is `date`

### Define number of processors
#NPROCS=`wc -l < $PBS_NODEFILE`
#echo This job has allocated $NPROCS cpus

# Tell me which nodes it is run on
echo " "
echo This jobs runs on the following processors:
echo `cat $PBS_NODEFILE`
echo " "

# 
# Run the parallel job
#

parallel -a commands.batch''' % vars()

    filename = '%(jobname)s.pbs' % vars()
    control_d['qsub'] = filename
    file = open( filename, 'w' )
    file.write( '%(text)s' % vars() )
    file.close()

#====================================================================
#====================================================================
#====================================================================
def make_pbs_sub_script( control_d ):

    '''Write PBS submission script to a file.'''

    if verbose: print( now(), 'make_pbs_submission_script:' )

    # get variables
    jobname = control_d.get('jobname','Create_History_Parallel.py')
    nodes = control_d['nodes']
    # 12 hour default walltime if not specified
    walltime = control_d.get('walltime','12:00:00')

    text='''#PBS -N %(jobname)s
#PBS -l nodes=%(nodes)s
#PBS -S /bin/bash
#PBS -V
#PBS -l walltime=%(walltime)s
#PBS -q default
#PBS -m ae
#PBS -o out.$PBS_JOBID.$PBS_JOBNAME
#PBS -e err.$PBS_JOBID.$PBS_JOBNAME

#change the working directory (default is home directory)
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

# Write out some information on the job
echo Running on host `hostname`
echo Time is `date`

### Define number of processors
NPROCS=`wc -l < $PBS_NODEFILE`
echo This job has allocated $NPROCS cpus

# Tell me which nodes it is run on
echo " "
echo This jobs runs on the following processors:
echo `cat $PBS_NODEFILE`
echo " "

# 
# Run the parallel job
#

parallel --sshloginfile $PBS_NODEFILE  -a commands.batch''' % vars()

    filename = 'qsub.Create_History_Parallel'
    control_d['qsub'] = filename
    file = open( filename, 'w' )
    file.write( '%(text)s' % vars() )
    file.close()

#====================================================================
#====================================================================
#====================================================================
def make_sbatch_sub_script( control_d ):

    '''Write SBATCH submission script to a file.'''

    if verbose: print( now(), 'make_sbatch_sub_script:' )

    # get variables
    jobname = control_d.get('jobname','Create_History_Parallel.py')
    nodes = control_d['nodes']
    # 2 hour default walltime if not specified
    #walltime = settings.get('walltime','12:00:00')

    text = '''#!/bin/bash

# Batch file for running CitcomS on Titan at UiO
# ALB Oct/Nov 2011

# Job Details
#SBATCH --job-name=%(jobname)s
#SBATCH --account=pgp
#SBATCH --constraint=intel
# Job time and memory limits
#SBATCH --time=96:00:00 ## YOU MUST CHANGE THIS FOR LONG JOBS
#SBATCH --mem-per-cpu=2GB
#
#Parallel and mpi settings
#SBATCH --ntasks=12 ## MPI KNOWS HOW MANY NODES IT CAN USE, DON'T SPECIFY THEM
#
# Set up job environment:
source /site/bin/jobsetup
module load python/2.6.2
module load openmpi/1.4.3.intel

## Copy the CASE1 dir to the scratch, just in case
srun --ntasks=$SLURM_JOB_NUM_NODES cp -r OUTPUTFILES/ $SCRATCH

#Run program
bin/citcoms cookbook1.cfg --solver.datadir=/usit/titan/u1/abigailb/CITCOM_S/CitcomS_CIG/OUTPUTFILES/Cookbook1''' % vars()

    filename = 'qsub.Create_History_Parallel'
    settings['qsub'] = filename
    file = open(filename,'w')
    file.write('%(text)s' % vars())
    file.close()

#====================================================================
#====================================================================
#====================================================================
def make_example_config_file():

    # get current working directory
    cwd = os.getcwd()
    text='''#============
# job details
#============
# N.B., keeping with python convention you must capitalize
# True and False for Booleans!

job = smp ; options are 'smp' or 'cluster' or 'raijin' or 'baloo'.
nproc = 1 ; -1 to use all available procs.
# For 'smp', 'nproc'=1 is the serial case.
# For 'cluster' or 'raijin' or 'baloo', 'nproc' is ignored.

jobname = mkhist ; for cluster or raijin or baloo  job only
walltime = 3:00:00 ; for cluster or raijin  or baloo job only
mem = 8 ; (in GB) for raijin job only 
ppn = 16 ; for baloo job only
email = rezg@statoil.com ; for raijin  or baloo job only
age_start = 1
age_end = 0

DEBUG = False ; generic switch for debugging
VERBOSE = True ; show terminal output

# do not remove processed age and final temperature grids
KEEP_GRIDS = True
PLOT_SUMMARY_POSTSCRIPT = True; make a summary ps file for each depth
KEEP_PS = False

model_name = out ; history model name for output

#============
# data output
#============

# [INITIAL CONDITION]
# temperature initial condition
OUTPUT_TEMP_IC = False

# tracer initial condition
OUTPUT_TRAC_IC = False

# [HISTORY]
# slab temperature history
OUTPUT_TEMP = False

# internal velocity (slab descent) 
OUTPUT_IVEL = False

# thermal age of lithosphere
OUTPUT_LITH_AGE = True

#========================
# data output directories
#========================
# N.B. use an absolute path for job = cluster

grid_dir = %(cwd)s/grid ; intermediate grids
hist_dir = %(cwd)s/hist ; history
ic_dir = %(cwd)s/ic ; initial condition
ivel_dir = %(cwd)s/ivel ; ivel
lith_age_dir = %(cwd)s/age ; age
log_dir = %(cwd)s ; parameters
ps_dir = %(cwd)s/ps ; postscripts
trac_dir = %(cwd)s/trac ; tracers

#===========
# data input
#===========
# N.B. use an absolute path for job = cluster

pid_file = %(cwd)s/pid00000.cfg ; CitcomS pid file

# default coordinate file path is:
#     [datadir]/[proc]/[datafile].coord.[proc]
# or define a user-specified directory to all of the
# [datafile].coord.[proc] files:
coord_dir = %(cwd)s/coord ; CitcomS *.coord.* files

# spatial resolution of the prescribed ivel bcs
# levels = 1 ; prescribe at finest mesh
# levels = 2 ; coarsen by 2 in each dimension
# levels = 3 ; coarsen by 4 in each dimension
# for global models you'll likely need levels >= 2
# for regional models try levels = 1 and then increase
# if convergence is poor or solve time is unreasonable
levels = 2


#===================
# thermal parameters
#===================
# mantle temperature at the surface
temperature_mantle = 1.0

BUILD_LITHOSPHERE = True ; include an upper thermal boundary layer
UTBL_AGE_GRID = True ; True will use age grids
utbl_age = 300 ; if UTBL_AGE_GRID is False
lith_age_min = 0.01 ; minimum lithosphere thermal age
# note that the below only truncates oceanic regions
oceanic_lith_age_max = 300.0 ; maximum oceanic thermal age
# thermal age for non-oceanic regions if CONTINENTAL_TYPES = False
NaN_age = 200.0


BUILD_SLAB = True ; build slabs
radius_of_curvature = 200.0 ; km
# default values for slab dip and depth if GPML_HEADER = False
default_slab_dip = 45.0 ; degrees
default_slab_depth = 500.0 ; km
UM_advection = 1.0 ; non-dim factor
LM_advection = 3.0 ; non-dim factor
vertical_slab_depth = 660.0 ; depth at which to make slabs vertical

# GPML_HEADER must be True for subduction initiation
GPML_HEADER = True ; override defaults with GPML header data
slab_UM_descent_rate = 3.0 ; cm/yr
# from van der Meer et al. (2010)
slab_LM_descent_rate = 1.2 ; cm/yr

FLAT_SLAB = False ; include flat slabs

# lower thermal boundary layer
# with LTBL, the temperature of the CMB is always 1
BUILD_LTBL  = False ; lower thermal boundary layer
ltbl_age = 300.0 ; age (Ma) of tbl

# thermal blobs
BUILD_BLOB = False ; thermal blobs
blob_center_lon = 50, 130 ; degrees
blob_center_lat = 45, 45 ; degrees colat
blob_center_depth = 2867, 2867 ; km
blob_radius = 200, 400 ; km
blob_birth_age = 230, 220 ; Ma
blob_dT = 0.1, 0.1 ; non-dimensional temperature anomaly
blob_profile = constant, constant ; valid profiles (constant, exponential, gaussian1, gaussian2)

# thermal silos
BUILD_SILO = False ; thermal silos
silo_base_center_lon = 50, 130 ; degrees
silo_base_center_lat = 90, 90 ; degrees colat
silo_base_center_depth = 2867, 2867 ; km
silo_radius = 200, 400 ; km
silo_cylinder_height= 500, 500 ; km
silo_birth_age = 230, 220 ; Ma
silo_dT = 0.1, 0.1 ; non-dimensional temperature anomaly
silo_profile = constant, constant ; valid profiles (constant, exponential, gaussian1, gaussian2)

# ADIABAT only for extended-Boussinesq or compressible models
BUILD_ADIABAT = False ; linear temp increase across mantle
# non-dimensional adiabatic temperature drop scaled with respect
# to the total temperature drop across the model
adiabat_temp_drop = 0.3


#===========
# Continents
#===========
# Build upper thermal boundary layer with continents
# Also continental tracers (if OUTPUT_TRAC_IC is also True)
CONTINENTAL_TYPES = True

# For continental types, list stencil values with no spaces delimited 
# by a comma. Use negative integers.
# Then, for each stencil give the age for reassignment
# In the example below, -1 is Archean, -2 Proterozoic, -3 Phanerozoic,
# and -4 COB.
# ensure stencil_values are small negative integers

stencil_values = -4,-3,-2,-1
stencil_ages = 104,103,153,369

# No assimilation in areas that have been deformed
NO_ASSIM = True
no_ass_age = -1000
no_ass_padding = 100

# Exclude tracers in areas that have been deformed
# to a depth defined by 'tracer_no_ass_depth'
# also uses no_ass_age and no_ass_padding
# and CONTINENTAL_TYPES must be True
# also, no_ass_age must be more negative than the
# most negative stencil value
TRACER_NO_ASSIM = False
tracer_no_ass_depth = 350 ; km

# Build tracer field with continents using 'stencil_values'
# tracer flavors and depths

# for positive thermal ages (i.e., oceanic)
# note: must be '0' suffix
flavor_stencil_value_0 = 0
depth_stencil_value_0 = 410

# for stencil value -1
flavor_stencil_value_1 = 1,2
depth_stencil_value_1 = 40,250

# for stencil value -2
flavor_stencil_value_2 = 1,3
depth_stencil_value_2 = 40,160

# for stencil value -3
flavor_stencil_value_3 = 1,4
depth_stencil_value_3 = 40,130

# for stencil value -4
flavor_stencil_value_4 = 1,4
depth_stencil_value_4 = 40,130

# etc. for more stencil values, e.g.,
# flavor_stencil_value_5 = 0 
# depth_stencil_value_5 = 410

#=========================
# tracer initial condition
#=========================
# this is approximate, because tracers are uniformly distributed
tracers_per_element = 30

# set region around slabs to ambient flavor (0)
SLAB_STENCIL = True
# stencil width: 300 km is consistent with the default width of the thermal stencil
# wide stencils limit crustal thickening along convergent margins
# narrow stencils avoid a gap along convergent margins but may result in significant
# crustal thickening and unrealistic elevations along convergent margins
slab_stencil_width = 300 ; km - suggested range: 100-300 km

# uniform dense layer at base of mantle
DEEP_LAYER_TRACERS = False
deep_layer_thickness = 300 ; km - 113 km gives 2 per cent of Earth's volume.
# flavor should not be 0 (0 is always ambient flavor)
deep_layer_flavor = 5

# eliminate tracers between these bounds
# this saves memory when using the hybrid method to compute composition
NO_TRACER_REGION = True
no_tracer_min_depth = 410 ; km
no_tracer_max_depth = 2604 ; km

#==============================
# synthetic regional model only
#==============================
SYNTHETIC = False ; master switch for synthetic regional models
OUTPUT_BVEL = True ; output velocity boundary conditions
bvel_dir = %(cwd)s/bvel ; bvel
fi_trench = 0.8 ; radians
TRENCH_CURVING = False
curving_trench_lat = 0.0 ; degrees
subduction_zone_age = 100 ; Ma
plate_velocity = 5 ; cm/yr
plate_velocity_theta = -1 ; direction of velocity (non-dim)
plate_velocity_phi = 1 ; direction of velocity (non-dim)
velocity_smooth = 200 ; bvel smoothing (Gaussian filter)
no_of_edge_nodes_to_zero = 3 ; no of edge nodes to smooth across (x and y)
overriding_age = 50 ; Ma
rollback_start_age = 100 ; Ma
rollback_cm_yr = 0 ; cm/yr (direction is always -phi)
''' % vars()

    print( text )

#====================================================================
#====================================================================
#====================================================================

if __name__ == "__main__":

    # check for script called wih no arguments
    if len(sys.argv) < 2:
        usage()
        sys.exit(-1)

    # create example config file 
    if '-e' in sys.argv:
        make_example_config_file()
        sys.exit(0)

    # create example geodynamic_framework_defaults.conf
    if '-d' in sys.argv:
        Core_Util.parse_geodynamic_framework_defaults()
        sys.exit(0)

    # run the main script workflow
    main()
    sys.exit(0)

#====================================================================
#====================================================================
#====================================================================

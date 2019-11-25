#!/usr/bin/env python
#=====================================================================
#                Geodynamic Framework Scripts for 
#         Preprocessing, Data Assimilation, and Postprocessing
#
#                 AUTHORS: Mark Turner, Dan J. Bower
#                  ---------------------------------
#             (c) California Institute of Technology 2014
#                  ---------------------------------
#                        ALL RIGHTS RESERVED
#=====================================================================
#=====================================================================
#=====================================================================
import subprocess, sys
import Core_Citcom, Core_Util
from Core_Util import now
#=====================================================================
#=====================================================================
#=====================================================================
def usage():
    '''print usage message, and exit'''

    # Mark to update

    sys.exit()

#=====================================================================
#=====================================================================
def main():
    print( now(), 'copy_citcom_model_from_cluster.py')

    # Mark - these should probably be user inputs
    # You could also allow the user to specify the usual types of
    # time strings like we have for grid_maker.py  Therefore, the user
    # could use timesteps, run times, or ages in the various comma-sep
    # lists or start/end/range formats

    # for testing I was running this script on citerra in this directory:
    # /home/danb/lat/lat01

    field_list = ['velo','visc'] # list to loop over
    time_list = ['0','290'] # list to loop over
    # local processing directory that can be 'seen' from the cluster
    # e.g., I can see this from citerra and is just a test location
    rootdir = '/home/danb/beno/test_copy/model'

    pid_file = 'pid14289.cfg'

    # pid_file should also be an input argument
    # I'm assuming the script will always be run in the directory of
    # the CitcomS model on the cluster where the data was generated

    # parsing the pid file is helpful because it gives us the datafile
    # etc.
    master_d = Core_Citcom.get_all_pid_data( pid_file )
    pid_d = master_d['pid_d']

    # make data directory and determine structure
    datafile = pid_d['datafile']
    datadir = pid_d['datadir']

    if datadir.endswith( '%RANK' ):
        print( 'data stored by processor' )
        datadir = datadir[:-5] # strip '%RANK'
        print(datadir)
        PROC = True
    else:
        PROC = False # not sure if this will be necessary, but
        # easy to include in this development draft

    # copy top level files
    cmd = 'cp %(pid_file)s %(rootdir)s' % vars()
    subprocess.call( cmd, shell=True)
    cmd = 'cp stderr.txt %(rootdir)s/stderr.txt' % vars()
    subprocess.call( cmd, shell=True)
    cmd = 'cp stdout.txt %(rootdir)s/stdout.txt' % vars()
    subprocess.call( cmd, shell=True)
    # copy user-created coordinate file if it exists
    coor_file = pid_d['coor_file']
    cmd = 'cp %(coor_file)s %(rootdir)s/%(coor_file)s' % vars()
    subprocess.call( cmd, shell=True)
    cmd = 'cp %(datafile)s.cfg %(rootdir)s/%(datafile)s.cfg' % vars()
    subprocess.call( cmd, shell=True)

    datadir_abs = rootdir + '/'+ datadir

    # make the root (if doesn't exist) and data directory
    Core_Util.make_dir( datadir_abs )

    # copy data
    if PROC:
        for proc in range( pid_d['total_proc'] ):
            datadir_proc = datadir_abs + str(proc) + '/'
            Core_Util.make_dir( datadir_proc )
            for field in field_list:
                # always need coordinate file
                coord_name = str(proc) + '/' + datafile + '.coord.' + str(proc)
                filename1 = datadir + coord_name
                filename2 = datadir_abs + coord_name
                cmd = 'cp %(filename1)s %(filename2)s' % vars()
                print(cmd)
                # Mark - this command actually calls the copy command
                subprocess.call( cmd, shell=True )
                for time in time_list:
                    # create filename
                    file_name = str(proc) + '/' + datafile + '.' + field + '.'
                    file_name += str(proc) + '.' + str(time)
                    filename1 = datadir + file_name
                    filename2 = datadir_abs + file_name
                    cmd = 'cp %(filename1)s %(filename2)s' % vars()
                    print(cmd)
                    #subprocess.call( cmd, shell=True )

        # now copy essential files from 0/ directory
        zero_proc_dir = datadir_abs + '0/' + datafile
        for suffix in ['.time', '.log']:
            file_name = '0/' + datafile + suffix
            filename1 = datadir + file_name
            filename2 = datadir_abs + file_name
            cmd = 'cp %(filename1)s %(filename2)s' % vars()
            print(cmd)
            subprocess.call( cmd, shell=True )

    else:

        # non-processor (%RANK) branch
        # all files are stored in data
        # although we could code this up here, I think having
        # all the files in one directory will break grid_maker.py
        # at the moment.
        pass
    

#=====================================================================
#===================================================================== 
if __name__ == "__main__":

    main()
    sys.exit(0)


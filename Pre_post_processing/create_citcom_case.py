#!/usr/bin/env python
#=====================================================================
#                Geodynamic Framework Scripts for 
#         Pre-processing, Data Assimilation, and Postprocessing
#
#                 AUTHORS: Mark Turner, Dan J. Bower
#                  ---------------------------------
#             (c) California Institute of Technology 2015
#                  ---------------------------------
#                        ALL RIGHTS RESERVED
#=====================================================================
#=====================================================================
# create_citcom_case.py
#=====================================================================
# This script creates the standard directory structure for a Citcom case.
#
# Please see the usage() function below for more information.
#=====================================================================
#=====================================================================
import sys, string, os, subprocess
import numpy as np
#=====================================================================
import Core_Citcom
import Core_GMT
import Core_Util
from Core_Util import now

Core_Util.verbose = True

#=====================================================================
#=====================================================================
def usage():
    '''print usage message, and exit'''

    print('''usage: create_citcom_case.py case_name [-r int] 

This script creates the standard GeoDynamic Framework directory and file structure, 
compatible with the GDF pre- and post- processing scripts.

The GDF uses the 'geodynamic_framework_defaults.conf' file to determine many 
locations for fundamental files used in processing data for a Citcom case.

The system-wide GDF .conf file is contained in the GDF distribution,
and becomes the default for each new case created with this script.

This script creates a single copy of the GDF configuration file 
in the top level directory of the new case.

This script makes a copy of that file with no changes.
The user is expected to adjust the new local case directory copy as needed.

This script will also make soft links from the newly created case top level directory .conf file 
to various sub-directories of the case - see below for details.

Options and arguments: 

case_name is the top level name for the citcoms case.

[ -r create a number of run directories intended to share the same kinematic plate reconstruction 
(i.e. the same GPlates global model).]

''')
    sys.exit()
#=====================================================================
#=====================================================================
def main():
    '''main workflow of the script'''

    # report the start time and the name of the script
    print( now(), 'create_citcom_case.py')

    # get the case name from user cmd line 
    case_name = str( sys.argv[1] )
    print( now(), 'Creating GDF directory structure for case:', case_name)

    # create the top level case dir
    Core_Util.make_dir( case_name )

    # set some case level file names 
    case_gdf_conf = case_name + '/' + Core_Util.gdf_conf
    pdir_gdf_conf = '..'      + '/' + Core_Util.gdf_conf

    # copy the system file to the main case directory 
    if not os.path.exists( case_gdf_conf ): 
        cmd = 'cp ' + Core_Util.sys_gdf_conf + ' ' + case_gdf_conf
        print(now(), cmd)
        subprocess.call( cmd, shell=True )
    else:
        print(now(), 'Local GDF .conf file found; NOT copying system .conf file')

    # Make sub dirs for case-based Reconstruction/ kinematic and surface data 
    Core_Util.make_dir( case_name + '/Reconstruction' )

    # Create specific sub-dirs for pre- and post- processing
    in_list = ['Coord', 'ICHist', 'Tracers', 'Topologies', 'Velocity'] 
    for d in in_list:
        Core_Util.make_dir( case_name + '/Reconstruction/' + d )

    # NOTE: A few similar Reconstruction/ type system-wide input directories 
    # are directly referenced by specific entries in the GFD .conf file.
    # (types of age grids, coastlines, velocity, etc.)
    # 
    # Many GDF pre- and post- scripts use the current working directory 
    # copy (or link) of the .conf file to control processing steps 
    # and locate base file paths.
    # 
    # Be sure to sychronize your .conf for case- and run- level work.

    # Check cmd line args to create multiple runs 
    n_runs = 1
    if '-r' in sys.argv:
       n_runs = int( sys.argv[ sys.argv.index('-r') + 1] )

    # Create specific run directories
    for i in list(range( n_runs )):
        # make a string and pad with zeros 
        if   n_runs <  10 : d = '%01d'
        elif n_runs < 100 : d = '%02d'
        else              : d = '%04d'
        r = d % i
        # make the dir
        Core_Util.make_dir( case_name + '/Run-' + r )
        # link the case-level .conf file
        Core_Util.make_link( pdir_gdf_conf , case_name + '/Run-' + r + '/' + Core_Util.gdf_conf )
#=====================================================================
#=====================================================================
if __name__ == '__main__':

    # check for script called with no arguments
    if len(sys.argv) < 2:
        usage()
        sys.exit(-1)

    # run the main script workflow
    main()
    sys.exit(0)
#=====================================================================
#=====================================================================

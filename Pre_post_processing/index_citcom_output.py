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
# index_citcom.py
#=====================================================================
# This script is a general purpose tool to process Citcoms output data
# into one ore more GMT style .grd format data files.  
# Please see the usage() function below, and the sample configuration 
# file: /sample_data/indexer_citcom.cfg for more info.
#=====================================================================
#=====================================================================
import sys, string, os
import numpy as np
#=====================================================================
import Core_Citcom
import Core_GMT
import Core_Util
from Core_Util import now

from scipy.io import netcdf

import netCDF4
from netCDF4 import Dataset


# FIXME: turn these False when stable
#Core_Util.verbose = True 
#Core_Citcom.verbose = True 
#Core_GMT.verbose = True

#=====================================================================
#=====================================================================
def usage():
    '''print usage message, and exit'''

    print('''usage: citcoms_indexer.py [-e] configuration_file.cfg

Options and arguments:
  
-e : if the optional -e argument is given this script will print to standard out an example configuration control file.
   The parameter values in the example config.cfg file may need to be edited or commented out depending on intended use.

'configuration_file.cfg' : is a geodynamic framework formatted control file, with at least these entries: 

    pid_file = /path/to/a/citcoms_pid_file # the path of a citcoms pid0000.cfg file 
    time_spec = multi-value specification (single value, comma delimted list, start/stop/step trio),
    level_spec = multi-value specification (single value, comma delimted list, start/stop/step trio),

and at least one sub-section block:

    [Subsection], where 'Subsection' may be any string, followed by:
    field = standard Citcom field name ('temp', 'visc', 'comp', etc. - see Core_Citcom.py for more info)
      
See the example config.cfg file for more info.
''')
    sys.exit()
#=====================================================================
#=====================================================================
def main():
    print( now(), 'index_citcom.py')

    # get the .cfg file as a dictionary
    control_d = Core_Util.parse_configuration_file( sys.argv[1] )
    #Core_Util.tree_print( control_d )

    # set the pid file 
    pid_file = control_d['pid_file']

    # get the master dictionary and define aliases
    master_d = Core_Citcom.get_all_pid_data( pid_file )
    coor_d = master_d['coor_d']
    pid_d = master_d['pid_d']

    # Double check for essential data 
    if master_d['time_d'] == None : 
        print( now() )
        print('ERROR: Required file "[CASE_NAME].time:" is missing from this model run.')
        print('       Aborting processing.')
        sys.exit(-1)

    # set up working variables
    datadir       = pid_d['datadir']
    datafile      = pid_d['datafile']
    startage      = pid_d['start_age']
    output_format = pid_d['output_format']

    depth_list   = coor_d['depth_km']
    nodez        = pid_d['nodez']
    nproc_surf   = pid_d['nproc_surf']


    found_depth_list = []

    # Check how to read and parse the time spec:
    read_time_d = True

    # Compute the timesteps to process
    if read_time_d : 
        time_spec_d = Core_Citcom.get_time_spec_dictionary(control_d['time_spec'], master_d['time_d'])
    else :
        time_spec_d = Core_Citcom.get_time_spec_dictionary(control_d['time_spec'])
    print ( now(), 'index_citcom.py: time_spec_d = ')
    Core_Util.tree_print( time_spec_d )

    # levels to process 
    level_spec_d = Core_Util.get_spec_dictionary( control_d['level_spec'] )
    print ( now(), 'index_citcom.py: level_spec_d = ')
    Core_Util.tree_print( level_spec_d )

    #
    # Main looping, first over times, then sections, then levels
    # 

    print(now(), '=========================================================================')
    print(now(), 'index_citcom.py: Main looping, first over times, then sections, then levels')
    print(now(), '=========================================================================')

    # Loop over times
    for T, time in enumerate( time_spec_d['time_list'] ) :
        #print( now(), 'index_citcom.py: Processing time = ', time) 

        if 'Ma' in time:
            # strip off units and make a number
            time = float( time.replace('Ma', '') )

            # determine what time steps are available for this age 
            # NOTE: 'temp' is requried to set which output files to check 
            found_d = Core_Citcom.find_available_timestep_from_age( master_d, 'temp', time )

        else:
            # model time steps
            time = float( time ) 
             
            # determine what time steps are available for this timestep 
            # NOTE: 'temp' is requried to set which output files to check 
            found_d = Core_Citcom.find_available_timestep_from_timestep( master_d, 'temp', time )

        # end of check on time format 

        # set variables for subsequent loops
        timestep = found_d['found_timestep']
        runtime_Myr = found_d['found_runtime']
        # convert the found age to an int
        age_Ma = int(np.around( found_d['found_age'] ) )

        print( now(), 'index_citcom.py: time data: requested value ->found value ')
        print( now(), '  ', \
'age =', found_d['request_age'],      '->', age_Ma, \
'step =', found_d['request_timestep'], '->', timestep, \
'r_tm =', found_d['request_runtime'],  '->', runtime_Myr )

        # empty file_data
        file_data = []

        # Loop over sections (fields) 
        for S, s in enumerate (control_d['_SECTIONS_'] ) :

                # FIXME: this extra indent is probably from when sections loop was inside level loop ? 

                #print( now(), 'index_citcom.py: Processing section = ', s) 

                # check for required parameter 'field'
                if not 'field' in control_d[s] :
                   print('ERROR: Required parameter "field" missing from section.')
                   print('       Skipping this section.')
                   continue # to next section
 
                # get the field name 
                field_name = control_d[s]['field']

                #print('')
                #print( now(), 'index_citcom.py: Processing: field =', field_name) 

                # set the region
                #if nproc_surf == 12:
                #    grid_R = 'g'
                #    # optionally adjust the lon bounds of the grid to -180/180
                #    if 'shift_lon' in control_d :
                #        print( now(), 'index_citcom.py: grid_R set to to "d" : -180/+180/-90/90')
                #        grid_R = 'd'
                #    else :
                #        print( now(), 'index_citcom.py: grid_R set to to "g" : 0/360/-90/90')
                #else:
                #    grid_R  = str(pid_d['lon_min']) + '/' + str(pid_d['lon_max']) + '/'
                #    grid_R += str(pid_d['lat_min']) + '/' + str(pid_d['lat_max'])
  
                # get the data file name specifics for this field 
                file_name_component = Core_Citcom.field_to_file_map[field_name]['file']

                # get the data file column name specifics for this field 
                field_column = Core_Citcom.field_to_file_map[field_name]['column']

                # report
                #print( now(), 'index_citcom.py: field = ', field_name, '; file_comp =', file_name_component, '; col =', field_column)
                # process data from Citcoms 
                file_format = ''
   
                # check for various data dirs:
                if os.path.exists( datadir + '/0/') :
                    file_format = datadir + '/#/' + datafile + '.' + file_name_component + '.#.' + str(timestep)

                elif os.path.exists( datadir + '/' ) :
                    file_format = datadir + '/' + datafile + '.' + file_name_component + '.#.' + str(timestep)

                elif os.path.exists('data') :
                    file_patt = './data/#/' + datafile + '.' + file_name_component + '.#.' + str(timestep)

                elif os.path.exists('Data') :
                    file_patt = './Data/#/' + datafile + '.' + file_name_component + '.#.' + str(timestep)

                # report error 
                else :
                    print( now() )
                    print('ERROR: Cannot find output data.')
                    print('       Skipping this section.')
                    print( now(), 'index_citcom.py: file_format = ', file_format)
                    continue # to next section

                print( now(), 'index_citcom.py: file_format = ', file_format )

                #
                # Loop over levels 
                #
                for L, level in enumerate( level_spec_d['list'] ) :

                #    print( now(), 'index_citcom.py: Processing level = ', level) 

                    # ensure level is an int value 
                    level = int(level)
                    depth = int(depth_list[level])
                    found_depth_list.append(depth)

                    #print( now(), '------------------------------------------------------------------------------')
                    print( now(), 'index_citcom.py: ', s, \
': ts =', timestep, \
'; age =', age_Ma, \
#'; runtime_Myr =', runtime_Myr, \
'; level =', level, \
'; depth_km =', depth, \
'; field =', field_name,\
)
                    #print( now(), '------------------------------------------------------------------------------')

                    # FIXME: is it ok to chanage the default name to have age, rather than timestep? 
                    xyz_filename = datafile + '-' + field_name + '-' + str(age_Ma) + 'Ma-' + str(depth) + '.xyz'
                    #print( now(), 'index_citcom.py: xyz_filename =', xyz_filename)
            
                    #xy_filename = ''
                    #xy_path = master_d['geoframe_d']['gplates_line_dir']
                    #xy_filename = xy_path + '/' + 'topology_platepolygons_' + age + '.00Ma.xy' 
                    #print( now(), 'index_citcom.py: xy_filename = ', xy_filename)
    
                    # Make a plot of the grids

                    # citcoms 
                
                # end of loop over levels 

            # end of loop over sections

    # end of loop over times

    print( now(), 'depth_list = ', depth_list)
    print( now(), 'found_depth_list = ', found_depth_list)

#=====================================================================
#=====================================================================
# SAVE This code for reference:
#                    # optionally adjust the lon bounds of the grid to -180/180
#                    #if 'shift_lon' in control_d : 
#                    #    print( now(), 'index_citcom.py: shifting values to -180/+180')
#                    #    arg = grid_filename
#                    #    opts = {'R' : 'd', 'S' : '' }
#                    #    Core_GMT.callgmt('grdedit', arg, opts)
#=====================================================================
#=====================================================================
def make_example_config_file( ):
    '''print to standard out an example configuration file for this script'''

    text = '''#=====================================================================
# config.cfg file for the index_citcom.py script
# 
# This example has information on creating grids for both Citcom data,
# and GPlates data.  Each type of data source has differnt parameter values,
# and only one source should be use in a single .cfg file.
# In each section the source choices are separated by '# # # # #' lines.
# See each section below for details.
#
# ==============================================================================
# Set the basic model coordinate information common to both source types:

# set path to the model pid file 
#pid_file = pid6750.cfg ; global case with surf data 
pid_file = pid32395.cfg ; global cookbook case 

# CitcomS coordinate files by processor (i.e. [datafile].coord.[proc])
# first, look in this user-specified directory for all files
coord_dir = coord

# second, look in data/%RANK

# NOTE: index_citcom.py will fail if coord files cannot be located

# Optional global settings

# If 'shift_lon' is set to True, then the grids will have data in the -180/+180 longitude range
# The default is for data in the 0/360 longitude range.
# shift_lon = True

# If 'make_plate_frame_grid' is set to True, then this script with produce additional data on the plate frame
#make_plate_frame_grid = True

# ==============================================================================
# Set the times to grid 

# Citcoms : use model time steps or reconstruction age, in one of these forms:

# single value:
#time_spec = 4500
#time_spec = 7Ma

# comma separated list:
#time_spec = 5400, 6200, 6961
#time_spec = 0Ma, 50Ma, 100Ma

# range of values: start/stop/step
# time_spec = 2000/2700/100
time_spec = 0Ma/200Ma/10Ma

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# GPlates : use values in Ma, and include the Ma suffix
# (note: 'Ma' will be stripped off to compute int values)

# single value:
#time_spec = 0Ma

# comma separated list:
#time_spec = 0,100,200Ma

# range of values: start/stop/step
#time_spec = 0/230/1Ma


# ==============================================================================
# Set the levels to grid 

# Citcoms : use int values from 0 to nodez-1, in one of these forms:

# single value:
#level_spec = 63

# comma separated list:
#level_spec = 64/0/10

# range of values: start/stop/step
#level_spec = 64/0/10

# NOTE: The level_spec settings must match the Citcoms field data types:

# Volume data fields : the level_spec may be values from 0 to nodez-1
level_spec = 63

# Surface (surf) data fields : the level_spec must be set to only nodez-1 
#level_spec = 64 ; for surf data 

# Botttom (botm) data fields : lthe evel_spec must be set to only 0 
#level_spec = 0 ; for bot data 

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# GPlates : use only a single value of 0 to signify the surface data
# NOTE: citcoms data will have many levels, gplates only has surface data 
# but a valid level value is requred for both data sources because 
# the level value will become part of the file names for resultant .xyz and .grd files

#level_spec = 0 

# ==============================================================================
# Set the fields to grid 
#
# Each field will be a separate section, delimited by brackets [Section_1], 
# each field requires the field name, e.g. 
# field = temp
# Each field may set optional arguments to set GMT parameters.
# Each field may set the optional parameter 'dimensional' to 'True',
# to produce an additional dimensionalized grid with the '.dimensional' filename component.
#
# See Core_Citcoms.field_to_file_map data struct for a list of field names.
#

# Citcoms :

[Grid_1]
field = temp
dimensional = True
#blockmedian_I = 0.5
#surface_I = 0.25
#Ll = 
#Lu = 
#T = 

#[Grid_2]
#field = surf_topography
#dimensional = True
#blockmedian_I = 0.5
#surface_I = 0.25
#Ll = 
#Lu =
#T = 
 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# GPlates : 

#[GP_VMAG]
#field = gplates_vmag
#blockmedian_I = 0.5
#surface_I = 0.25
#
#[GP_VY]
#field = gplates_vy
#blockmedian_I = 0.5
#surface_I = 0.25
#
#[GP_VX]
#field = gplates_vx
#blockmedian_I = 0.5
#surface_I = 0.25
#
#=====================================================================
'''
    print( text )
#=====================================================================
#=====================================================================
if __name__ == "__main__":

    #print ( str(sys.version_info) ) 

    # check for script called wih no arguments
    if len(sys.argv) != 2:
        usage()
        sys.exit(-1)

    # create example config file 
    if sys.argv[1] == '-e':
        make_example_config_file()
        sys.exit(0)

    # run the main script workflow
    main()
    sys.exit(0)
#=====================================================================
#=====================================================================

#!/usr/bin/env python
#=====================================================================
#                Geodynamic Framework Scripts for 
#         Preprocessing, Data Assimilation, and Postprocessing
#
#                 AUTHORS: Mark Turner, Dan J. Bower
#                  ---------------------------------
#             (c) California Institute of Technology 2013
#                  ---------------------------------
#                        ALL RIGHTS RESERVED
#=====================================================================
#=====================================================================
# This script is a general purpose tool to process Citcoms output data
# into one ore more GMT style .ps image files.
#
# Please see the usage() and make_example_config_file() 
# functions below for more info.
#
# This script also serves as an introduction to how to use the various
# functions in the Core_* geodyanmic framework modules, and the typical
# coding conventions and style guide used throughout the project.
#=====================================================================
#=====================================================================

# Import Python and standard modules 
import sys, string, os
import numpy as np

# Import geo dynamic framework Core modules
import Core_Citcom
import Core_GMT
import Core_Util

# The now() function is uses to print the current wall clock time,
# and can be usefull to measure runtimes of scripts.
from Core_Util import now

# Each Core modeule has a verbose option that will enable diagnostic 
# output to standard out.  Turn these True and False as needed.
Core_Citcom.verbose = False 
Core_Util.verbose = False
Core_GMT.verbose = True

#=====================================================================
#=====================================================================
# Global Variables
# some scripts will require global variables; place them at the top.

# Control how much diagnostic output is sent to the user
verbose = True

#=====================================================================
#=====================================================================
def usage():
    '''print usage message, and exit'''

    print('''simple_map.py [-e] configuration_file.cfg

Options and arguments:
  
-e : if the optional -e argument is given this script will print to standard out an example configuration control file.
    The parameter values in the example config.cfg file may need to be edited or commented out depending on intended use.
 
'configuration_file.cfg' : is a geodynamic framework formatted control file, with at least these entries: 
      
        pid_file = /path/to/a/citcoms_pid_file.cfg

    and at least one sub section block:
        
        [Subsection], where 'Subsection' may be any string, folllowed by lines 
        with these entiries:

        time = integer value of model time steps 
        level = citcoms radial node number (z level) 
        field = a valid citcoms field name ('temp', 'visc', etc.)
''')

    sys.exit()
#=====================================================================
#=====================================================================
def main():
    '''This is the main function of simple_map.py

    main performs several steps detailed below:
    
    Parse the configuration control file into a control dictionary (control_d).  

    Read the citcoms pid file and establish a master dictionary 
    of citcoms parameters (master_d)

    Getting Surfce Coordinate Data (lat and lon)

    
    # Loop over each subsection in the control_d dictionary 
'''
    # associate the script global variable with this function
    global verbose

    # In general each script and each function should report it's name
    # as the first step, to better debug workflows.
    # 
    # Most diagnostic output to the user will include the now() function
    # to easily measure wall clock runtimes of processes.
    print( now(), 'simple_map.py')

    # Parse cmd line input for basic parameters of plot
    control_d = Core_Util.parse_configuration_file( sys.argv[1] )

    # under verbose diagnostics, print out the control dictionary 
    if verbose: 
        print( now(), 'main: control_d =')
        # Core_Util.tree_print() gives nice formatted printing for dictionaries and lists
        Core_Util.tree_print( control_d )
   

    # Set the pid file as a variable name.
    pid_file = control_d['pid_file']

    # Get the master dictionary and define aliases 
    master_d = Core_Citcom.get_all_pid_data( pid_file )

    # Now master_d has all the information realted to a citcom run, and the geoframe defaults.

    # We could print out all the citcom run data file info with this code:
    #
    #if verbose: 
    #    print( now(), 'main: master_d =')
    #    Core_Util.tree_print( master_d )
    # 
    # but this usually gives tens of thousands of lines of data,
    # showing all the time values, all the level coordinate values, etc.

    # We can define aliases for the most commonly used sub dictionaries:
    geo_d = master_d['geoframe_d'] # holds all the geoframe default settings and paths 
    pid_d = master_d['pid_d']      # holds all the .pid file info 
    coor_d = master_d['coor_d']    # holds all the coordinate info 
    time_d = master_d['time_d']    # holds all the time info 

    # Under verbose mode it's good to show the basic defaults, for reference in the script log
    if verbose: 
       print( now(), 'main: geo_d =')
       Core_Util.tree_print( geo_d )

    # We also want to establish variables some commonly used data about the citcom run:
    datafile = pid_d['datafile']
    nodez = pid_d['nodez']
    nproc_surf = pid_d['nproc_surf']

    # Now we are ready to set up the basic info common to all subsection maps:

    # Getting Surfce Coordinate Data (lat and lon)
    # 
    # Because the surface coordinates are depth-independant, we get this information first,
    # before any looping over sections.

    # First, check if an optional 'coord_dir' entry was in the control file, 
    # and then check for CitcomS *.coord.* files in that user-specified coord_dir.  
    # In this case you should manually copy all the processor *.coord.* files to a single directory.
    try:
        if 'coord_dir' in control_d: 
          coord_file_format = control_d['coord_dir'] + '/%(datafile)s.coord.#' % vars()
          coord = Core_Citcom.read_citcom_surface_coor( master_d['pid_d'], coord_file_format )
    except FileNotFoundError:
       print( now(), 'WARNING: *.coord.* files not found in:', control_d['coord_dir'] )
    # Second, check for CitcomS *.coord.* files in data/%RANK dirs
    try:
        coord_file_format = 'data/#/' + datafile + '.coord.#'
        coord = Core_Citcom.read_citcom_surface_coor( master_d['pid_d'], coord_file_format )
    # If the coordinate files are missing we cannot continue:
    except FileNotFoundError:
        print( coord_file_format )
        print( now(), 'ERROR: cannot find coordinate files in',
            control_d['coord_dir'], 'or data/%RANK' )

    # Now flatten the coordinate data since we don't care about specific cap numbers for a given depth
    coord = Core_Util.flatten_nested_structure( coord )

    # extract lon and lat data as lists from tuples 
    lon = [line[0] for line in coord]
    lat = [line[1] for line in coord]


    # Now that we have the coordinate data for all levels and all times, 
    # we can process each sub section of the control file.
    #
    # The control_d dictionary has a special top level entry with key of '_SECTIONS_'.
    # The value is a list of all the subsection names.  We can iterate this way:
    # Loop over each subsection in the control_d dictionary 
    for section_name in control_d['_SECTIONS_'] : 

        print( now() )
        print( 'Processing subsection:', section_name)

        # get the subsection dictionary 
        section_d = control_d[section_name]

        # Get the specific time and level field data to map:
        time  = section_d['time'] 
        level = section_d['level'] 
        field_name = section_d['field']

        # We can use time_d to get equivalent times:
        time_triple = Core_Citcom.get_time_triple_from_timestep( time_d['triples'], float(time) )
        age     = time_triple[1] # get the equivalent reconstuction age in Ma
        runtime = time_triple[2] # get the equivalent model runtime in Myr
        print( 'time    =', time, 'steps')
        print( 'age     =', age , 'Ma')
        print( 'runtime =', runtime , 'Myr')

        # We can use the coor_d to find equivalent values for the level to map:
        radius = coor_d['radius'][level] # The non-dimentional value of the radius for this level
        radius_km = coor_d['radius_km'][level] # the equivalent radius in km
        depth = coor_d['depth'][level] # The non-dimentional value of the depth for this level
        depth_km = coor_d['depth_km'][level] # the equivalent depth in km
        print( 'level =', level)
        print( 'non-dim radius =', radius, '; radius in km =', radius_km)
        print( 'non-dim depth =', depth, '; depth in km =', depth_km)

        # 
        # Now we will extract data for this specific time, level and field:
        #

        # Core_Citcom module has the standard mapping from field_name to the specific info
        # for file name component, and column number, for each field.

        # get the file name component for this field
        file_name_component = Core_Citcom.field_to_file_map[field_name]['file']

        # get that column number for this field
        field_column = Core_Citcom.field_to_file_map[field_name]['column']

        # Create the total filename to read 
        file_format = 'data/#/' + datafile + '.' + file_name_component + '.#.' + str(time)

        # For data read in by proc, e.g., velo, visc, comp_nd use this form:
        file_data = Core_Citcom.read_proc_files_to_cap_list( master_d['pid_d'], file_format )

        # the next few diagnostic messages show how the data is reduced with each step
        print( now(), 'main: len(file_data) = ', len(file_data) )

        # flatten the field data since we don't care about specific cap numbers for a single level
        file_data = Core_Util.flatten_nested_structure( file_data )

        # get the specific data column for this field_name
        field_data = [line[field_column] for line in file_data]
        print( now(), 'main: len(field_data) = ', len(field_data) )

        # slice out the values for this level 
        field_slice = field_data[level::nodez]
        print( now(), 'main: len(field_slice) = ', len(field_slice) )


        # 
        # Creating an .xyz file 
        #

        # Some fields will require scaling: use the NumPy functions on slices:
        if field_name == 'visc': field_slice = np.log10( field_slice )

        # Assemble the coordinate data with the field data to create a .xyz file
        xyz_data = np.column_stack( (lon, lat, field_slice) )
        # create the xyz file name from other filename components:
        xyz_filename = datafile + '.' + field_name + '.' + str(depth_km) + 'km.' + str(time) + '.xyz'
        # write the file
        np.savetxt( xyz_filename, xyz_data, fmt='%f %f %f' )
        print( now(), 'main: write: xyz_filename =', xyz_filename)

 
        # 
        # Creating a grid file
        # 

        # Set the region based on the the model run:
        if nproc_surf == 12:
            R = 'g'
        else:
            R = str(pid_d['lon_min']) + '/' + str(pid_d['lon_max']) + '/'
            R += str(pid_d['lat_min']) + '/' + str(pid_d['lat_max'])

        # Set some defaults for the gridding process
        blockmedian_I = '0.1'
        surface_I = '0.1'

        # use Core_GMT.callgmt() to create the median file 
        median_xyz_filename = xyz_filename.rstrip('xyz') + 'median.xyz'
        args = xyz_filename + ' -I' + str(blockmedian_I) + ' -R' + R
        Core_GMT.callgmt( 'blockmedian', args, '', '>', median_xyz_filename )

        # Use Core_GMT to create the grid with required arguments ... 
        args = median_xyz_filename + ' -I' + str(surface_I) + ' -R' + R 
        # ... and with any optional argumetns passed in via the control file sub section_d 
        if 'Ll' in section_d:
            cmd += ' -Ll' + str(control_d[s]['Ll'])
        if 'Lu' in section_d:
            cmd += ' -Lu' + str(control_d[s]['Lu'])
        if 'T' in section_d:
            cmd += ' -T' + str(control_d[s]['T'])
        grid_filename = datafile + '.' + field_name + '.' + str(depth_km) + 'km.' + str(time) + '.grd'
        Core_GMT.callgmt( 'surface', args, '', '', ' -G' + grid_filename )

        # 
        # Creating the Map
        # 

        # Get the GPlates exported line data for this age 
        # be sure to truncate age to nearest int and make a string for the file name 
        age = str( int(age) )

        # Get the base path for gplates line data, as set in the geo framework defaults file:
        xy_path = master_d['geoframe_d']['gplates_line_dir']

        # get the full path to the line data:
        xy_filename = xy_path + '/' + 'topology_platepolygons_' + age + '.00Ma.xy' 
        print( now(), 'main: xy_filename = ', xy_filename)

        # make a plot of the grid 'TEST.ps'
        Core_GMT.plot_grid( grid_filename, xy_filename, R )
                
    # end of loop over sub section dictionary 

    # exit the script
    sys.exit()
#=====================================================================
#=====================================================================
def make_example_config_file( ):
    '''print to standard out an example configuration file for this script'''

    text = '''#=====================================================================
# config.cfg file for the simple_map.py script
#
# Copy this file to the top level of your citcom run directory,
# and change the values as needed.
#

# Set full, or local path to the pid file:
pid_file = /home/mturner/sample_data/global/pid32395.cfg

# Each subsection block will produce a separate map
[Grid_1]
time = 2700
level = 63
field = temp

# 
#=====================================================================
'''
    print( text )
#=====================================================================
#=====================================================================
if __name__ == "__main__":

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
#=====================================================================
#=====================================================================
# END

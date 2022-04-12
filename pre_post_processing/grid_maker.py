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
# grid_maker.py
#=====================================================================
# This script is a general purpose tool to process Citcoms output data
# into one ore more GMT style .grd format data files.  
# Please see the usage() function below, and the sample configuration 
# file: /sample_data/grid_maker.cfg for more info.
#=====================================================================
#=====================================================================
import sys, string, os
import numpy as np
#=====================================================================
import Core_Citcom
import Core_GMT
import Core_Util
from Core_Util import now

#from scipy.io import netcdf
#import netCDF4
#from netCDF4 import Dataset


# FIXME: turn these False when stable
#Core_Util.verbose = True 
#Core_Citcom.verbose = True 
Core_GMT.verbose = True

#=====================================================================
#=====================================================================
def usage():
    '''print usage message, and exit'''

    print('''usage: grid_maker.py [-e] configuration_file.cfg

Options and arguments:
  
-e : if the optional -e argument is given this script will print to standard out an example configuration control file.
   The parameter values in the example config.cfg file may need to be edited or commented out depending on intended use.

'configuration_file.cfg' : is a geodynamic framework formatted control file, with at least these entries: 

    pid_file = /path/to/a/citcoms_pid_file # the path of a citcoms pid0000.cfg file 
    time_spec = multi-value specification (single value, comma delimted list, start/stop/step trio),
    level_spec = multi-value specification (single value, comma delimted list, start/stop/step trio),

and at least one sub-section:

    [Subsection], where 'Subsection' may be any string, followed by:
    field = standard Citcom field name ('temp', 'visc', 'comp', etc. - see Core_Citcom.py for more info)
      
and where each sub-section may have one or more of the following optional entries:

     dimensional = True to also generate a dimensionalized grid 
     blockmedian_I = value to pass to GMT blockmedian -I option
     surface_I = value to pass to GMT surface -I option

See the example config.cfg file for more info.
''')
    sys.exit()
#=====================================================================
#=====================================================================
def main():
    print( now(), 'grid_maker.py')

    # get the .cfg file as a dictionary
    control_d = Core_Util.parse_configuration_file( sys.argv[1], False, False )
    Core_Util.tree_print( control_d )

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
    # get basic info about the model run
    datadir       = pid_d['datadir']
    datafile      = pid_d['datafile']
    start_age     = pid_d['start_age']
    output_format = pid_d['output_format']

    depth_list = coor_d['depth_km']
    nodez      = pid_d['nodez']
    nproc_surf = pid_d['nproc_surf']

    # Check how to read and parse the time spec:
    read_time_d = True
    #read_time_d = False
    
    # Compute the timesteps to process
    if read_time_d : 
        time_spec_d = Core_Citcom.get_time_spec_dictionary(control_d['time_spec'], master_d['time_d'])
        
    else :
        time_spec_d = Core_Citcom.get_time_spec_dictionary(control_d['time_spec'])
    print ( now(), 'grid_maker.py: time_spec_d = ')
    Core_Util.tree_print( time_spec_d )

    # levels to process 
    level_spec_d = Core_Util.get_spec_dictionary( control_d['level_spec'] )
    print ( now(), 'grid_maker.py: level_spec_d = ')
    Core_Util.tree_print( level_spec_d )

    # Get coordinate data 
    lon = []
    lat = []
    
    # Check for existing coordinate data
    lon_file_cache = '_cache_lon_coords.txt'
    lat_file_cache = '_cache_lat_coords.txt'

    if os.path.exists( lon_file_cache ) and os.path.exists( lat_file_cache ) :
        print( now(), 'grid_maker.py: loadtxt: ', lon_file_cache )
        print( now(), 'grid_maker.py: loadtxt: ', lat_file_cache )
        lon = np.loadtxt( lon_file_cache )
        lat = np.loadtxt( lat_file_cache )
    else : 
        # gets lon, lat for one depth because these are depth-independent
        coord_file_format = control_d.get('coord_dir','') + '/%(datafile)s.coord.#' % vars()
        coord = Core_Citcom.read_citcom_surface_coor( master_d['pid_d'], coord_file_format )

        # flatten data since we don't care about specific cap numbers for the loop over depth
        coord = Core_Util.flatten_nested_structure( coord )

        # extract data from tuples and make into numpy array
        lon = [line[0] for line in coord]
        lat = [line[1] for line in coord]

        # save the lat data 
        np.savetxt( lon_file_cache, lon, fmt='%f' )
        np.savetxt( lat_file_cache, lat, fmt='%f' )

    # end of get coords
    print( now(), 'grid_maker.py: len(lon) = ', len(lon) )
    print( now(), 'grid_maker.py: len(lat) = ', len(lat) )

    #
    # Main looping, first over times, then sections, then levels
    # 

    # Variables that will be updated each loop:
    # age_Ma will be a zero padded string value used for filenames and reporting 
    # depth will be a zero padded string value used for filenames and reporting 

    # Variables to hold data for all grids created
    # grid_list is a list of tuples: (grid_filename, age_Ma) 
    grid_list = []

    print(now(), '=========================================================================')
    print(now(), 'grid_maker.py: Main looping, first over times, then sections, then levels')
    print(now(), '=========================================================================')

    # Loop over times
    for tt, time in enumerate( time_spec_d['time_list'] ) :
 
        print( now(), 'grid_maker.py: Processing time = ', time) 
        #print( now(), 'grid_maker.py: Processing time = ', tt) 
        if 'Ma' in time:
 
            # strip off units and make a number
            time = float( time.replace('Ma', '') )

            # determine what time steps are available for this age 
            # NOTE: 'temp' is requried to set which output files to check 
            found_d = Core_Citcom.find_available_timestep_from_age( master_d, 'temp', time )
            print( now(), 'grid_maker.py: WARNING: Adjusting times to match available data:')
            print( now(), '  request_age =', found_d['request_age'], '; request_timestep =', found_d['request_timestep'], '; request_runtime =', found_d['request_runtime'])
            print( now(), '  found_age =', found_d['found_age'], '; found_timestep =', found_d['found_timestep'], '; found_runtime =', found_d['found_runtime'])

            # set variables for subsequent loops
            timestep = found_d['found_timestep']
            runtime_Myr = found_d['found_runtime']

            # convert the found age to an int
            age_Ma = int(np.around( found_d['found_age'] ) )

            # make a string and pad with zeros 
            age_Ma = '%03d' % age_Ma


        else:

            time = float( time ) 
             
            # determine what time steps are available for this timestep 
            # NOTE: 'temp' is requried to set which output files to check 

            found_d = Core_Citcom.find_available_timestep_from_timestep( master_d, 'temp', time )

            print( now(), 'grid_maker.py: WARNING: Adjusting times to match available data:')
            print( now(), '  request_age =', found_d['request_age'], '; request_timestep =', found_d['request_timestep'], '; request_runtime =', found_d['request_runtime'])
            print( now(), '  found_age =', found_d['found_age'], '; found_timestep =', found_d['found_timestep'], '; found_runtime =', found_d['found_runtime'])

            # set variables for subsequent loops
            timestep = found_d['found_timestep']
            runtime_Myr = found_d['found_runtime']

            # convert the found age to an int
            age_Ma = int(np.around( found_d['found_age'] ) )

            # make a string and pad with zeros 
            age_Ma = '%03d' % age_Ma

        # report on integer age
        print( now(), '  age_Ma =', age_Ma)

        # empty file_data
        file_data = []
   
        # cache for the file_format
        file_format_cache = ''

        # Loop over sections (fields) 
        for ss, s in enumerate (control_d['_SECTIONS_'] ) :

                # FIXME: this extra indent is probably from when sections loop was inside level loop ? 

                print( now(), 'grid_maker.py: Processing section = ', s) 

                # check for required parameter 'field'
                if not 'field' in control_d[s] :
                   print('ERROR: Required parameter "field" missing from section.')
                   print('       Skipping this section.')
                   continue # to next section
 
                # get the field name 
                field_name = control_d[s]['field']

                # check for compound field
                field_name_req = ''
                if field_name == 'horiz_vmag':
                    # save the requested name
                    field_name_req = field_name
                    # reset to get one component 
                    field_name = 'vx'

                print('')
                print( now(), 'grid_maker.py: Processing: field =', field_name) 

                # set the region
                if nproc_surf == 12:
                    grid_R = 'g'
                    # optionally adjust the lon bounds of the grid to -180/180
                    if 'shift_lon' in control_d :
                        print( now(), 'grid_maker.py: grid_R set to to "d" : -180/+180/-90/90')
                        grid_R = 'd'
                    else :
                        print( now(), 'grid_maker.py: grid_R set to to "g" : 0/360/-90/90')
                else:
                    grid_R  = str(pid_d['lon_min']) + '/' + str(pid_d['lon_max']) + '/'
                    grid_R += str(pid_d['lat_min']) + '/' + str(pid_d['lat_max'])
  
                # get the data file name specifics for this field 
                file_name_component = Core_Citcom.field_to_file_map[field_name]['file']
                print( now(), 'grid_maker.py: file_name_component = ', file_name_component )

                # get the data file column name specifics for this field 
                field_column = Core_Citcom.field_to_file_map[field_name]['column']
                print( now(), 'grid_maker.py: field_column = ', field_column )

                # create the total citcoms data filenames to read 
                file_format = ''
                
                # check for various data dirs 
                if os.path.exists( datadir + '/0/') :
                    print( now(), 'grid_maker.py: path found = ', datadir + '/0/' )
                    file_format = datadir + '/#/' + datafile + '.' + file_name_component + '.#.' + str(timestep)

                elif os.path.exists( datadir + '/' ) :
                    print( now(), 'grid_maker.py: path found = ', datadir + '/' )
                    file_format = datadir + '/' + datafile + '.' + file_name_component + '.#.' + str(timestep)

                elif os.path.exists('data') :
                    print( now(), 'grid_maker.py: path found = ', 'data' )
                    file_format = './data/#/' + datafile + '.' + file_name_component + '.#.' + str(timestep)

                elif os.path.exists('Data') :
                    print( now(), 'grid_maker.py: path found = ', 'Data' )
                    file_format = './Data/#/' + datafile + '.' + file_name_component + '.#.' + str(timestep)
                    
                # Added path to dynamic topography restart data - RC
                elif os.path.exists('Age'+str(start_age)+'Ma') :
                    print( now(), 'grid_maker.py: path found = ', 'Age'+str(start_age)+'Ma' )
                    file_format = './Age'+str(start_age)+'Ma/#/' + datafile + '.' + file_name_component + '.#.'+ str(timestep)
                  
                # report error 
                else :
                    print( now() )
                    print('ERROR: Cannot find output data.')
                    print('       Skipping this section.')
                    print( now(), 'grid_maker.py: file_format = ', file_format)
                    continue # to next section

                print( now(), 'grid_maker.py: file_format = ', file_format )

                # check if this file data has already been read in 
                if not file_format == file_format_cache: 

                    # read data by proc, e.g., velo, visc, comp_nd, surf, botm 
                    file_data = Core_Citcom.read_proc_files_to_cap_list( master_d['pid_d'], file_format, field_name)
                    # flatten data since we don't care about specific cap numbers for the loop over levels/depths
                    file_data = Core_Util.flatten_nested_structure( file_data )
                    print( now(), 'grid_maker.py: len(file_data) = ', len(file_data) )

                    # update cache for next pass in loop over fields
                    file_format_cache = file_format

                # Get the specific column for this field_name
                field_data = np.array( [ line[field_column] for line in file_data ] )
                print( now(), 'grid_maker.py:  len(field_data) = ', len(field_data) )

                # Check for compound field
                if field_name_req == 'horiz_vmag':
                    
                    # Get the second component data ('vy')
                    field_column = 1
                    # read data by proc, e.g., velo, visc, comp_nd, surf, botm 
                    file_data2 = Core_Citcom.read_proc_files_to_cap_list( master_d['pid_d'], file_format, field_name)
                    # flatten data since we don't care about specific cap numbers for the loop over levels/depths
                    file_data2 = Core_Util.flatten_nested_structure( file_data2 )
                    print( now(), 'grid_maker.py: len(file_data2) = ', len(file_data2) )
                    field_data2 = np.array( [ line[field_column] for line in file_data2 ] )
                    print( now(), 'grid_maker.py:  len(field_data2) = ', len(field_data) )

                    # combine the data and rest the main variable
                    field_data3 = np.hypot( field_data, field_data2)
                    field_data = field_data3

                    # put back field name to requested name
                    field_name = field_name_req 
                # end if check on compound field


                print( now(), 'grid_maker.py:  len(field_data) = ', len(field_data) )
                print( now() )
               
                #
                # Loop over levels 
                #
                for ll, level in enumerate( level_spec_d['list'] ) :

                    print( now(), 'grid_maker.py: Processing level = ', level) 

                    # ensure level is an int value 
                    level = int(level)
                    depth = int(depth_list[level])
                    # pad the depth value 
                    depth = '%04d' % depth

                    print( now(), '------------------------------------------------------------------------------')
                    print( now(), 'grid_maker.py: tt,ss,ll = ', tt, ',', ss, ',', ll, ';')
                    print( now(), 'grid_maker.py: summary for', s, ': timestep =', timestep, '; age =', age_Ma, '; runtime_Myr =', runtime_Myr, '; level =', level, '; depth =', depth, ' km; field_name =', field_name)
                    print( now(), '------------------------------------------------------------------------------')

                    if field_name.startswith('vertical_'):
                        # perform a z slice for citcom data 
                        field_slice = field_data[level::nodez] # FIXME : how to get a v slice 
                        xyz_filename = datafile + '-' + field_name + '-' + str(age_Ma) + 'Ma-' + str(depth) + 'km.xyz'
                    else:
                        # perform a z slice for citcom data 
                        field_slice = field_data[level::nodez]
                        #xyz_filename = datafile + '-' + field_name + '-' + str(timestep) + '-' + str(depth) + '.xyz'
                        xyz_filename = datafile + '-' + field_name + '-' + str(age_Ma) + 'Ma-' + str(depth) + 'km.xyz'

                    print( now(), 'grid_maker.py: xyz_filename =', xyz_filename)
            
                    if field_name == 'visc': field_slice = np.log10( field_slice )

                    print( now(), 'grid_maker.py: type(field_slice) = ', type(field_slice) )
                    print( now(), 'grid_maker.py:  len(field_slice) = ', len(field_slice) )
                    print( now() )


                    # create the xyz data
                    xyz_data = np.column_stack( (lon, lat, field_slice) )
                    np.savetxt( xyz_filename, xyz_data, fmt='%f %f %f' )

                    #print( now(), 'grid_maker.py: type(xyz_data) = ', type(xyz_data) )
                    #print( now(), 'grid_maker.py:  len(xyz_data) = ', len(xyz_data) )
                    #print( now() )

                    # recast the slice 
                    #fs = np.array( field_slice )  
                    #fs.shape = ( len(lat), len(lon) )
                    #print( now(), 'grid_maker.py: type(fs) = ', type(field_slice) )
                    #print( now(), 'grid_maker.py:  len(fs) = ', len(field_slice) )
                    #print( now() )

                    # check for a grid_R 
                    if 'R' in control_d[s] :
                        grid_R = control_d[s]['R']

                    # create the median file 
                    median_xyz_filename = xyz_filename.rstrip('xyz') + 'median.xyz'

                    blockmedian_I = control_d[s].get('blockmedian_I', '0.5')
                    cmd = xyz_filename + ' -I' + str(blockmedian_I) + ' -R' + grid_R

                    Core_GMT.callgmt( 'blockmedian', cmd, '', '>', median_xyz_filename )

                    # get a T value for median file 
                    if not 'Ll' in control_d[s] or not 'Lu' in control_d[s]:
                        T = Core_GMT.get_T_from_minmax( median_xyz_filename )
                    else:
                        dt = (control_d[s]['Lu']-control_d[s]['Ll'])/10
                        T = '-T' + str(control_d[s]['Ll']) + '/'
                        T += str(control_d[s]['Lu']) + '/' + str(dt)

                    print( now(), 'grid_maker.py: T =', T)

                   

                    # create the grid
                    grid_filename = xyz_filename.rstrip('xyz') + 'nc'

                    surface_I = control_d[s].get('surface_I', '0.25')
                    cmd = median_xyz_filename + ' -I' + str(surface_I) + ' -R' + grid_R 

                    if 'Ll' in control_d[s]:
                        cmd += ' -Ll' + str(control_d[s]['Ll'])
                    if 'Lu' in control_d[s]:
                        cmd += ' -Lu' + str(control_d[s]['Lu'])
                    if 'T' in control_d[s]:
                        cmd += ' -T' + str(control_d[s]['T'])

                    #opt_a = 
                    Core_GMT.callgmt( 'surface', cmd, '', '', ' -G' + grid_filename )

                    # label the variables
			
                    # −Dxname/yname/zname/scale/offset/title/remark
                    cmd = grid_filename + ' -D/=/=/' + str(field_name) + '/=/=/' + str(field_name) + '/' + str(field_name)
                    Core_GMT.callgmt( 'grdedit', cmd, '', '', '')
            
                    # Dimensionalize grid   

                    if control_d[s].get('dimensional'):
                        print( now(), 'grid_maker.py: dimensional = ', control_d[s]['dimensional'])
                        dim_grid_name = grid_filename.replace('.nc', '.dimensional.nc')
                        Core_Citcom.dimensionalize_grid(pid_file, field_name, grid_filename, dim_grid_name)

                        # FIXME: for dynamic topo remove  mean 
                        # grdinfo to get mean ; see To_Refactor for example 

                    # save this grid and its age in a list
                    if control_d[s].get('dimensional'):
                        grid_list.append( (dim_grid_name, age_Ma) )
                    else: 
                        grid_list.append( (grid_filename, age_Ma) )



                    # Optional step to transform grid to plate frame
                    if 'make_plate_frame_grid' in control_d :
                        cmd = 'frame_change_pygplates.py %(age_Ma)s %(grid_filename)s %(grid_R)s' % vars()
                        print(now(), 'grid_maker.py: cmd =', cmd)
                        os.system(cmd)


                    # Assoicate this grid with GPlates exported line data in .xy format:
                    # compute age value 
                    age_float = 0.0

                    # time_list values for citcom data uses timesteps; get age 
                    time_triple = Core_Citcom.get_time_triple_from_timestep(master_d['time_d']['triples'], timestep)
                    age_float = time_triple[1]

                    # truncate to nearest int and make a string for the gplates .xy file name 
                    if age_float < 0: age_float = 0.0
                    xy_path = master_d['geoframe_d']['gplates_line_dir']
                    xy_filename = xy_path + '/' + 'topology_platepolygons_' + str(int(age_float)) + '.00Ma.xy' 
                    print( now(), 'grid_maker.py: xy_filename = ', xy_filename)
    

                    # Make a plot of the grids
                    J = 'X5/3' #'R0/6'
                    #J = 'M5/3'
                    if 'J' in control_d[s] :
                        J = control_d[s]['J']

                    C = 'polar'
                    if 'C' in control_d[s] :
                        C = control_d[s]['C']
       
                    # citcoms 
                    # plot non-dimensional grid
                    Core_GMT.plot_grid( grid_filename, xy_filename, grid_R, T, J, C)

                    # also plot dimensional grid 
                    if control_d[s].get('dimensional') :
                        print( now(), 'grid_maker.py: plotting dimensional = ', control_d[s]['dimensional'])
                        dim_grid_name = grid_filename.replace('.nc', '.dimensional.nc')
                        T = Core_GMT.get_T_from_grdinfo( dim_grid_name )
                        Core_GMT.plot_grid( dim_grid_name, xy_filename, grid_R, T, J)

                    # plot plate frame grid 
                    if 'make_plate_frame_grid' in control_d :
                        plateframe_grid_name = grid_filename.replace('.nc', '-plateframe.nc')
                        xy_filename = ''
                        xy_path = master_d['geoframe_d']['gplates_line_dir']
                        # present day plate outlines : use '0' 
                        xy_filename = xy_path + '/' + 'topology_platepolygons_0.00Ma.xy' 
                        print( now(), 'grid_maker.py: xy_filename = ', xy_filename)

                        T = Core_GMT.get_T_from_grdinfo( plateframe_grid_name )
                        print( now(), 'grid_maker.py: T =', T)
                        Core_GMT.plot_grid( plateframe_grid_name, xy_filename, grid_R, T, J)
                    # end of plotting 
                
                # end of loop over levels 

            # end of loop over sections

    # end of loop over times

#=====================================================================
#=====================================================================
# SAVE This code for reference:
#                    # optionally adjust the lon bounds of the grid to -180/180
#                    #if 'shift_lon' in control_d : 
#                    #    print( now(), 'grid_maker.py: shifting values to -180/+180')
#                    #    arg = grid_filename
#                    #    opts = {'R' : 'd', 'S' : '' }
#                    #    Core_GMT.callgmt('grdedit', arg, opts)
#=====================================================================
#=====================================================================
def make_example_config_file( ):
    '''print to standard out an example configuration file for this script'''

    text = '''#=====================================================================
# config.cfg file for the grid_maker.py script
# This example has information on creating grids for Citcom data.
# ==============================================================================
# Set the basic model coordinate information common to both source types:

# set path to the model pid file 
pid_file = pid18637.cfg 

# CitcomS coordinate files by processor (i.e. [datafile].coord.[proc])
# first, look in this user-specified directory for all files
coord_dir = coord

# second, look in data/%RANK

# NOTE: grid_maker.py will fail if coord files cannot be located

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
time_spec = 0Ma/10Ma/2Ma

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
#=====================================================================
'''
    print( text )
#=====================================================================
#=====================================================================
if __name__ == "__main__":

    # print ( str(sys.version_info) ) 

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

#!/usr/bin/env python
#=====================================================================
#                Geodynamic Framework Scripts for 
#         Preprocessing, Data Assimilation, and Postprocessing
#
#                 AUTHORS: Mark Turner, Dan J. Bower
#                  ---------------------------------
#             (c) California Institute of Technology 2015
#                  ---------------------------------
#                        ALL RIGHTS RESERVED
#=====================================================================
#=====================================================================
# grid_maker_gplates.py
#=====================================================================
# This script is a general purpose tool to process Citcoms output data
# into one ore more GMT style .grd format data files.  
# Please see the usage() function below, and the sample configuration 
# file: /sample_data/grid_maker_gplates.cfg for more info.
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

    print('''usage: grid_maker_gplates.py [-e] configuration_file.cfg

Options and arguments:
  
-e : if the optional -e argument is given this script will print to standard out an example configuration control file.
   The parameter values in the example config.cfg file may need to be edited or commented out depending on intended use.

'configuration_file.cfg' : is a geodynamic framework formatted control file, with at least these entries: 

    time_spec = multi-value specification (single value, comma delimted list, start/stop/step trio),

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
    print( now(), 'grid_maker_gplates.py')

    # get the .cfg file as a dictionary
    control_d = Core_Util.parse_configuration_file( sys.argv[1], False, False )
    Core_Util.tree_print( control_d )


    time_spec_d = Core_Citcom.get_time_spec_dictionary(control_d['time_spec'])
    print ( now(), 'grid_maker_gplates.py: time_spec_d = ')
    Core_Util.tree_print( time_spec_d )


    # Get the coordinate data from the 0 Ma files
    print ( now(), 'grid_maker_gplates.py: get coordinate data from .xy files:')
    lon = []
    lat = []
    for i in range( control_d['nproc_surf'] ) :
        # get the lat lon from the .xy file
        vel_xy_filename = control_d['velocity_prefix'] + '0.%(i)s.xy' % vars()
        print ( now(), 'grid_maker_gplates.py: vel_xy_filename = ', vel_xy_filename)
        i_lat, i_lon = np.loadtxt( vel_xy_filename , usecols=(0,1), unpack=True )
        lat.append( i_lat )
        lon.append( i_lon )

    lon = Core_Util.flatten_nested_structure( lon )
    lat = Core_Util.flatten_nested_structure( lat )

    print( now(), 'grid_maker_gplates.py: len(lon) = ', len(lon) )
    print( now(), 'grid_maker_gplates.py: len(lat) = ', len(lat) )

    #
    # Main looping, first over times, then sections, then levels
    # 

    # Variables that will be updated each loop:
    # time will be a zero padded string value used for filenames and reporting 
    # depth will be a zero padded string value used for filenames and reporting 

    print(now(), '=========================================================================')
    print(now(), 'grid_maker_gplates.py: Main looping, first over times, then sections, then levels')
    print(now(), '=========================================================================')

    # Loop over times
    for tt, time in enumerate( time_spec_d['time_list'] ) :
 
        print( now(), 'grid_maker_gplates.py: Processing time = ', time) 

        # empty file_data
        file_data = []

        # cache for the file_format
        file_format_cache = ''

        # Loop over sections (fields) 
        for ss, s in enumerate (control_d['_SECTIONS_'] ) :

                # FIXME: this extra indent is probably from when sections loop was inside level loop ? 
                # FIXME: this extra indent is probably from when sections loop was inside level loop ? 

                print( now(), 'grid_maker_gplates.py: Processing section = ', s) 

                # check for required parameter 'field'
                if not 'field' in control_d[s] :
                   print('ERROR: Required parameter "field" missing from section.')
                   print('       Skipping this section.')
                   continue # to next section
 
                # get the field name 
                field_name = control_d[s]['field']

                print('')
                print( now(), 'grid_maker_gplates.py: Processing: field =', field_name) 

                # reset region to use -Rg for gplates
                grid_R = 'g'

                if 'shift_lon' in control_d :
                    print( now(), 'grid_maker_gplates.py: grid_R set to to "d" : -180/+180/-90/90')
                    grid_R = 'd'
                else:
                    print( now(), 'grid_maker_gplates.py: grid_R set to to "g" : 0/360/-90/90')
                 
                # get the data file name specifics for this field 
                file_name_component = Core_Citcom.field_to_file_map[field_name]['file']
                print( now(), 'grid_maker_gplates.py: file_name_component = ', file_name_component )

                # get the data file column name specifics for this field 
                field_column = Core_Citcom.field_to_file_map[field_name]['column']
                print( now(), 'grid_maker_gplates.py: field_column = ', field_column )

               
                # remove potential zero padding from age values
                time = time.replace('Ma', '')
                # process data from GPlates 
                file_format = control_d['velocity_prefix'] + '%(time)s.#' % vars()

                print( now(), 'grid_maker_gplates.py: file_format = ', file_format )

                # read data in by cap
                file_data = Core_Citcom.read_cap_files_to_cap_list( control_d, file_format )
   
                # flatten data since we don't care about specific cap numbers for the loop over levels/depths
                file_data = Core_Util.flatten_nested_structure( file_data )
                print( now(), 'grid_maker_gplates.py: len(file_data) = ', len(file_data) )

                # Get the specific column for this field_name
                field_data = np.array( [ line[field_column] for line in file_data ] )

                print( now(), 'grid_maker_gplates.py: type(field_data) = ', type(field_data) )
                print( now(), 'grid_maker_gplates.py:  len(field_data) = ', len(field_data) )
                print( now() )

                # check for gplates_vmag
                if field_name == 'gplates_vmag' :
                    # read the vy data from col 1 
                    field_data_vy = [line[1] for line in file_data]
                    # compute the magnitude 
                    vx_a = np.array(field_data)
                    vy_a = np.array(field_data_vy)
                    vmag_a = np.hypot( vx_a, vy_a )
                    # convert back to list
                    field_data = vmag_a.tolist()

                print( now(), '------------------------------------------------------------------------------')
                print( now(), 'grid_maker_gplates.py: tt,ss = ', tt, ',', ss, ';')
                print( now(), 'grid_maker_gplates.py: summary for', s, ': time =', time, '; field_name =', field_name)
                print( now(), '------------------------------------------------------------------------------')

                depth = 0
                field_slice = field_data
                xyz_filename = field_name + '-' + str(time) + '-' + str(depth) + '.xyz'

                print( now(), 'grid_maker_gplates.py: xyz_filename =', xyz_filename)
            
                print( now(), 'grid_maker_gplates.py: type(field_slice) = ', type(field_slice) )
                print( now(), 'grid_maker_gplates.py:  len(field_slice) = ', len(field_slice) )
                print( now() )

                # create the xyz data
                xyz_data = np.column_stack( (lon, lat, field_slice) )
                np.savetxt( xyz_filename, xyz_data, fmt='%f %f %f' )

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

                print( now(), 'grid_maker_gplates.py: T =', T)

                # create the grid
                grid_filename = xyz_filename.rstrip('xyz') + 'grd'

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
        
                # Assoicate this grid with GPlates exported line data in .xy format:
                # compute age value 
                age_float = 0.0
                if field_name.startswith('gplates_'):
                    # time_list value for gplates data is set with age values 
                    age_float = float( time )

                # truncate to nearest int and make a string for the gplates .xy file name 
                geoframe_d = Core_Util.parse_geodynamic_framework_defaults()

                if age_float < 0: age_float = 0.0
                xy_path = geoframe_d['gplates_line_dir']
                xy_filename = xy_path + '/' + 'topology_platepolygons_' + str(int(age_float)) + '.00Ma.xy' 
                print( now(), 'grid_maker_gplates.py: xy_filename = ', xy_filename)

                # Make a plot of the grids
                J = 'X5/3' #'R0/6'
                #J = 'M5/3'
                if 'J' in control_d[s] :
                    J = control_d[s]['J']

                C = 'polar'
                if 'C' in control_d[s] :
                    C = control_d[s]['C']
   
                # gplates 
                Core_GMT.plot_grid( grid_filename, xy_filename, grid_R, '-T-10/10/1' )
                # end of plotting 
                
                # Optional step to transform grid to plate frame
                if 'make_plate_frame_grid' in control_d :
                    cmd = 'frame_change_pygplates.py %(time)s %(grid_filename)s %(grid_R)s' % vars()
                    print(now(), 'grid_maker_gplates.py: cmd =', cmd)
                    os.system(cmd)
                    filename = grid_filename.replace('.grd', '-plateframe.grd')
                    Core_GMT.plot_grid( filename, xy_filename, grid_R, '-T-10/10/1' )


    # end of loop over times

#=====================================================================
#=====================================================================
# SAVE This code for reference:
#                    # optionally adjust the lon bounds of the grid to -180/180
#                    #if 'shift_lon' in control_d : 
#                    #    print( now(), 'grid_maker_gplates.py: shifting values to -180/+180')
#                    #    arg = grid_filename
#                    #    opts = {'R' : 'd', 'S' : '' }
#                    #    Core_GMT.callgmt('grdedit', arg, opts)
#=====================================================================
#=====================================================================
def make_example_config_file( ):
    '''print to standard out an example configuration file for this script'''

    text = '''#=====================================================================
# config.cfg file for the grid_maker_gplates.py script
# 
# This example has information on creating grids for GPlates velocity data,
# ==============================================================================
# Required global settings:
# set the same values used in GPlates to build the mesh cap files:
nproc_surf=12
nodex=10
nodey=10
nodez=1
velocity_prefix=bvel

# Optional global settings

# If 'shift_lon' is set to True, then the grids will have data in the -180/+180 longitude range
# The default is for data in the 0/360 longitude range.
# shift_lon = True

# If 'make_plate_frame_grid' is set to True, then this script with produce additional data on the plate frame
#make_plate_frame_grid = True

# ==============================================================================
# GPlates : use values in Ma, and include the Ma suffix
# (note: 'Ma' will be stripped off to compute int values)

# single value:
time_spec = 0Ma

# comma separated list:
#time_spec = 0,100,200Ma

# range of values: start/stop/step
#time_spec = 0/230/1Ma

# ==============================================================================
# Set the fields to grid 
#
# Each field will be a separate section, delimited by brackets [Section_1], 
# each field requires the field name, e.g. 
# field = temp
# Each field may set optional arguments to set GMT parameters.

[GP_VMAG]
field = gplates_vmag
blockmedian_I = 0.5
surface_I = 0.25
#
[GP_VY]
field = gplates_vy
blockmedian_I = 0.5
surface_I = 0.25
#
[GP_VX]
field = gplates_vx
blockmedian_I = 0.5
surface_I = 0.25
#
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
